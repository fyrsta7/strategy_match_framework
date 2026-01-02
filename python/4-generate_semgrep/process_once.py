import os
import json
import uuid
import time
import threading
import subprocess
from import_configs import global_config, generate_config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder

def print_info(message):
    """关键信息输出 - 始终显示"""
    print(message)

def print_verbose(message):
    """详细信息输出 - 仅在VERBOSE模式下显示"""
    if generate_config.VERBOSE:
        print(message)

def generate_unique_session_id(commit_num, prefix="SemgrepGen"):
    """生成唯一的会话ID"""
    timestamp = int(time.time() * 1000)  # 毫秒时间戳
    thread_id = threading.current_thread().ident
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{commit_num}_{timestamp}_{thread_id}_{unique_id}"

def should_continue_fixing(json_result, error_msg):
    """判断是否需要继续修复迭代"""
    # 如果有执行错误，需要继续修复
    if error_msg:
        return True
    
    # 如果没有JSON结果或没有errors，不需要修复
    if not json_result or not json_result.get("errors"):
        return False
    
    # 检查每个错误的级别
    for error in json_result.get("errors", []):
        level = error.get("level", "").lower()
        # 如果有严重错误，需要继续修复
        if level in ["fatal", "error"]:
            return True
    
    # 所有错误都是警告级别或更低，不需要继续修复
    return False

def create_isolated_llm_chain(use_langsmith=None):
    """创建完全独立的LLM链，不共享会话历史"""
    if use_langsmith is None:
        use_langsmith = generate_config.LLM_USE_LANGSMITH
    
    # 条件性设置环境变量
    if use_langsmith:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGSMITH_API_KEY"] = global_config.LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = "semgrep-optimization-agent"
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
    
    os.environ["OPENAI_API_KEY"] = global_config.xmcp_api_key_unlimit
    
    # 本地会话存储，只存在于当前函数调用期间
    local_store = {}
    
    def get_local_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in local_store:
            local_store[session_id] = ChatMessageHistory()
        
        if len(local_store[session_id].messages) > 4:
            x = local_store[session_id].messages.pop()
            local_store[session_id].messages.pop()
            local_store[session_id].messages.pop()
            local_store[session_id].messages.append(x)
        
        return local_store[session_id]
    
    # 系统提示
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in C/C++ program analysis and Semgrep rule creation. Your task is to identify performance optimization patterns in code changes and create Semgrep rules that can detect similar optimization opportunities in other codebases.
When writing Semgrep rules for C/C++, please ensure that:
1. The rule follows proper Semgrep YAML format with correct metadata including 'id', 'message', 'languages', 'severity', and 'pattern' fields.
2. The rule must support both C and C++ languages with 'languages: [cpp, c]' in the metadata.
3. The rule should detect patterns where code could be optimized similar to the provided diff.
4. The rule should focus on performance improvements rather than security issues.
5. Include helpful messages that explain the performance impact and suggested improvements.
6. Ensure that the generated YAML code is enclosed within triple backticks with 'yaml' language specification (```yaml).
7. The pattern should be syntactically correct for C/C++ and follow Semgrep pattern syntax.
8. Use appropriate Semgrep metavariables (like $VAR, $EXPR, $STMT) to make patterns generic.
9. Include severity level appropriate for performance optimizations (usually 'info' or 'warning').
Example Semgrep rule structure:
```yaml
rules:
  - id: performance-optimization-example
    message: "Performance optimization opportunity detected"
    languages: [cpp, c]
    severity: INFO
    pattern: |
      pattern-content-here
```"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    output_parser = StrOutputParser()
    llm = ChatOpenAI(
        model_name=global_config.xmcp_deepseek_model,
        temperature=generate_config.LLM_TEMPERATURE,
        openai_api_base=global_config.xmcp_base_url,
        verbose=generate_config.VERBOSE
    )
    chain = prompt | llm | output_parser
    
    conversational_rag_chain = RunnableWithMessageHistory(
        chain,
        get_local_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return conversational_rag_chain

def extract_yaml_from_output(output):
    """从LLM输出中提取YAML代码"""
    if "```yaml" in output:
        code = output.split("```yaml")[1].split("```")[0]
    elif "```yml" in output:
        code = output.split("```yml")[1].split("```")[0]
    elif "```" in output:
        code = output.split("```")[1].split("```")[0]
        if not code.strip().startswith(('rules:', 'id:', '-')):
            parts = output.split("```")
            for i in range(1, len(parts), 2):
                if parts[i].strip().startswith(('rules:', 'id:', '-')):
                    code = parts[i]
                    break
    else:
        code = output
    
    return code.strip()

def add_line_prefix(text):
    """为文本的每一行添加行号前缀"""
    lines = text.splitlines()
    cleaned_lines = [f"line {i + 1}: {line}" for (i, line) in enumerate(lines)]
    return "\n".join(cleaned_lines)

def run_semgrep_rule(yaml_file_path, before_path, json_output_path):
    """运行Semgrep规则并返回结果"""
    cmd = [
        "semgrep",
        f"--config={yaml_file_path}",
        before_path,
        "--json"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=generate_config.TIMEOUT
        )
        
        # 解析JSON输出
        if result.stdout.strip():
            try:
                json_result = json.loads(result.stdout)
                with open(json_output_path, 'w') as f:
                    json.dump(json_result, f, indent=generate_config.DEFAULT_OUTPUT_INDENT)
                return json_result, None
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON output: {e}\nStdout: {result.stdout}\nStderr: {result.stderr}"
                return None, error_msg
        else:
            # 创建默认结果结构
            default_result = {
                "version": "unknown",
                "results": [],
                "errors": [],
                "paths": {"scanned": [before_path]},
                "skipped_rules": []
            }
            
            if result.stderr.strip():
                default_result["errors"].append({
                    "code": result.returncode,
                    "level": "error",
                    "type": "Execution error",
                    "message": result.stderr
                })
            
            with open(json_output_path, 'w') as f:
                json.dump(default_result, f, indent=semgrep_config.DEFAULT_OUTPUT_INDENT)
            return default_result, None
            
    except subprocess.TimeoutExpired:
        error_msg = f"Semgrep execution timed out ({semgrep_config.TIMEOUT} seconds)"
        return None, error_msg
    except Exception as e:
        error_msg = f"Error running Semgrep: {str(e)}"
        return None, error_msg

def process_semgrep_generation(diff_content, before_path, target_yaml_path, 
                             commit_num, max_round=None, use_langsmith=None):
    """
    处理Semgrep规则生成的主要函数
    
    Args:
        diff_content: diff内容字符串
        before_path: 用于测试Semgrep规则的代码文件路径
        target_yaml_path: 目标YAML文件保存路径
        commit_num: commit编号
        max_round: 最大迭代轮数，默认使用config配置
        use_langsmith: 是否使用LangSmith，默认使用config配置
    
    Returns:
        tuple: (是否成功: bool, 错误信息: str|None)
    """
    if max_round is None:
        max_round = generate_config.LLM_MAX_GENERATION_ROUNDS
    if use_langsmith is None:
        use_langsmith = generate_config.LLM_USE_LANGSMITH
    
    print_info(f"\n==== Processing Semgrep rule generation ====")
    print_verbose(f"Target YAML file: {target_yaml_path}")
    print_verbose(f"Before file: {before_path}")
    print_verbose(f"Max rounds: {max_round}")
    print_verbose(f"Use LangSmith: {use_langsmith}")
    
    # 确保输出目录存在
    target_dir = os.path.dirname(target_yaml_path)
    os.makedirs(target_dir, exist_ok=True)
    
    # 生成JSON文件路径
    yaml_basename = os.path.splitext(os.path.basename(target_yaml_path))[0]
    json_output_path = os.path.join(target_dir, f"{yaml_basename}.json")
    
    # 生成唯一会话ID
    unique_session_id = generate_unique_session_id(commit_num)
    print_verbose(f"Session ID: {unique_session_id}")
    
    # 创建独立的LLM链
    try:
        conversational_rag_chain = create_isolated_llm_chain(use_langsmith)
    except Exception as e:
        error_msg = f"Error creating LLM chain: {e}"
        print_info(f"Error: {error_msg}")
        return False, error_msg
    
    print_verbose(f"Diff content preview (first 500 chars):")
    print_verbose(diff_content[:500] + "..." if len(diff_content) > 500 else diff_content)
    
    # 第一轮对话：分析优化模式
    question_1 = """What performance optimization pattern does the following diff implement? 
Analyze the changes and identify the optimization technique being applied. 
Do not write Semgrep rules at this time, just analyze the diff and explain the performance improvement."""
    
    print_info("=== First Round: Analyzing optimization pattern ===")
    try:
        analysis_result = conversational_rag_chain.invoke(
            {"input": question_1 + "\n\nDiff content:\n" + diff_content},
            config={"configurable": {"session_id": unique_session_id}},
        )
        print_verbose("Analysis result:")
        print_verbose(analysis_result)
    except Exception as e:
        error_msg = f"Error in first round analysis: {e}"
        print_info(f"Error: {error_msg}")
        return False, error_msg
    
    # 第二轮对话：生成初始Semgrep规则
    question_2 = """Write a Semgrep rule for C/C++ that can detect similar optimization opportunities in other codebases. 
The rule should identify code patterns that could benefit from the same optimization technique shown in the diff. 
Focus on creating a rule that helps developers find performance improvement opportunities.
Make sure the rule follows proper Semgrep YAML format, uses correct C/C++ pattern syntax, and supports both C and C++ languages with 'languages: [cpp, c]'."""
    
    print_info("=== Second Round: Generating initial Semgrep rule ===")
    try:
        output = conversational_rag_chain.invoke(
            {"input": question_2},
            config={"configurable": {"session_id": unique_session_id}},
        )
    except Exception as e:
        error_msg = f"Error in second round generation: {e}"
        print_info(f"Error: {error_msg}")
        return False, error_msg
    
    # 迭代修复循环
    for i in range(max_round):
        print_info(f"=== Attempt #{i + 1} ===")
        
        # 提取YAML代码
        yaml_code = extract_yaml_from_output(output)
        print_verbose(f"Generated YAML code:\n{yaml_code}")
        
        # 保存YAML文件（覆盖）
        with open(target_yaml_path, "w") as f:
            f.write(yaml_code)
        print_verbose(f"Saved YAML rule to: {target_yaml_path}")
        
        # 运行Semgrep规则
        print_verbose("Running Semgrep...")
        json_result, error_msg = run_semgrep_rule(target_yaml_path, before_path, json_output_path)
        
        # 使用新的错误级别判断逻辑
        if should_continue_fixing(json_result, error_msg):
            # 需要继续修复的情况
            if error_msg:
                print_verbose(f"Execution error: {error_msg}")
                fix_prompt = f"""There was an execution error when running the Semgrep rule:
Error: {error_msg}
The YAML rule you provided:
{add_line_prefix(yaml_code)}
Please fix the Semgrep rule to resolve this execution error. Provide a complete, corrected YAML rule that can run successfully."""
            
            elif json_result and json_result.get("errors"):
                print_verbose(f"Semgrep reported serious errors: {json.dumps(json_result, indent=2)}")
                fix_prompt = f"""The Semgrep rule execution completed but reported serious errors:
Full JSON result:
{json.dumps(json_result, indent=2)}
The YAML rule you provided:
{add_line_prefix(yaml_code)}
Please analyze the errors and fix the Semgrep rule. Focus on:
1. Correct YAML syntax and structure
2. Valid C/C++ pattern syntax for Semgrep
3. Proper use of metavariables
4. Correct rule metadata
Provide a complete, corrected YAML rule that addresses all reported errors."""
            
            # 发送修复请求
            print_verbose("Requesting fix from LLM...")
            try:
                output = conversational_rag_chain.invoke(
                    {"input": fix_prompt},
                    config={"configurable": {"session_id": unique_session_id}},
                )
            except Exception as e:
                error_msg = f"Error in iteration {i+1}: {e}"
                print_info(f"Error: {error_msg}")
                return False, error_msg
        
        else:
            # 可以接受当前结果的情况
            if json_result and json_result.get("errors"):
                # 有警告但可以接受
                warning_count = len(json_result.get("errors", []))
                print_info(f"Success! Semgrep rule executed successfully with {warning_count} minor warning(s).")
                print_verbose("Warning details:")
                for warning in json_result.get("errors", []):
                    print_verbose(f"  - Level: {warning.get('level', 'unknown')}, Message: {warning.get('message', 'No message')}")
            else:
                # 完全无错误
                print_info("Success! Semgrep rule executed without any errors.")
            
            print_info(f"Final rule saved as: {target_yaml_path}")
            print_info(f"Results saved as: {json_output_path}")
            if json_result:
                print_info(f"Results found: {len(json_result.get('results', []))}")
            return True, None
    
    error_msg = f"Failed to generate working Semgrep rule in {max_round} rounds"
    print_info(error_msg)
    return False, error_msg

def generate_semgrep_rule_from_file(diff_file_path, before_path, target_yaml_path, 
                                  commit_num, max_round=None, use_langsmith=None):
    """
    主要接口函数：从文件生成Semgrep规则
    
    Args:
        diff_file_path: commit diff文件路径
        before_path: 用于测试Semgrep规则的代码文件路径
        target_yaml_path: 目标YAML文件保存路径
        commit_num: commit编号
        max_round: 最大迭代轮数，默认使用config配置
        use_langsmith: 是否使用LangSmith，默认使用config配置
    
    Returns:
        tuple: (是否成功: bool, 错误信息: str|None)
    """
    # 读取diff内容
    try:
        with open(diff_file_path, 'r', encoding='utf-8') as f:
            diff_content = f.read()
    except Exception as e:
        error_msg = f"Error reading diff file {diff_file_path}: {e}"
        print_info(error_msg)
        return False, error_msg
    
    # 生成Semgrep规则
    return process_semgrep_generation(
        diff_content=diff_content,
        before_path=before_path,
        target_yaml_path=target_yaml_path,
        commit_num=commit_num,
        max_round=max_round,
        use_langsmith=use_langsmith
    )

def main():
    """主函数：使用默认参数处理测试案例"""
    # 默认测试参数
    COMMIT_NUM = 2
    BEFORE_FILE_NAME = "before.cc"  # 直接在脚本中定义
    
    # 使用config路径构建
    BASE_DIR = global_config.semgrep_path
    DIFF_FILE_PATH = f"{BASE_DIR}/commit_info/{COMMIT_NUM}/diff.txt"
    TARGET_YAML_PATH = f"{BASE_DIR}/yaml/{COMMIT_NUM}/0.yaml"
    BEFORE_PATH = f"{BASE_DIR}/commit_info/{COMMIT_NUM}/{BEFORE_FILE_NAME}"
    
    # 使用config配置
    MAX_ROUND = semgrep_config.LLM_MAX_GENERATION_ROUNDS
    USE_LANGSMITH = semgrep_config.LLM_USE_LANGSMITH
    
    print_info(f"=== Running main function with default parameters ===")
    print_verbose(f"Commit number: {COMMIT_NUM}")
    print_verbose(f"Diff file: {DIFF_FILE_PATH}")
    print_verbose(f"Before file: {BEFORE_PATH}")
    print_verbose(f"Target YAML file: {TARGET_YAML_PATH}")
    print_verbose(f"Max rounds: {MAX_ROUND}")
    print_verbose(f"Use LangSmith: {USE_LANGSMITH}")
    
    # 验证必要文件是否存在
    if not os.path.exists(DIFF_FILE_PATH):
        print_info(f"Error: Diff file not found: {DIFF_FILE_PATH}")
        return
    
    if not os.path.exists(BEFORE_PATH):
        print_info(f"Error: Before file not found: {BEFORE_PATH}")
        return
    
    # 生成Semgrep规则
    success, error_message = generate_semgrep_rule_from_file(
        diff_file_path=DIFF_FILE_PATH,
        before_path=BEFORE_PATH,
        target_yaml_path=TARGET_YAML_PATH,
        commit_num=COMMIT_NUM,
        max_round=MAX_ROUND,
        use_langsmith=USE_LANGSMITH
    )
    
    if success:
        print_info(f"\n==== SUCCESS ====")
        print_info(f"Successfully generated Semgrep rule: {TARGET_YAML_PATH}")
    else:
        print_info(f"\n==== FAILED ====")
        print_info(f"Failed to generate working Semgrep rule: {error_message}")

if __name__ == "__main__":
    main()