# SemOpt Arch 代码指南

这份文档旨在帮助 LLM 快速理解 `semopt_arch` 项目的代码架构、数据流向以及开发者的编码风格偏好。在编写新代码时，请严格遵守以下规范。

文档编写要求：写文档时尽量不要使用加粗标记（**text** 或 __text__），保持文档简洁。

## 1. 项目架构概览

本项目专注于体系结构相关的 C/C++ 性能优化，通过从已构建的知识库（semopt_c_paper_backup）中筛选与体系结构相关的优化 commits，进行聚类分析，并生成 Semgrep 规则。

### 核心目录结构

- `python/`: 包含所有核心源代码，按功能分层。
    - `config.py`: 全局配置文件。包含项目根路径 (`root_path`)、GitHub Token、LLM API Key 等。所有脚本都应引用此文件获取配置。
    - `1-knowledge_base/`: 数据获取与预处理。从 GitHub 获取 C/C++ 代码库，筛选性能优化相关的 commit。（继承自 semopt_c 项目）
    - `1-diff_regex/`: 代码差异分析。使用正则表达式分析 commit 是否只修改单个函数。
    - `2-copy_file/`: 文件复制工具。从 semopt_c_paper_backup 批量复制知识库文件到本项目。
    - `2-filter_arch/`: 体系结构相关筛选。在已有知识库基础上，筛选与体系结构相关的性能优化 commits。
    - `3-cluster/`: 聚类分析（通用策略）。使用 Embedding 对优化策略进行聚类分析。
    - `3-cluster_arch/`: 聚类分析（体系结构相关策略）。
    - `3-cluster_huawei_stage2/`: 聚类分析（华为特定需求）。从 `one_func.json` 随机采样，复用已有 summary 以降低成本。
- `knowledge_base/`: 核心数据存储目录。存放各代码库的 commit 数据和文件变更信息。
- `repo_list_*.json`: 代码库列表文件。
    - `repo_list_100.json`: 100个代码库
    - `repo_list_3628.json`: 3628个代码库
    - `repo_list_30342.json`: 30342个代码库（主要使用）
- `all_is_opt_final.json`: 汇总所有代码库中最终确定的性能优化 commits（来自 semopt_c_paper_backup）。
- `all_is_opt_arch_final.json`: 汇总所有代码库中体系结构相关的性能优化 commits。

## 2. 外部数据源与架构

本项目与 `semopt_c` 和 `semopt_c_paper_backup` 项目形成上下游关系。

### 2.1 数据流向

1. `semopt_c_paper_backup`: 已完成的知识库，包含大量 C/C++ 代码库的性能优化 commits
   - 位置: `/data2/zyw/semopt_c_paper_backup/knowledge_base_all/`
   - 内容: 30342个代码库的完整数据

2. `semopt_arch` (本项目): 在已有知识库基础上进行体系结构相关筛选
   - 从 semopt_c_paper_backup 复制基础数据
   - 应用体系结构相关的筛选规则
   - 进行聚类分析

### 2.2 详细文件含义

在 `knowledge_base/{repo_name}/` 目录下：

- `all_commit.json`: 代码库的所有 commit 信息
- `one_file.json`: 只修改一个文件的 commit
- `c_language.json`: 修改 C/C++ 文件的 commit
- `is_opt_keyword.json`: 通过关键词筛选的性能优化 commit
- `has_file.json`: 能获取修改前后文件的 commit
- `has_file_deduplicate.json`: 去重后的 has_file
- `diff.json`: 用于处理的中间文件
- `one_func.json`: 只修改一个函数的 commit
- `line_block.json`: 带行数和块数统计的 commit
- `func_name.json`: 行数块数筛选后的 commit
- `func_name_result.json`: 函数名筛选后的 commit
- `is_opt_llm.json`: LLM 判断的中间结果
- `is_opt_final.json`: 最终的性能优化 commit
- `is_opt_arch_keyword.json`: 通过体系结构关键词筛选的 commit（在 `is_opt_final.json` 基础上）
- `is_opt_arch_llm.json`: 通过 LLM 判断的体系结构相关 commit（包含 `is_opt_arch_llm` 字段）
- `is_opt_arch_final.json`: 最终确定的体系结构相关优化 commit（`is_opt_arch_llm = "true"`）
- `huawei.json`: 从 `one_func.json` 中随机采样的 commit（华为特定需求）
- `summary.json`: 包含优化策略总结的 commit
- `summary_arch.json`: 包含体系结构相关优化策略总结的 commit
- `summary_huawei.json`: 包含华为特定需求的优化策略总结（带复用机制）
- `summary_filter.json`: 筛选后的 summary
- `summary_filter_arch.json`: 筛选后的 summary_arch
- `summary_filter_huawei.json`: 筛选后的 summary_huawei

在 `knowledge_base/{repo_name}/modified_file/{commit_hash}/` 目录下：

- `diff.txt`: Git diff 输出
- `before.*`, `after.*`: 修改前后的完整文件（扩展名可能是 .c, .cpp, .h 等）
- `before_func.*`, `after_func.*`: 修改前后的函数内容

## 3. 代码风格与规范 (Critical)

### 3.1 变量与配置管理

全局变量优先：脚本的配置参数（如输入/输出路径、阈值、模型名称等）尽量定义为脚本顶部的全局大写变量，而不是通过命令行参数 (`argparse`) 传入。
    - Rationale: 方便在 IDE 中直接运行和调试，也方便直接修改脚本配置。
    - 路径配置必须从 `config.py` 中获取，不要写死绝对路径。
    - Example:
        ```python
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import config
        
        # ============ 配置变量 ============
        SOURCE_BASE = os.path.join(config.semopt_c_paper_backup_path, "knowledge_base_all")
        TARGET_BASE = os.path.join(config.root_path, "knowledge_base")
        REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
        ```

引用全局 Config：所有跨脚本的通用配置（路径、API Keys）必须从 `python/config.py` 导入。
    - `config.root_path`: 当前项目（semopt_arch）的根目录，自动获取
    - `config.semopt_c_paper_backup_path`: semopt_c_paper_backup 项目的根目录，自动构造（假设与 semopt_arch 在同一父目录下）
    - Import Pattern:
        ```python
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 添加 python/ 目录到 path
        import config
        ```

### 3.2 路径处理

- 绝对路径：尽量使用绝对路径。基于 `config.root_path` 或 `config.semopt_c_paper_backup_path` 构建路径。
- 禁止写死路径：不要在脚本中直接写绝对路径（如 `/data2/zyw/semopt_arch/...`），必须通过 `config` 模块获取。
- 路径拼接：使用 `os.path.join`。
- 目录结构感知：代码应感知自己在 `python/subdir/` 下，通常需要向上引用父目录的 `config.py`。
- 自动路径：`config.py` 中的 `root_path` 会自动获取项目根目录，无需手动配置。

### 3.3 数据处理

- JSON 为主：数据交换格式统一使用 JSON。
- Deep Copy：在处理 Commit 对象列表时，如果涉及修改或重组，注意使用 `copy.deepcopy` 防止副作用。
- 进度条：耗时循环操作使用 `tqdm` 显示进度。
- 并行处理：对于大量独立任务（如处理多个代码库），使用并行处理加速。
    - 多进程：使用 `multiprocessing.Pool` 处理 CPU 密集型任务
    - 多线程：使用 `concurrent.futures.ThreadPoolExecutor` 处理 I/O 密集型任务
    - 通常使用 `cpu_count()` 获取 CPU 核心数
    - 根据任务类型限制最大进程/线程数（如 16, 32, 128, 256）
    - 线程安全：使用 `threading.Lock` 保护共享资源（如文件写入）
    - Example (多进程):
        ```python
        from multiprocessing import Pool, cpu_count
        
        num_processes = min(cpu_count(), 128)
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_func, data_list),
                total=len(data_list),
                desc="处理进度"
            ))
        ```
    - Example (多线程):
        ```python
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=128) as executor:
            futures = [executor.submit(process_func, item) for item in data_list]
            for future in tqdm(futures, desc="处理进度"):
                result = future.result()
        ```

### 3.4 LLM 调用

- 使用 `config.py` 中定义的 `xmcp_base_url` 和 `xmcp_api_key`。
- 模型名称使用 `config.py` 中预定义的变量（如 `config.xmcp_gpt_model`）。
- 支持多种模型：GPT-4o, DeepSeek-v3, Qwen-max, Claude-3.7 等。

### 3.5 错误处理与日志

- 详细报告：处理大量数据时，生成 JSON 格式的详细报告，包含统计信息和错误记录。
- 日志输出：使用 `print` 输出关键状态，便于跟踪进度。
- 错误收集：将错误信息收集到列表中，最后统一输出或保存到报告文件。

## 4. 主要流程与逻辑

### 第一步：从 semopt_c_paper_backup 复制文件 (`python/2-copy_file/`)

1. `add_name_long.py`: 为代码库添加 `name_long` 字段（格式为 `用户名_仓库名`）
   - 输入: `repo_list_30342.json`（项目根目录）
   - 输出: `repo_list_30342_with_name_long.json`（项目根目录）
   - 作用: 避免不同作者的同名仓库冲突
   - 路径使用 `config.root_path` 获取

2. `check_duplicate_names.py`: 检查代码库名称重复情况
   - 输入: `repo_list_30342_with_name_long.json`（项目根目录）
   - 输出: 控制台统计报告
   - 作用: 验证名称唯一性
   - 路径使用 `config.root_path` 获取

3. `copy_knowledge_base_files.py`: 批量复制知识库文件
   - 输入: 
     - `repo_list_30342.json`（从 `config.root_path` 获取）
     - `all_is_opt_final.json`（从 `config.semopt_c_paper_backup_path` 获取）
   - 源路径: `config.semopt_c_paper_backup_path/knowledge_base_all/`
   - 目标路径: `config.root_path/knowledge_base/`
   - 输出:
     - 复制 262036 个 JSON 文件
     - 复制 178340 个 commit 详细文件
     - `copy_report.json` (详细报告，保存在脚本所在目录)
     - `copy_knowledge_base_files.log` (执行日志)
   - 特点:
     - 使用 128 个并行进程
     - 自动排除旧版/不需要的文件
     - 完整的错误处理和进度显示
     - 所有路径从 `config` 模块获取，避免写死路径

4. `copy_one_func_commits.py`: 复制 `one_func.json` 中的 commits
   - 输入: 源知识库中所有代码库的 `one_func.json`
   - 输出: 
     - `all_one_func.json`（项目根目录，汇总所有 commits）
     - 复制每个 commit 的详细文件到 `knowledge_base/`
   - 特点:
     - 使用多线程读取 JSON（I/O 密集）
     - 使用多进程复制文件（CPU 密集）
     - 自动去重
     - 支持与已有数据对比（避免重复复制）
     - 支持断点续传（`SKIP_EXISTING`）

5. `copy_semgrep_folders.py`: 复制 semgrep 文件夹
   - 输入: `all_one_func.json`（项目根目录）
   - 功能: 检查每个 commit 是否有 semgrep 子文件夹，如果有则复制
   - 源路径: `knowledge_base_all/{repo}/modified_file/{commit}/semgrep/`
   - 目标路径: `knowledge_base/{repo}/modified_file/{commit}/semgrep/`
   - 输出: `copy_semgrep_report.json`（详细报告）
   - 特点:
     - 使用 128 个并行进程
     - 支持断点续传（`SKIP_EXISTING`）
     - 复制整个 semgrep 文件夹（包含 .json 和 .yaml 文件）
     - 自动统计覆盖率（有多少比例的 commits 有 semgrep 文件夹）
     - 路径从 `config` 模块获取

### 第二步：体系结构相关筛选 (`python/2-filter_arch/`)

1. `filter_commit_keyword_arch.py`: 使用体系结构相关关键词筛选 commits
   - 输入: `is_opt_final.json`（每个代码库）
   - 输出: `is_opt_arch_keyword.json`（每个代码库）
   - 关键词包括: CPU架构（x86、ARM、RISC-V等）、SIMD（SSE、AVX、NEON等）、缓存优化、内存架构、原子操作等
   - 筛选与 CPU 体系结构、内存层次结构相关的优化
   - 支持并行处理（`MAX_WORKERS`）
   - 输出详细统计信息

2. `filter_commit_llm_arch.py`: 使用 LLM 判断是否为体系结构相关优化
   - 调用 LLM（DeepSeek-v3）分析 commit message 和代码变更
   - 判断优化是否与体系结构特性相关
   - 四个阶段：
     - 阶段1: 复制 `is_opt_arch_keyword.json` 到 `is_opt_arch_llm.json`
     - 阶段2: LLM 筛选，更新 `is_opt_arch_llm` 字段
     - 阶段3: 提取通过的 commits 到 `is_opt_arch_final.json`
     - 阶段4: 汇总到根目录的 `all_is_opt_arch_final.json`
   - 支持两层并行：仓库级别（`REPO_PARALLEL_WORKERS`）+ commit级别（`MAX_WORKERS`）
   - 支持断点续传（`SKIP_PROCESSED`）

3. `check_llm_progress.py`: 检查 LLM 处理进度
   - 统计待处理和已处理的 commit 数量
   - 显示 LLM 判断结果分布
   - 列出处理进度最慢的仓库

4. `statistic_repo.py`: 统计各阶段的 commit 数量
   - 输入: `repo_list_30342.json`
   - 统计文件: `is_opt_final.json`, `is_opt_arch_keyword.json`, `is_opt_arch_llm.json`
   - 输出: `commit_statistics.csv`
   - 显示汇总数据：总数、平均值、有效库数量
   - 支持并行处理（128个线程）

### 第三步：聚类分析 (`python/3-cluster/`)

1. `other.py`: 辅助函数
   - `copy_opt_file`: 将 `is_opt_final.json` 复制到 `summary.json`

2. `summary_3.py`: 为 commits 生成优化策略总结
   - 使用 Self-consistency voting
   - 并行处理（代码库级别并行 + 代码库内部并行）
   - 输出到 `summary.json`

3. `get_line_num.py`: 计算被修改代码片段的行号
   - 输入: `summary.json`
   - 输出: 添加 `file_start_line`, `file_end_line` 字段

4. `get_line_offset_new.py`: 计算文件和函数的行号偏移
   - 输入: `summary.json`
   - 输出: 添加 `line_offset`, `func_start_line`, `func_end_line`, `before_func_total_lines` 字段

5. `filter_commit.py`: 筛选完整的 commits
   - 输入: `summary.json`
   - 输出: `summary_filter.json`
   - 作用: 只保留有完整信息的 commits

6. `cluster.py`: 使用 Embedding 进行聚类
   - 基于优化策略总结进行相似度计算
   - 输出聚类结果

7. `order.py`: 对聚类内部的 commits 排序
   - 按相似度降序排列
   - 越前面的越能代表该聚类

### 第四步：华为特定需求的聚类分析 (`python/3-cluster_huawei_stage2/`)

从 `one_func.json` 进行随机采样，复用已有 summary 以降低成本。

1. `sample_commits.py`: 随机采样 commits
   - 从各代码库的 `one_func.json` 中采样
   - 采样策略：总量控制（从所有代码库共采样 n 个commits）
   - 输出：`huawei.json`（各代码库）和 `all_huawei.json`（汇总）
   - 配置：`SAMPLE_SIZE`（默认10用于测试）、`RANDOM_SEED`（随机种子）
   - 支持128个线程并行加载

2. `other.py`: 复制文件
   - `copy_huawei_file()`: 将 `huawei.json` 复制到 `summary_huawei.json`

3. `summary.py`: 生成优化策略总结（带复用机制）
   - 优先从 `summary.json` 和 `summary_arch.json` 复用
   - 只对新 commit 调用 LLM（Self-consistency voting）
   - 输出字段：`optimization_summary_huawei`, `optimization_summary_huawei_final`, `reused_from`
   - 输出详细复用统计报告（复用率、节省的LLM调用次数）
   - 支持两层并行：代码库级别（`MAX_REPO_WORKERS`=16）和 commit级别（`MAX_WORKERS`=8）

4-7. 后续流程与通用策略类似：
   - `get_line_num.py`: 计算行号（`file_start_line`, `file_end_line`）
   - `get_line_offset_new.py`: 计算偏移（`line_offset`, `func_start_line`, `func_end_line`）
   - `filter_commit.py`: 筛选完整的 commits（输出 `summary_filter_huawei.json`）
   - `cluster.py`: 聚类分析，输出到 `result_30342/`，文件名格式 `{threshold}_{min_cluster_size}.json`（无filter项）
   - `order.py`: 排序，自动扫描目录中的所有聚类文件，输出 `*_order.json`

### 第五步：规则生成 (待实现)

参考 semopt_c 项目中的 Semgrep 规则生成流程。

## 5. 开发新脚本时的 Checklist

1.  [ ] 头部引入 `config` 模块（调整 `sys.path`）。
2.  [ ] 在顶部定义所有可调参数为全大写全局变量。
3.  [ ] 路径使用 `os.path.join(config.root_path, ...)` 或 `os.path.join(config.semopt_c_paper_backup_path, ...)`。
4.  [ ] 包含 `if __name__ == "__main__": main()` 入口。
5.  [ ] 使用 `tqdm` 展示进度，使用 `print` 输出关键状态。
6.  [ ] 如果涉及文件读写，确保目录存在 (`os.makedirs(..., exist_ok=True)`)。
7.  [ ] 对于大量独立任务，考虑使用并行处理：
    - I/O 密集型：使用 `concurrent.futures.ThreadPoolExecutor`
    - CPU 密集型：使用 `multiprocessing.Pool`
8.  [ ] 如果有并发写入，使用 `threading.Lock` 保证线程安全。
9.  [ ] 生成详细的执行报告（JSON 格式），包含统计信息和错误记录。
10. [ ] 对于 C/C++ 特定的逻辑，需要考虑函数定义语法、文件扩展名等差异。
11. [ ] 输出格式保持一致：使用 emoji 图标、分隔线、清晰的统计信息。

## 6. 项目特色

### 6.1 体系结构相关优化关注点

本项目专注于与 CPU 体系结构和内存层次结构相关的性能优化，包括：

- CPU 架构: x86/x86_64/amd64, ARM/ARM64/AArch64, RISC-V, PowerPC, MIPS, SPARC, LoongArch, S390x
- SIMD 和向量化: SSE/SSE2/SSE3/SSE4, AVX/AVX2/AVX-512, NEON (ARM), AltiVec (PowerPC), MSA (MIPS)
- Cache 优化: cache line/cache alignment, cache-friendly structures, cache coherence, prefetch, cache warmup, false sharing
- Memory 架构: memory alignment, NUMA, memory ordering, memory barriers/fences, memory model, huge pages, TLB
- 原子操作和同步: atomic operations, compare-and-swap (CAS), load-acquire/store-release, relaxed ordering
- 编译器和指令: intrinsics, inline assembly, builtin functions, CPU features detection
- CPU 特性: branch prediction, instruction pipelining, out-of-order execution, speculative execution, hardware prefetch
- 性能监控: performance counters, PMU (Performance Monitoring Unit), hardware counters
- 其他: endianness (little-endian/big-endian), word size, registers, calling convention, ABI

### 6.2 与 semopt_c 的关系

- semopt_c: 广泛的 C/C++ 性能优化知识库
- semopt_arch: 在 semopt_c 基础上，专注于体系结构相关优化的子集
- 数据复用: 从 semopt_c_paper_backup 复制基础数据，避免重复构建
- 额外筛选: 应用体系结构相关的筛选规则，缩小研究范围

### 6.3 并行处理优化

本项目在多个脚本中使用并行处理来加速数据处理：

- `copy_knowledge_base_files.py`: 使用 128 个进程并行复制文件
- `filter_commit_keyword_arch.py`: 使用多线程并行处理代码库（关键词筛选）
- `filter_commit_llm_arch.py`: 两层并行（仓库级别 + commit级别）
- `check_llm_progress.py`: 并行读取仓库的 LLM 处理进度
- `statistic_repo.py`: 使用 128 个线程并行统计
- `summary_3.py`: 代码库级别并行 + 代码库内部并行
- `get_line_offset_new.py`: 多层次并行处理

并行处理的原则：
- 任务独立: 确保各任务之间无依赖关系
- 进程/线程数限制: 根据 CPU 核心数和任务特性限制并行数（通常 16-256）
- 进度显示: 使用 `tqdm` 与并行处理结合显示进度
- 线程安全: 对于文件写入操作，使用文件锁保证数据安全
- 结果合并: 使用专门的函数合并并行任务的结果

## 7. 常见问题

### 7.1 如何处理大量文件？

使用并行处理和批量操作：
```python
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_item(item):
    # 处理单个项目
    return result

items = [...]  # 待处理项目列表
num_processes = min(cpu_count(), 128)

with Pool(processes=num_processes) as pool:
    results = list(tqdm(
        pool.imap(process_item, items),
        total=len(items),
        desc="处理进度"
    ))
```

### 7.2 如何生成详细报告？

创建包含统计信息的字典，最后保存为 JSON：
```python
report = {
    "execution_time": {
        "start": start_time,
        "end": end_time,
        "duration_seconds": duration
    },
    "statistics": {
        "total_items": total,
        "success": success_count,
        "errors": error_count
    },
    "errors": error_list  # 详细错误信息
}

with open("report.json", 'w', encoding='utf-8') as f:
    json.dump(report, indent=2, ensure_ascii=False, fp=f)
```

### 7.3 如何避免重复处理？

- 检查输出文件是否已存在
- 使用标记字段记录处理状态
- 支持断点续传和增量更新

