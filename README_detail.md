# SemOpt Arch


## 目录
- [SemOpt Arch](#semopt-arch)
  - [目录](#目录)
  - [运行前准备](#运行前准备)
  - [1. 从零开始构建知识库](#1-从零开始构建知识库)
  - [2. 从 semopt\_c\_paper\_backup/ 中复制文件](#2-从-semopt_c_paper_backup-中复制文件)
  - [3. 使用体系结构相关的筛选](#3-使用体系结构相关的筛选)
  - [4. （针对通用策略）总结commit用到的策略，做聚类](#4-针对通用策略总结commit用到的策略做聚类)
  - [5. （针对体系结构相关策略）总结commit用到的策略，做聚类](#5-针对体系结构相关策略总结commit用到的策略做聚类)
  - [5.5. （华为特定需求）从 one\_func.json 采样并进行聚类分析](#55-华为特定需求从-one_funcjson-采样并进行聚类分析)
  - [6. 生成semgrep规则](#6-生成semgrep规则)
  - [7. 使用策略库优化代码](#7-使用策略库优化代码)


自动生成目录：
- 在vscode中安装插件 Markdown All in One
- 使用 ctrl / cmd + shift + p，选择 Markdown All in One: Create Table of Contents



## 运行前准备

在 `config.py` 的 `headers` 变量中填入 GitHub 令牌，在 `root_path` 变量中填入本项目的根目录。

在根目录中新建 `repository` 文件夹，然后下载用于提取 benchmark 的开源代码库，并放入 `repository` 文件夹。例如可以下载 [Ceph](https://github.com/ceph/ceph)，并放入文件夹中，相对路径就会是 `repository/ceph`。注意在获取代码时需要保留 git 信息，例如使用 `git clone`。也可以使用 `python/1-knowledge_base/download_repo.py` 来批量下载代码库。


## 1. 从零开始构建知识库

脚本在 `python/1-knowledge_base/`, `python/1-diff_regex/` 中

- 获取代码库
    - 使用 `get_repo_from_github.py` ，从 GitHub 上获取符合条件的一定数量的代码库，结果在 `repo_list.json` 中。
    - 使用 `download_repo.py` 自动下载所有代码库到 `repository/`。
    - （不重要）使用 `summary_repo.py` ，调用 llm 来获取代码库的简要说明，并存储到 `repo_list.json`。
    - （不重要）判断是否为系统软件相关的代码库
        - 使用 `copy_readme.py` 获取每个代码库的 README 文件。
        - 使用 `filter_repo_system_software.py`，调用 llm 来判断代码库是否主要为系统软件方向。
- 获取代码库的所有 commit
    - 使用 `filter_commit_other.py` 中的 `get_all_commits`，将每个代码库中的所有 commit 信息保存到 `knowledge_base/all_commit.json`
- 要求 commit 只修改一个文件
    - 使用 `get_commit_file_count.py` 获取每个 commit 修改的文件数量
    - 使用 `filter_commit_other.py` 中的 `filter_one_file_commits`，从 `knowledge_base/<repository name>/all_commit.json` 中提取 `modified_files_count = 1` 的元素（表明该 commit 中只修改了一个文件）并放到 `one_file.json` 中。
- 修改文件为 C / C++
    - 使用 `filter_commit_other.py` 中的 `filter_c_language_commits`，从 `knowledge_base/<repository name>/one_file.json` 中提取被修改文件为 C/C++ 的元素并放到 `c_language.json` 中。
- 筛选出主要实现性能优化的 commit，使用关键词筛选
    - 使用 `filter_commit_keyword.py` 筛选出实现性能优化的 commit，处理 `c_language.json` 中的所有 commit，结果保存到 `knowledge_base/<repository name>/is_opt_keyword.json`
- 能获取修改前后的文件信息
    - 使用 `get_commit_file_changes.py` 获取 `is_opt_keyword.json` 中每个 commit 中文件修改前后的完整内容，以及 git diff 的内容。这一步如果出错的话，可以使用 `get_commit_file_encoding.py` 来获取指定代码库中的指定 commit 中被修改文件的编码方式，然后调整前面这个脚本。
        - 如果commit的修改就是增加了一个文件，那么针对这个commit，只会生成diff.txt文件，而不会生成before.\*, after.\* 这两个文件。
    - （提取后检查文件是否都存在）使用 `check_before_file.py` 检查 `is_opt_keyword.json` 中的commit是否都有对应的 before.\*, after.\*, diff.txt 这些文件。
    - （根据 `check_before_file.py` 的结果，如果存在commit的before和after文件都存在，但是diff.txt文件不存在或者为空，可以重新计算diff信息并放到文件中）`get_diff_file.py`：为知识库中所有不包含 diff.txt 或者文件为空的commit重新根据 `before.*`, `after.*` 生成 diff.txt
    - 使用 `filter_commit_other.py` 中的 `get_has_file`，从 `knowledge_base/<repository name>/is_opt_keyword.json` 中提取最终符合要求的元素（在 `modified_file/` 中存在的 commit）并放到 `has_file.json` 中。
- 去重
    - 使用 `filter_commit_other.py` 中的 `aggregate_and_deduplicate_has_file`，将 `has_file.json` 汇总到 `all_has_file.json`，去重之后将结果放到 `has_file_deduplicate.json` 中。
- 筛选出只修改一个函数的 commit，并获取相关信息
    - 使用 `filter_commit_other.py` 中的 `copy_has_file` ，复制一份 `has_file_deduplicate.json` 到 `diff.json`
    - （如果需要删除之前获取的 before_func.txt, after_func.txt） 运行 `delete_file.py`
    - （如果错误地删除了 before.\*, after.\* 文件，并且在另一个backup文件夹中存在对应的文件）运行 `copy_before_file.py`，将backup文件夹中存储的commit对应的before和after文件复制到当前的知识库目录中。
        - 根据“获取修改前后的文件信息”这一步，如果commit的修改就是增加了一个文件，那么针对这个commit，只会生成diff.txt文件，而不会生成before.\*, after.\* 这两个文件。所以这个脚本运行时报错 missing_before_file 是正常的，其他可能都不太正常？
    - 使用正则表达式匹配
        - 运行 `1-diff_regex/get_func_our.py`，整体目的和 `get_func_claude37.py` 一样，具体见下。
    - 使用 `filter_commit_other.py` 中的 `filter_one_func_commits`，筛选 `diff.json` 中的 commit，将所有 modified_func_count = 1 并且 modified_other 为 false 并且存在 before_func.\*, after_func.\* 这两个文件的 commit保存到 `one_func.json` 中。
- 使用修改的总行数和代码块数来筛选
    - 使用 `filter_commit_other.py` 中的 `copy_one_func_file` ，复制一份 `one_func.json` 到 `line_block.json`
    - 使用 `get_code_line_num.py`，统计 `line_block.json` 中所有 commit 的修改的总行数，一共输出四个字段 `added_lines`（增加的行数）, `deleted_lines`（减少的行数）, `total_changed_lines`（一共修改的行数）, `net_line_change`（行数的净增减）；另外统计一个修改了多少个代码块，输出到字段 `modified_code_blocks`
    - 使用 `get_code_block_num.py`，统计 `line_block.json` 中所有 commit 的修改的代码块的总数，输出到字段 `modified_code_blocks`
    - 使用 `get_code_stat.py`，统计修改总行数和总块数的分布情况，也就是统计修改小于等于 k 行的 commit 一共有多少个，修改块数同理。
    - 使用 `filter_commit_line_block.py`，用修改的总行数以及修改的代码块数量筛选 commit（暂定为行数<=20，代码块数量<=5），输入为 `line_block.json`，输出为 `func_name.json`
- 使用被修改函数的函数名来筛选，删除修改了单元测试相关函数的commit
    - 使用 `filter_commit_func_name.py`，根据修改函数的函数名来筛选 commit，主要是删除和单元测试相关函数的 commit（但现在不进行真正的删除）。输入文件为 `func_name.json`，输出文件为 `func_name_result.json`。 
- 筛选出主要实现性能优化的 commit，调用 llm 筛选（deepseek-v3）
    - 使用 `filter_commit_other.py` 中的 `copy_func_name_file` ，复制一份 `func_name_result.json` 到 `is_opt_llm.json`
    - 调用llm生成判断结果
        - 使用 `filter_commit_llm.py`，处理 `is_opt_llm.json` 中的所有 commit，结果也保存到 `is_opt_llm.json`。每个commit只进行一次询问，生成 `is_opt_ds_simple` 字段表示使用 DeepSeek-v3 以及只用 commit message 信息来判断 commit 是否为性能优化的结果。
        - （效果不好，不再使用）使用 `filter_commit_llm_2.py`，处理 `is_opt_llm_2.json` 中的所有 commit，结果也保存到 `is_opt_llm_2.json`。每个commit进行两次询问，生成 `is_opt_ds_simple` 字段表示使用 DeepSeek-v3 以及只用 commit message 信息来判断 commit 是否为性能优化的结果，生成 `is_general_ds_simple` 字段表示llm判断commit中的性能优化方式是否通用的结果（如果 `is_opt_ds_simple` 字段为 false，则不进行第二步判断且该字段为 unknown）
    - （不一定要用）使用 `statistic_opt_commit.py` 删除 `is_opt_keyword_deepseek.json` 中的重复 commit，并统计数据，输出到 `statistic_result.csv`。
    - 使用 `filter_commit_other.py` 中的 `filter_optimization_commits`，从 `is_opt_llm.json` 中提取 `is_opt_ds_simple = true` 的元素并放到 `is_opt_final.json` 中。`is_opt_llm_2.json` 的筛选结果放在 `is_opt_final_2.json`，但效果不好所以不用了。
- 汇总知识库
    - 使用 `filter_commit_other.py` 中的 `aggregate_final_commits`，将所有 `is_opt_final.json` 中的内容汇总到 `all_is_opt_final.json` 中，其中删除了 `all_functions` 字段（否则文件太大，而且这个字段意义不大），并增加了 `repository_name` 字段用于记录 commit 对应的代码库。
- 统计数据
    - 使用 `statistic_repo.py` 来统计数据。



## 2. 从 semopt_c_paper_backup/ 中复制文件

脚本在 `python/2-copy_file` 中

从已完成的知识库（`semopt_c_paper_backup/knowledge_base_all/`）复制文件到本项目（`semopt_arch/knowledge_base/`）。

- 使用 `add_name_long.py` 为代码库添加 `name_long` 字段（格式为 `用户名_仓库名`），避免同名仓库冲突
- 使用 `check_duplicate_names.py` 检查名称重复情况
- 使用 `copy_knowledge_base_files.py` 批量复制知识库文件（支持 128 个进程并行）
    - 为 30,342 个代码库复制 JSON 文件（`all_commit.json`, `c_language.json`, `is_opt_final.json` 等）
    - 为 35,668 个 commits 复制详细文件（`diff.txt`, `before.*`, `after.*`, `before_func.*`, `after_func.*`）
    - 自动排除不需要的文件（`*_no_comment.*`, `*_api.json`, `rapgen_*.json` 等）
    - 生成执行报告（`copy_report.json`, `copy_knowledge_base_files.log`）
- 使用 `copy_one_func_commits.py` 复制 `one_func.json` 中的 commits 详细文件
    - 从源知识库读取所有代码库的 `one_func.json`
    - 汇总并去重 commits，保存到 `all_one_func.json`
    - 复制这些 commits 的详细文件（`diff.txt`, `before.*`, `after.*`, `before_func.*`, `after_func.*`）
    - 可选地与 `all_is_opt_final.json` 对比，避免重复复制
    - 支持断点续传（`SKIP_EXISTING = True`）
    - **使用场景**：为华为特定需求（从 `one_func.json` 采样）提供完整数据
- 使用 `copy_semgrep_folders.py` 复制 `all_one_func.json` 中的 commits 对应的 semgrep 文件夹
    - 从源知识库检查所有 `all_one_func.json` 中的 commits
    - 如果源目录中存在 `semgrep` 子文件夹，则复制到目标目录
    - 复制完整的 `semgrep/` 文件夹（包含 `.json` 和 `.yaml` 文件）
    - 支持断点续传（`SKIP_EXISTING = True`）
    - 生成详细报告（`copy_semgrep_report.json`）
    - **使用场景**：为需要 semgrep 规则的 commits 提供对应的规则文件

（以下脚本是用于提取 `all_is_opt_final.json` 中所有commit的信息，并汇总到 `semopt_commit_list/` 文件夹中发给别人，正常运行流程用不到）
- 使用 `extract_commit_files.py` 提取 commit 详细信息文件夹
    - 从 `knowledge_base/` 中提取 `all_is_opt_final.json` 中所有 commit 的详细信息文件夹
    - 复制到 `semopt_commit_list/knowledge_base/` 目录下，保持原有目录结构
    - 支持 128 个进程并行处理，支持跳过已存在的目录
- 使用 `check_commit_files.py` 检查 commit 详细信息文件夹是否已正确复制
    - 验证 `semopt_commit_list/knowledge_base/` 中每个 commit 的详细信息文件夹是否存在且完整
    - 检查必需文件（`diff.txt`, `before.*`, `after.*`, `before_func.*`, `after_func.*`）是否存在且非空
    - 支持 128 个进程并行检查，输出详细的统计信息




## 3. 使用体系结构相关的筛选

脚本在 `python/2-filter_arch` 中

从通用的性能优化 commits 中筛选出与计算机体系结构相关的优化（CPU 架构、SIMD、缓存、内存架构等）。

- 使用 `filter_commit_keyword_arch.py` 进行关键词筛选
    - 输入：`is_opt_final.json`，输出：`is_opt_arch_keyword.json`
    - 关键词：CPU 架构（x86、ARM、RISC-V 等）、SIMD（SSE、AVX、NEON 等）、缓存、内存、原子操作等
    - 支持并行处理，输出详细统计
- 使用 `filter_commit_llm_arch.py` 进行 LLM 精细筛选（DeepSeek-v3）
    - 四个阶段：复制文件 → LLM 筛选（两层并行）→ 提取通过的 commits → 汇总到 `all_is_opt_arch_final.json`
    - LLM 判断标准：性能优化 + 体系结构相关 + 技术通用 + 非可读性改进
    - 支持断点续传（`SKIP_PROCESSED`）
- 使用 `check_llm_progress.py` 检查处理进度
    - 统计代码库和 commit 的处理状态、通过率、待处理数量等
- 使用 `statistic_repo.py` 统计各阶段数据
    - 输出 `commit_statistics.csv`，包含每个代码库在各阶段的 commit 数量和汇总信息


## 4. （针对通用策略）总结commit用到的策略，做聚类

脚本在 `python/3-cluster` 中

- 运行 `other.py` 中的 `copy_opt_file`，将 `is_opt_final.json` 复制到 `summary.json` 中
- 运行 `summary.py`，调用llm给所有 `summary.json` 中的commit写一句话的性能优化策略总结。每个commit上独立生成固定数量的总结，并进行 Self-consistency voting，选出最靠谱的。（参考 `2-general_strategy/summary_commit_vote_1_1.py`）
    - 两个脚本实现的功能基本是类似的，但 `summary_2.py` 的代码会更稳定一点？`summary_3.py` 是在 `summary_2.py` 的基础上，增加了代码库之间的并行（并保留了代码库内部不同commit之间的并行）。
- 运行 `get_line_num.py` ，针对 `summary.json` 中的所有commit，计算被修改的代码片段在 `before.*` 上对应的起始行号和结束行号，记录为字段 `file_start_line`, `file_end_line`
- 运行 `get_line_offset_new.py`，针对 `summary.json` 中的所有commit，计算 before.\*, before_func.\* 这两个文件的行号差值，并记录到 `line_offset` 字段中。然后根据行号差值，从 `file_start_line`, `file_end_line` 计算得到 `file_start_line` 和 `func_start_line` 这两个字段。另外计算 before_func.\* 这个文件的总行数，记录到 `before_func_total_lines` 这个字段中。
    - `get_line_offset.py` 只能在代码库之间并行，代码库内部所有的commit都是串行处理的。但是 `get_line_offset_new.py` 是两个层面都可以并行。
- 运行 `filter_commit.py`，处理 `summary.json` 中的 commit，筛选出所有包含 before.*, after.*, before.*, after.*, func_start_line, func_end_line的commit，放到 `summary_filter.json` 中。主要服务于 `eval1/`，因为后续构建的benchmark显然需要有正确的提取出来的函数结果用于优化，而不能是一个提取函数的结果都错误的commit。但是这里做聚类的时候，可以考虑开启/不开启筛选，看后续效果。
- 运行 `cluster.py` ，对 `summary.json` 中的所有commit做聚类，结果输出到 `python/2-cluster_new/result_100`，文件名类似 `0_7_{True|False}.json`，表示设置的相似度超参数为0.7，后面的 True / False 表示是否先用 filter_commit.py 中的类似方法筛选 commit，只有筛选后的commit才能用于聚类，True表示使用筛选，False表示不适用筛选，所有commit都用于计算聚类。
- 运行 `order.py`，对所有聚类内部的commit根据相似度进行降序排序，越前面的越能代表所在的聚类，结果输出到 `python/2-cluster_new/result_100`，文件名类似 `0_7_True_order.json`，说明是对 `0_7_True.json` 的排序结果。


## 5. （针对体系结构相关策略）总结commit用到的策略，做聚类

脚本在 `python/3-cluster_arch` 中

从体系结构相关的性能优化 commits（`is_opt_arch_final.json`）中总结优化策略并进行聚类分析。流程与通用策略类似，但输入源和输出文件名不同（添加 `_arch` 后缀），且 LLM prompt 强调体系结构特性。

- 运行 `other_arch.py`，将 `is_opt_arch_final.json` 复制到 `summary_arch.json` 中
- 运行 `summary_arch.py`，调用 LLM 给所有 `summary_arch.json` 中的commit写一句话的体系结构相关优化策略总结
    - 每个commit生成多个总结并进行 Self-consistency voting
    - 使用强调体系结构特性的 prompt（SIMD、缓存、内存架构、原子操作等）
    - 生成字段：`optimization_summary_arch`（所有总结）和 `optimization_summary_arch_final`（投票结果）
    - 支持仓库级别并行（`MAX_REPO_WORKERS`）和 commit级别并行（`MAX_WORKERS`）
- 运行 `get_line_num_arch.py`，计算被修改的代码片段在 `before.*` 上对应的起始行号和结束行号
    - 输出字段：`file_start_line`, `file_end_line`
- 运行 `get_line_offset_new_arch.py`，计算 `before.*` 和 `before_func.*` 这两个文件的行号差值
    - 输出字段：`line_offset`, `func_start_line`, `func_end_line`, `before_func_total_lines`
    - 支持两层并行处理
- 运行 `filter_commit_arch.py`，筛选出包含完整信息的 commits
    - 输入：`summary_arch.json`
    - 输出：`summary_filter_arch.json`
    - 筛选条件：存在所有必要文件（`before.*`, `after.*`, `before_func.*`, `after_func.*`, `diff.txt`）和字段（`func_start_line`, `func_end_line`, `optimization_summary_arch_final`）
- 运行 `cluster_arch.py`，对 `summary_filter_arch.json` 中的所有commit做聚类
    - 基于 `optimization_summary_arch_final` 进行相似度聚类
    - 使用DBSCAN算法，支持缓存embeddings
    - 输出到 `result_arch/` 目录，文件名格式：`{threshold}_{min_cluster_size}_{filter}.json`
    - 注意：由于输入已经是 `summary_filter_arch.json`（已完成文件筛选），`cluster_arch.py` 中的 `FILTER_COMMITS_BY_FILES` 参数设置为 True 或 False 结果相同
- 运行 `order_arch.py`，对聚类内部的commit按代表性排序
    - 计算每个commit在聚类中的中心性得分（平均相似度）
    - 按得分降序排列，越前面的越能代表该聚类
    - 输出文件名格式：`{threshold}_{min_cluster_size}_{filter}_order.json`



## 5.5. （华为特定需求）从 one_func.json 采样并进行聚类分析

脚本在 `python/3-cluster_huawei_stage2/` 中

从更早期的数据源（`one_func.json`）进行随机采样，复用已有的 summary 结果以降低成本。输出文件统一使用 `_huawei` 后缀，输入为已筛选的 `summary_filter_huawei.json`（无需额外筛选选项）。

- 运行 `sample_commits.py`，从各代码库的 `one_func.json` 中随机采样 n 个 commit
    - 输入：`one_func.json`（各代码库）
    - 输出：`huawei.json`（各代码库）+ `all_huawei.json`（汇总）
    - 配置：`SAMPLE_SIZE`（采样数量，默认10用于测试）、`RANDOM_SEED`（随机种子，保证可重复性）
    - 采样策略：总量控制（从所有代码库共采样 n 个commits）
    - 支持128个线程并行加载
- 运行 `other.py` 中的 `copy_huawei_file()`，将 `huawei.json` 复制到 `summary_huawei.json`
- 运行 `summary.py`，生成优化策略总结（优先复用已有结果）
    - 优先从 `summary.json` 和 `summary_arch.json` 中查找已有的 summary
    - 只对新 commit 调用 LLM 生成总结（Self-consistency voting）
    - 输出字段：`optimization_summary_huawei`（所有总结）、`optimization_summary_huawei_final`（投票结果）、`reused_from`（复用来源标记：general/arch/generated）
    - 输出详细的复用统计报告（复用率、节省的 LLM 调用次数）
    - 支持代码库级别并行（`MAX_REPO_WORKERS`=16）和 commit级别并行（`MAX_WORKERS`=8）
    - 支持断点续传（`SKIP_PROCESSED`=True），中断后可继续运行
    - 显示双层进度条（代码库级别和commit级别）
- （可选）运行 `summary_statistic.py`，查看 `summary.py` 的处理进度
    - 功能：统计总commit数、已处理数、剩余数，分析剩余commit中可复用和需生成的数量
    - 输出：详细的进度报告（包含已处理来源分析、剩余任务分析、工作量预估）
    - 生成 `progress_report.json` 包含所有有剩余任务的代码库详情
    - 使用场景：
        - 运行 `summary.py` 之前：了解总体工作量
        - 运行过程中：在另一个终端实时查看进度（非侵入式，只读文件）
        - 中断后：确认已完成进度和剩余工作量
    - 命令：`python3 summary_statistic.py`
- 运行 `get_line_num.py`，计算被修改的代码片段在 `before.*` 上对应的起始行号和结束行号
    - 输出字段：`file_start_line`, `file_end_line`
- 运行 `get_line_offset_new.py`，计算 `before.*` 和 `before_func.*` 的行号偏移
    - 输出字段：`line_offset`, `func_start_line`, `func_end_line`, `before_func_total_lines`
    - 支持两层并行处理
- 运行 `filter_commit.py`，筛选出包含完整信息的 commits
    - 输入：`summary_huawei.json`
    - 输出：`summary_filter_huawei.json`
    - 筛选条件：存在所有必要文件（`before.*`, `after.*`, `before_func.*`, `after_func.*`, `diff.txt`）和字段（`func_start_line`, `func_end_line`, `optimization_summary_huawei_final`）
- 运行 `cluster.py`，对 `summary_filter_huawei.json` 中的所有commit做聚类
    - 基于 `optimization_summary_huawei_final` 字段进行相似度聚类
    - 使用DBSCAN算法，支持缓存embeddings
    - 输出到 `result_30342/` 目录（可通过全局变量 `OUTPUT_DIR` 配置）
    - 文件名格式：`{threshold}_{min_cluster_size}.json`（无filter项）
    - 注意：输入已是筛选后的文件，无需额外的筛选选项
- （可选）运行 `merge_clusters.py`，合并多个聚类文件以达到目标聚类数量
    - 使用场景：如果现有聚类数量不足，可以从其他聚类结果中随机抽取并合并
    - 输入：现有聚类文件 + 候选聚类池文件
    - 输出：合并后的聚类文件（`result_final/0_8_2_merged.json`）
    - 功能：从候选池随机抽取指定数量的聚类（size在2-4之间），确保commit不重复，按size降序排列并重新分配cluster_id
    - 自动规范化字段名（`optimization_summary_final` → `optimization_summary_huawei_final`）
    - 输出格式与 `cluster.py` 完全兼容，可直接作为 `order.py` 的输入
    - 输出只包含聚类信息，不包含噪点
- 运行 `order.py`，对聚类内部的commit按代表性排序
    - 自动扫描 `result_30342/` 目录中的所有聚类文件
    - 计算每个commit在聚类中的中心性得分（平均相似度）
    - 按得分降序排列，越前面的越能代表该聚类
    - 输出文件名格式：`{threshold}_{min_cluster_size}_order.json`



## 6. 生成semgrep规则

脚本在 `python/4-generate_semgrep/` 中，使用 LLM 从 commit diff 生成 Semgrep 规则，用于检测类似的性能优化机会。

运行流程：
1. 使用 `process_cluster.py` 中的 `process_cluster_file_simple()`，为多个聚类批量生成规则
2. 使用 `process_cluster_stat.py`，分析聚类的 Semgrep 规则生成状态和统计

对所有脚本的介绍：
- 为单个 commit 生成规则
    - 使用 `process_once.py` 中的 `generate_semgrep_rule_from_file()`，为单个 commit 生成一个 Semgrep 规则
        - 读取 commit 的 diff 信息
        - 调用 LLM 分析优化模式
        - 生成 Semgrep YAML 规则
        - 验证规则语法，迭代修复直到规则正确
        - 结果保存到 `knowledge_base/{repo}/modified_file/{commit}/semgrep/` 目录
    - 使用 `process_commit.py` 中的 `process_commit_semgrep_rules()`，为单个 commit 并行生成多个 Semgrep 规则
        - 调用 `process_once.py` 生成多个规则（默认5个）
        - 支持复用已存在的规则文件（`COMMIT_REUSE_EXISTING = True`）
        - 如果规则有 fatal/error 级别的错误，可选择重新生成（`COMMIT_REGENERATE_ON_JSON_ERROR = True`）
        - 支持多线程并行生成（`COMMIT_MAX_WORKERS = 5`）
- 批量为聚类生成规则
    - 使用 `process_cluster.py` 中的 `process_cluster_file_simple()`，为多个聚类批量生成规则
        - 调用 `process_commit.py` 处理每个 commit
        - 支持聚类级和 commit 级两层并行处理
        - 可配置每个聚类处理的 commit 数量（`SIMPLE_COMMITS_PER_CLUSTER = 10`）
        - 可配置处理的聚类数量限制（`SIMPLE_PROCESS_CLUSTER_LIMIT = 0` 表示处理全部）
- 分析规则生成状态
    - 使用 `process_cluster_stat.py`，分析聚类的 Semgrep 规则生成状态和统计
        - 不生成新规则，只检查和报告现有规则质量
        - 统计每个聚类中已生成规则的 commit 数量
        - 分析规则的有效性（是否有 fatal/error 级别的错误）
- 配置文件
    - `config.py`：包含所有生成相关的配置项
        - LLM 配置：`LLM_MAX_GENERATION_ROUNDS`（最大迭代轮数，默认7），`LLM_TEMPERATURE`（温度参数，默认0.3）
        - Commit 级配置：`COMMIT_GENERATION_COUNT`（每个 commit 生成的规则数量，默认5），`COMMIT_MAX_WORKERS`（并行线程数，默认5）
        - Cluster 级配置：`SIMPLE_CLUSTER_MAX_WORKERS`（聚类级并行线程数，默认8），`SIMPLE_COMMIT_MAX_WORKERS`（commit 级并行线程数，默认8）
    - `import_configs.py`：统一的配置导入接口
        - 导入全局配置（API keys、路径等）：`global_config`
        - 导入生成模块配置：`generate_config`



## 7. 使用策略库优化代码

脚本在 `python/5-opt/` 中，运行 Semgrep 规则、分析结果、测试规则质量、基于结果进行优化。

运行流程：
1. 使用 `evaluate_cluster_rule.py`，在测试集上运行所有聚类的 Semgrep 规则
2. 使用 `summary_semgrep_result_cluster.py`，处理所有 Semgrep 的扫描结果，包括在聚类内合并和全局排序。（这里是考虑随机给出一个函数进行优化，所以不存在需要排除同一commit / 同代码库中的commit对应的优化策略条目的问题）
3. 使用 `opt_eval1.py`，实际进行代码优化


所有脚本的介绍：
- 在代码上运行 Semgrep 规则
    - 使用 `get_semgrep_result_once.py` 中的 `run_semgrep_on_file()`，在单个文件上运行单个 Semgrep 规则
        - 执行 `semgrep` 命令行工具
        - 解析 JSON 格式的运行结果
        - 处理执行超时和错误
        - 保存结果到 JSON 文件
    - 使用 `get_semgrep_result_commit.py` 中的 `process_all_semgrep_rules()`，在单个文件上批量运行多个 Semgrep 规则
        - 扫描目录中的所有 YAML 规则文件
        - 调用 `run_semgrep_on_file()` 逐个运行
        - 支持跳过已存在的结果文件（`SEMGREP_COMMIT_SKIP_EXISTING = False`）
        - 显示处理进度和统计信息
- 分析规则覆盖率和质量
    - 使用 `analyze_semgrep_result.py` 中的 `analyze_cross_commit_semgrep_coverage()`，分析规则在目标代码上的覆盖情况
        - 计算规则匹配结果与目标函数的重叠情况
        - 评估覆盖率（最小重叠行数、覆盖比例等）
        - 支持精确边界匹配或部分覆盖
        - 生成详细的分析报告（JSON 格式）
    - 使用 `test_cluster.py`，在聚类级别测试规则质量
        - 交叉验证规则在其他 commit 上的表现
        - 统计规则的命中率和准确性
        - 可能涉及基于测试结果的优化
- 维护和清理
    - 使用 `delete_error_semgrep.py`，清理错误的 Semgrep 规则文件
        - 检测有 fatal/error 级别错误的规则
        - 批量删除这些错误规则
        - 用于维护规则质量
    - 使用 `delete_semgrep_result.py`，清理 Semgrep 运行结果
        - 批量删除运行结果 JSON 文件
        - 释放磁盘空间
- 配置文件
    - `config.py`：包含所有运行和分析相关的配置项
        - 执行配置：`TIMEOUT`（执行超时时间，默认60秒），`SEMGREP_COMMIT_SKIP_EXISTING`（是否跳过已存在的结果，默认False）
        - 分析配置：`ANALYZE_MIN_OVERLAP_LINES`（最小重叠行数，默认1），`ANALYZE_MIN_COVERAGE_RATIO`（最小覆盖比例，默认0.5）
        - 评估配置：`EVAL_MAX_RULES_PER_COMMIT`（每个 commit 最多使用的规则数，默认5），`EVAL_MAX_WORKERS_COMMITS`（commit 级别的并行数，默认12）
    - `import_configs.py`：统一的配置导入接口
        - 导入全局配置：`global_config`
        - 导入运行/优化模块配置：`opt_config`



和 eval1 相关的脚本介绍：
- 在代码上运行策略库中的所有 Semgrep 规则
    - 使用 `evaluate_cluster_rule.py`，在测试集上运行所有聚类的 Semgrep 规则
        - 输入：聚类文件（包含 commit 聚类结果和对应的 Semgrep 规则）、测试集文件（需要被评估的目标 commit 列表）
        - 过程：跨聚类规则评估，使用聚类中学习到的 Semgrep 规则去检测测试集中的 commit，验证规则的泛化能力
        - 在 `before` 文件上运行 semgrep 规则，汇总扫描结果后判断代码片段是否属于 `before_func`，并计算对应的行号
        - 输出：结果文件 `{OUTPUT_BASE_DIR}/<repo>/<hash>.json`，每个测试 commit 对应一个 JSON 文件，包含所有聚类规则的检测结果
    - 使用 `old/evaluate_cluster_rule_old.py`，旧版本的评估脚本（在 `before_func` 文件上运行，现已不用）
- 处理所有 Semgrep 的扫描结果，包括合并和排序
    - 使用 `summary_semgrep_result_all_False.py`：整个文件中都可以合并，然后全局根据扫描到的次数降序排序。只忽略当前 commit 对应的结果
    - 使用 `summary_semgrep_result_cluster_False.py`：只能在聚类内合并，然后全局根据扫描到的次数降序排序。只忽略当前 commit 对应的结果
    - 使用 `summary_semgrep_result_all_True.py`：整个文件中都可以合并，然后全局根据扫描到的次数降序排序。忽略同代码库的所有 commit 对应的结果
    - 使用 `summary_semgrep_result_cluster_True.py`：只能在聚类内合并，然后全局根据扫描到的次数降序排序。忽略同代码库的所有 commit 对应的结果
    - 输入：聚类文件、测试集文件、结果文件（`evaluate_cluster_rule.py` 的输出）
    - 输出：
        - 聚类内合并：保存到 `<repo>/<hash>_cluster.json`
        - 全局合并：保存到 `<repo>/<hash>_all.json`
        - 结果按代码片段对应的 semgrep 规则数量降序排列，并分配唯一的 id 字段
- 根据处理结果，选择前 n 个优化方案来调用 LLM 并给出优化结果
    - 使用 `opt_eval1.py`，实际进行代码优化
        - 输入：
            - `eval1/benchmark/` 中的 json 文件，记录所有需要被优化的 commit
            - 汇总文件（`summary_*` 脚本的输出），记录每个 commit 可优化的代码片段及对应的 semgrep 规则
        - 过程：
            - 可选择 `all` 或 `cluster` 模式（跨聚类合并或聚类内合并）
            - 选择前 n 个代码片段，调用 LLM 进行优化
        - 输出：存储到 `eval1/knowledge_base/<repository_name>/modified_file/<commit_hash>/our/<folder_name>/`
            - 使用第 m 个代码片段优化的第 n 次结果放在 `m_n.txt` 中
    - 使用 `old/opt_eval1_old.py`，旧版本的优化脚本（现已不用）
- 消融实验相关脚本
    - 每个策略中只使用一个 cluster：`evaluate_cluster_rule_semgrep1.py`，配合 `summary_semgrep_result_cluster_False.py` 和 `opt_eval1.py`
    - 处理 semgrep 运行结果时先聚类内合并然后随机选择：`summary_semgrep_result_cluster_False_random.py`，配合 `opt_eval1_random.py`
