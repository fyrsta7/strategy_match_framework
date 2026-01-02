#!/usr/bin/env python3
"""
统计代码库中commit的修改行数和代码块分布情况
"""
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局配置变量
INPUT_FILENAME = "line_block.json"
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base_all")

def collect_statistics():
    """
    收集所有代码库的统计数据
    Returns:
        tuple: (total_changed_lines列表, modified_code_blocks列表, 总commit数, 有效commit数)
    """
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"错误: 知识库路径不存在: {KNOWLEDGE_BASE_PATH}")
        return [], [], 0, 0

    repositories = [
        repo_name for repo_name in os.listdir(KNOWLEDGE_BASE_PATH)
        if os.path.isdir(os.path.join(KNOWLEDGE_BASE_PATH, repo_name))
    ]

    if not repositories:
        print("未找到任何代码仓库")
        return [], [], 0, 0

    print(f"找到 {len(repositories)} 个代码库，正在收集统计数据...")

    total_changed_lines_list = []
    modified_code_blocks_list = []
    total_commits = 0
    valid_commits = 0

    for repo_name in tqdm(repositories, desc="处理代码库"):
        json_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, INPUT_FILENAME)
        
        try:
            if not os.path.exists(json_path):
                print(f"警告: {repo_name} 中不存在文件 {INPUT_FILENAME}")
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                commits_data = json.load(f)

            if not isinstance(commits_data, list):
                print(f"警告: {repo_name} 的JSON文件格式错误")
                continue

            total_commits += len(commits_data)

            for commit in commits_data:
                # 检查是否有所需字段
                has_lines = 'total_changed_lines' in commit
                has_blocks = 'modified_code_blocks' in commit

                if has_lines or has_blocks:
                    valid_commits += 1

                if has_lines:
                    lines = commit['total_changed_lines']
                    if isinstance(lines, (int, float)) and lines >= 0:
                        total_changed_lines_list.append(int(lines))

                if has_blocks:
                    blocks = commit['modified_code_blocks']
                    if isinstance(blocks, (int, float)) and blocks >= 0:
                        modified_code_blocks_list.append(int(blocks))

        except Exception as e:
            print(f"处理 {repo_name} 时出错: {e}")
            continue

    return total_changed_lines_list, modified_code_blocks_list, total_commits, valid_commits

def calculate_cumulative_distribution(data_list, max_value=None):
    """
    计算累积分布
    Args:
        data_list: 数据列表
        max_value: 最大值，如果不指定则使用数据中的最大值
    Returns:
        dict: {值: 小于等于该值的数量}
    """
    if not data_list:
        return {}
    
    if max_value is None:
        max_value = max(data_list)
    
    counter = Counter(data_list)
    cumulative = {}
    cumulative_count = 0
    
    for i in range(max_value + 1):
        cumulative_count += counter.get(i, 0)
        cumulative[i] = cumulative_count
    
    return cumulative

def print_distribution_table(cumulative_dist, field_name, total_count):
    """
    打印分布表格
    """
    print(f"\n=== {field_name} Cumulative Distribution ===")
    print(f"{'Value':<6} {'Count':<8} {'Percent':<8} {'Cumulative %':<12}")
    print("-" * 40)
    
    # 显示一些关键点
    key_points = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500]
    if cumulative_dist:
        max_val = max(cumulative_dist.keys())
        key_points.append(max_val)
    
    key_points = sorted(set(key_points))
    
    for val in key_points:
        if val in cumulative_dist:
            count = cumulative_dist[val]
            percentage = (count / total_count * 100) if total_count > 0 else 0
            print(f"≤{val:<5} {count:<8} {percentage:<7.1f}% {percentage:<11.1f}%")

def plot_distributions(lines_data, blocks_data):
    """
    绘制分布图表
    """
    # 使用默认字体，避免中文字体问题
    # plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 修改行数直方图
    if lines_data:
        # 限制显示范围以便更好地观察分布
        lines_filtered = [x for x in lines_data if x <= 100]
        ax1.hist(lines_filtered, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Lines Changed')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Lines Changed Distribution (≤100 lines)\nTotal: {len(lines_data)}, Shown: {len(lines_filtered)}')
        ax1.grid(True, alpha=0.3)
    
    # 2. 代码块数直方图
    if blocks_data:
        blocks_filtered = [x for x in blocks_data if x <= 20]
        ax2.hist(blocks_filtered, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Code Blocks Modified')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Code Blocks Distribution (≤20 blocks)\nTotal: {len(blocks_data)}, Shown: {len(blocks_filtered)}')
        ax2.grid(True, alpha=0.3)
    
    # 3. 修改行数累积分布
    if lines_data:
        lines_cumulative = calculate_cumulative_distribution(lines_data, min(100, max(lines_data)))
        x_vals = sorted(lines_cumulative.keys())
        y_vals = [lines_cumulative[x] for x in x_vals]
        ax3.plot(x_vals, y_vals, 'b-', linewidth=2)
        ax3.set_xlabel('Lines Changed')
        ax3.set_ylabel('Cumulative Count')
        ax3.set_title('Lines Changed Cumulative Distribution (≤100 lines)')
        ax3.grid(True, alpha=0.3)
    
    # 4. 代码块数累积分布
    if blocks_data:
        blocks_cumulative = calculate_cumulative_distribution(blocks_data, min(20, max(blocks_data)))
        x_vals = sorted(blocks_cumulative.keys())
        y_vals = [blocks_cumulative[x] for x in x_vals]
        ax4.plot(x_vals, y_vals, 'g-', linewidth=2)
        ax4.set_xlabel('Code Blocks Modified')
        ax4.set_ylabel('Cumulative Count')
        ax4.set_title('Code Blocks Cumulative Distribution (≤20 blocks)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(os.path.dirname(__file__), 'commit_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {output_path}")
    
    # 显示图表
    try:
        plt.show()
    except Exception as e:
        print(f"Cannot display chart (headless environment?): {e}")

def print_summary_statistics(lines_data, blocks_data):
    """
    打印汇总统计信息
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if lines_data:
        print(f"\nLines Changed Statistics:")
        print(f"  Sample Size: {len(lines_data)}")
        print(f"  Mean: {np.mean(lines_data):.2f}")
        print(f"  Median: {np.median(lines_data):.2f}")
        print(f"  Std Dev: {np.std(lines_data):.2f}")
        print(f"  Min: {min(lines_data)}")
        print(f"  Max: {max(lines_data)}")
        print(f"  25th Percentile: {np.percentile(lines_data, 25):.2f}")
        print(f"  75th Percentile: {np.percentile(lines_data, 75):.2f}")
    
    if blocks_data:
        print(f"\nCode Blocks Modified Statistics:")
        print(f"  Sample Size: {len(blocks_data)}")
        print(f"  Mean: {np.mean(blocks_data):.2f}")
        print(f"  Median: {np.median(blocks_data):.2f}")
        print(f"  Std Dev: {np.std(blocks_data):.2f}")
        print(f"  Min: {min(blocks_data)}")
        print(f"  Max: {max(blocks_data)}")
        print(f"  25th Percentile: {np.percentile(blocks_data, 25):.2f}")
        print(f"  75th Percentile: {np.percentile(blocks_data, 75):.2f}")

def save_statistics_to_file(lines_data, blocks_data, total_commits, valid_commits):
    """
    将统计结果保存到文件
    """
    output_file = os.path.join(os.path.dirname(__file__), 'commit_statistics.txt')
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("COMMIT STATISTICS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Data Collection Summary:\n")
            f.write(f"  Total commits: {total_commits}\n")
            f.write(f"  Commits with statistics: {valid_commits}\n")
            f.write(f"  Commits with line data: {len(lines_data)}\n")
            f.write(f"  Commits with block data: {len(blocks_data)}\n\n")
            
            if lines_data:
                f.write("Lines Changed Statistics:\n")
                f.write(f"  Sample Size: {len(lines_data)}\n")
                f.write(f"  Mean: {np.mean(lines_data):.2f}\n")
                f.write(f"  Median: {np.median(lines_data):.2f}\n")
                f.write(f"  Std Dev: {np.std(lines_data):.2f}\n")
                f.write(f"  Min: {min(lines_data)}\n")
                f.write(f"  Max: {max(lines_data)}\n")
                f.write(f"  25th Percentile: {np.percentile(lines_data, 25):.2f}\n")
                f.write(f"  75th Percentile: {np.percentile(lines_data, 75):.2f}\n\n")
                
                # 累积分布
                lines_cumulative = calculate_cumulative_distribution(lines_data)
                f.write("Lines Changed Cumulative Distribution:\n")
                f.write(f"{'Value':<6} {'Count':<8} {'Percent':<8}\n")
                f.write("-" * 25 + "\n")
                
                key_points = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500]
                if lines_cumulative:
                    max_val = max(lines_cumulative.keys())
                    key_points.append(max_val)
                
                for val in sorted(set(key_points)):
                    if val in lines_cumulative:
                        count = lines_cumulative[val]
                        percentage = (count / len(lines_data) * 100)
                        f.write(f"≤{val:<5} {count:<8} {percentage:<7.1f}%\n")
                f.write("\n")
            
            if blocks_data:
                f.write("Code Blocks Modified Statistics:\n")
                f.write(f"  Sample Size: {len(blocks_data)}\n")
                f.write(f"  Mean: {np.mean(blocks_data):.2f}\n")
                f.write(f"  Median: {np.median(blocks_data):.2f}\n")
                f.write(f"  Std Dev: {np.std(blocks_data):.2f}\n")
                f.write(f"  Min: {min(blocks_data)}\n")
                f.write(f"  Max: {max(blocks_data)}\n")
                f.write(f"  25th Percentile: {np.percentile(blocks_data, 25):.2f}\n")
                f.write(f"  75th Percentile: {np.percentile(blocks_data, 75):.2f}\n\n")
                
                # 累积分布
                blocks_cumulative = calculate_cumulative_distribution(blocks_data)
                f.write("Code Blocks Modified Cumulative Distribution:\n")
                f.write(f"{'Value':<6} {'Count':<8} {'Percent':<8}\n")
                f.write("-" * 25 + "\n")
                
                key_points = [1, 2, 3, 4, 5, 10, 20, 50]
                if blocks_cumulative:
                    max_val = max(blocks_cumulative.keys())
                    key_points.append(max_val)
                
                for val in sorted(set(key_points)):
                    if val in blocks_cumulative:
                        count = blocks_cumulative[val]
                        percentage = (count / len(blocks_data) * 100)
                        f.write(f"≤{val:<5} {count:<8} {percentage:<7.1f}%\n")
        
        print(f"\nDetailed statistics saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving statistics file: {e}")

def main():
    """
    主函数
    """
    print("Starting commit statistics analysis...")
    
    # 收集数据
    lines_data, blocks_data, total_commits, valid_commits = collect_statistics()
    
    if not lines_data and not blocks_data:
        print("No valid statistical data found")
        return
    
    print(f"\nData collection completed:")
    print(f"  Total commits: {total_commits}")
    print(f"  Commits with statistics: {valid_commits}")
    print(f"  Commits with line data: {len(lines_data)}")
    print(f"  Commits with block data: {len(blocks_data)}")
    
    # 打印汇总统计
    print_summary_statistics(lines_data, blocks_data)
    
    # 计算并打印累积分布
    if lines_data:
        lines_cumulative = calculate_cumulative_distribution(lines_data)
        print_distribution_table(lines_cumulative, "Lines Changed", len(lines_data))
    
    if blocks_data:
        blocks_cumulative = calculate_cumulative_distribution(blocks_data)
        print_distribution_table(blocks_cumulative, "Code Blocks Modified", len(blocks_data))
    
    # 保存详细统计到文件
    save_statistics_to_file(lines_data, blocks_data, total_commits, valid_commits)
    
    # 绘制图表
    try:
        plot_distributions(lines_data, blocks_data)
    except ImportError:
        print("\nNote: matplotlib not installed, cannot generate charts")
        print("Run: pip install matplotlib to install")
    except Exception as e:
        print(f"\nError generating charts: {e}")

if __name__ == "__main__":
    main()