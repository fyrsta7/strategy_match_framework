import json
import os
import time
import shutil
from git import Repo
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import repo_name_1870_1003

# 全局变量，控制是否跳过已经处理过的代码库
SKIP_EXISTING_KEYWORD_RESULTS = True


optimization_keywords = [
    # 核心性能指标关键词
    "performance",
    "optimize", "optimization",
    "speedup", "speed up", "speed-up",
    "fast", "faster", "fastest",
    "efficient", "efficiency",
    "throughput",
    "latency", "low-latency",
    
    # 优化动作关键词
    "accelerate", "acceleration",
    "improve", "improvement",
    "enhance", "enhancement",
    "boost", "boosted",
    "tune", "tuning",
    "refactor for speed",
    "perf gain",
    "perf win",
    
    # 资源使用优化关键词
    "reduce overhead",
    "reduce memory",
    "memory usage",
    "memory footprint",
    "memory consumption",
    "reduce allocation",
    "cache friendly",
    "cache efficiency",
    "cache hit",
    "cache miss",
    
    # 具体优化技术
    "inlining", "inline function",
    "loop unrolling",
    "vectorization", "vectorize",
    "simd",
    "parallelization", "parallelize",
    "multithreading", "multi-threading",
    "memoization",
    "lazy evaluation",
    "lock-free",
    "zero-copy",
    "hot path",
    "branch prediction",
    "prefetch",
    
    # 性能问题修复
    "bottleneck",
    "hotspot",
    "slow path",
    "critical path",
    "profile guided",
    "profiler",
    "complexity",
    
    # 特定领域优化
    "buffer reuse",
    "avoid copy",
    "avoid allocation",
    "reduce copy",
    "batch processing"
]



def process_keywords_phase(repositories, keywords):
    """
    针对所有代码库，从c_language.json读取commit并执行关键词筛选，
    如果对应的 is_opt_keyword.json 已存在，则跳过。
    """
    print("===== 第一阶段：关键词筛选 =====")
    for repo in tqdm(repositories, desc="Keyword filtering"):
        knowledge_base_path = os.path.join(config.root_path, "knowledge_base", repo)
        c_language_file = os.path.join(knowledge_base_path, "c_language.json")
        output_keyword = os.path.join(knowledge_base_path, "is_opt_keyword.json")
        
        # 检查是否需要跳过当前代码库
        if SKIP_EXISTING_KEYWORD_RESULTS and os.path.exists(output_keyword):
            print(f"[Keyword] 文件 '{output_keyword}' 已存在，跳过代码库 {repo}。")
            continue
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_keyword), exist_ok=True)
        
        filter_commits_by_keywords(c_language_file, output_keyword, keywords, repo)


def filter_commits_by_keywords(input_file, output_file, keywords, repo_name):
    """
    从c_language.json文件读取commit信息，
    使用关键词匹配来筛选出实现性能优化的commit，
    保存结果到is_opt_keyword.json。
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"[Keyword] 错误：输入文件 '{input_file}' 不存在。")
            return
        
        # 读取c_language.json中的commit信息
        with open(input_file, "r", encoding="utf-8") as file:
            all_commits = json.load(file)
        
        # 筛选包含关键词的commit
        optimization_commits = []
        for commit in all_commits:
            message = commit.get("message", "").lower()
            if any(keyword.lower() in message for keyword in keywords):
                # 保留原始commit的所有信息，并添加一个标记
                commit_copy = commit.copy()
                commit_copy["contains_optimization_keyword"] = True
                optimization_commits.append(commit_copy)
        
        # 将筛选结果写入输出文件
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(optimization_commits, file, indent=4)
        
        print(f"[Keyword] 代码库 {repo_name}：从 {len(all_commits)} 个commit中找到 {len(optimization_commits)} 个疑似优化的commit，结果已保存到 {output_file}")
    
    except Exception as e:
        print(f"[Keyword] 处理 {repo_name} 时发生错误: {str(e)}")



if __name__ == "__main__":
    repository_root = os.path.join(config.root_path, "repository")
    result_root = os.path.join(config.root_path, "result")

    # 排除不处理的仓库
    # EXCLUDED_REPOSITORIES = ['v8', 'lede', 'Sandboxie', 'cmder', 'ClickHouse', 'rocksdb', 'obs-studio', 'FFmpeg', 'dragonfly', 'flameshot', 'netdata', 'calculator', 'tdesktop', 'llama.cpp', 'libuv', 'ish', 'react-native', 'qBittorrent', 'Proton', 'grpc', 'Magisk', 'redis', 'ceph', 'ladybird', 'xgboost', 'imgui', 'valkey', 'cosmopolitan', 'sway', 'Flipper', 'sumatrapdf', 'micropython', 'terminal', 'carbon-lang', 'rufus', 'srs', 'winget-cli', 'TDengine', 'tesseract', 'masscan', 'whisper.cpp', 'Tasmota', 'envoy', 'solidity', 'vlc', 'bitcoin', 'radare2', 'spdlog', 'curl', 'postgres', 'yabai', 'json', 'opencv', 'emscripten', 'serenity', 'nginx', 'mediapipe', 'mysql-server', 'swift', 'zstd', 'tensorflow', 'aseprite', 'HandBrake', 'mpv', 'openwrt', 'raylib', 'tmux', 'ecapture', 'googletest', 'bcc', 'php-src', 'faiss', 'duckdb', 'mongo', 'x64dbg', 'nnn', 'flatbuffers', 'C-Plus-Plus', 'BlackHole', 'ImHex', 'taichi', 'qmk_firmware', 'jq', 'electron', 'openssl', 'kcp', 'timescaledb', 'ExplorerPatcher', 'reactos', 'folly', 'protobuf', 'unleashed-firmware', 'git', 'godot', 'scrcpy', 'esp-idf', 'goaccess', 'gpt4all', 'lvgl']
    # EXCLUDED_REPOSITORIES = ["linux-xlnx", "linux"]
    EXCLUDED_REPOSITORIES = []
    # INCLUDED_REPOSITORIES = ["ytsaurus", "agensgraph", "glibc", "graaljs", "wesnoth", "QGIS", "ITK", "VTK", "root", "crawl", "OpenCPN", "scummvm", "proton", "haiku", "kicad-source-mirror", "pacemaker", "freebsd-src", "Floorp", "Slicer", "gcc", "vpp", "qt-creator", "qtbase", "linux", "pipewire", "ParaView", "source", "glib", "dealii"]
    # INCLUDED_REPOSITORIES = repo_name_1870_1003.INCLUDED_REPOSITORIES
    # INCLUDED_REPOSITORIES = ["rocksdb"]
    if not os.path.exists(repository_root):
        print(f"Error: 目录 '{repository_root}' 不存在。")
        sys.exit(1)

    repositories = [
        folder for folder in os.listdir(repository_root)
        if os.path.isdir(os.path.join(repository_root, folder)) and folder not in EXCLUDED_REPOSITORIES
    ]

    # print(repositories)

    # 第一阶段：关键词筛选
    process_keywords_phase(repositories, optimization_keywords)

    print("\n所有仓库处理完成！")
