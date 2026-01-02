"""
配置导入工具模块 - 4-generate_semgrep 目录专用
提供统一的函数来导入 global_config 和 generate_config

使用方法：
    from import_configs import global_config, generate_config
    
    # 然后就可以使用：
    # global_config.GITHUB_TOKEN
    # global_config.root_path
    # generate_config.LLM_MAX_GENERATION_ROUNDS
    # generate_config.COMMIT_GENERATION_COUNT
"""

import sys
from pathlib import Path
import importlib.util

# 获取当前文件所在目录（4-generate_semgrep 目录）
_current_dir = Path(__file__).parent
# 获取父目录（python 目录）
_parent_dir = _current_dir.parent

# 将父目录添加到 sys.path，以便导入父目录的 config.py
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# 导入父目录的 config.py 作为 global_config
import config as global_config

# 导入当前目录的 config.py 作为 generate_config
_current_config_path = _current_dir / "config.py"
_spec = importlib.util.spec_from_file_location("generate_config", _current_config_path)
generate_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_config)

# 导出两个配置模块
__all__ = ['global_config', 'generate_config']

