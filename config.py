"""
多 Agent 题目解答视频生成系统 - 配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== 模型配置 ====================
# 使用豆包大模型
DEFAULT_LLM_CONFIG = {
    "api_key": os.getenv("ARK_API_KEY"),
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "model": "doubao-seed-2-0-pro-260215",
    "temperature": 0.1,
}

# 视觉模型配置 (用于 OCR)
VISION_MODEL_CONFIG = {
    "api_key": os.getenv("ARK_API_KEY"),
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "model": "doubao-1.5-vision-pro-250328",  
    "temperature": 0.05,
}

# 语音模型配置 (用于 TTS)
VOICE_MODEL_CONFIG = {
    "api_key": os.getenv("ARK_API_KEY"),
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "model": "doubao-seed-2-0-pro-260215",
    "temperature": 0.05,
}

# ==================== Agent 配置 ====================
AGENT_CONFIGS = {
    "vision": {
        "name": "视觉 Agent",
        "description": "负责题目图片的 OCR 识别和理解",
        "temperature": 0.05,
        "max_tokens": 2048,
    },
    "reasoning": {
        "name": "解题 Agent",
        "description": "负责题目分析和逐步解答",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "storyboard": {
        "name": "分镜 Agent",
        "description": "负责设计视频分镜和解说词",
        "temperature": 0.2,
        "max_tokens": 4096,
    },
    "animation": {
        "name": "动画 Agent",
        "description": "负责生成 manim 动画代码",
        "temperature": 0.1,
        "max_tokens": 8192,
    },
    "voice": {
        "name": "语音 Agent",
        "description": "负责将分镜解说词转换为语音文件",
        "temperature": 0.05,
        "max_tokens": 2048,
        "output_dir": "output/audio",
    },
    "validator": {
        "name": "验证 Agent",
        "description": "负责验证解题和代码正确性",
        "temperature": 0.05,
        "max_tokens": 2048,
    },
    "orchestrator": {
        "name": "协调者 Agent",
        "description": "负责整体流程控制和任务分发",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
}

