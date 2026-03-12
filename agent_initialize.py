"""
Agent 初始化 - 创建并导出所有 Agent 实例
"""
from config import AGENT_CONFIGS
from agents.vision_agent import VisionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.storyboard_agent import StoryboardAgent
from agents.animation_agent import AnimationAgent
from agents.validator_agent import ValidatorAgent

# 创建各 Agent 实例
# 视觉 Agent 负责 OCR 识别和理解题目图片
vision_agent = VisionAgent(config=AGENT_CONFIGS["vision"])
# 解题 Agent 负责题目分析和逐步解答
reasoning_agent = ReasoningAgent(config=AGENT_CONFIGS["reasoning"])
# 分镜 Agent 负责设计视频分镜和解说词
storyboard_agent = StoryboardAgent(config=AGENT_CONFIGS["storyboard"])
# 动画 Agent 负责生成 manim 动画代码
animation_agent = AnimationAgent(config=AGENT_CONFIGS["animation"])
# 验证 Agent 负责验证解题和代码正确性
validator_agent = ValidatorAgent(config=AGENT_CONFIGS["validator"])

# 方便外部直接导入使用
__all__ = [
    "visionagent""vision_agent""visiona​gent",
    "reasoningagent""reasoning_agent""reasoninga​gent", 
    "storyboardagent""storyboard_agent""storyboarda​gent",
    "animationagent""animation_agent""animationa​gent",
    "validatoragent""validator_agent""validatora​gent",
]
