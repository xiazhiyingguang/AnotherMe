"""
分镜 Agent - 负责设计视频分镜和解说词
"""

from .base_agent import BaseAgent
from typing import Any, Dict
from config import DEFAULT_LLM_CONFIG
from langchain_core.prompts import ChatPromptTemplate
import json
import re


class StoryboardAgent(BaseAgent):
    """分镜 Agent - 负责设计视频分镜和解说词"""

    def __init__(self, config: Dict[str, Any], llm=None):
        """
        初始化 StoryboardAgent
        Args:
            config: StoryboardAgent 的配置字典
            llm: 可选的 LLM 实例，如果不传则内部创建
        """
        if not llm:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=DEFAULT_LLM_CONFIG["api_key"],
                base_url=DEFAULT_LLM_CONFIG["base_url"],
                model=DEFAULT_LLM_CONFIG["model"],
                temperature=config.get("temperature", 0.2),
                max_tokens=config.get("max_tokens", 4096),
            )

        super().__init__(config, llm)

    def process(self, input_data: Any) -> Any:
        return self.design_storyboard(input_data)

    def design_storyboard(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        完整分镜设计流水线：先生成人类可读分镜，再生成 Manim 动画指令。

        Args:
            reasoning_result: ReasoningAgent 输出的解题步骤

        Returns:
            包含 human_storyboard 和 manim_storyboard 的字典
        """
        human_storyboard = self.generate_human_storyboard(reasoning_result)
        manim_storyboard = self.generate_manim_storyboard(human_storyboard)
        return {
            "human_storyboard": human_storyboard,
            "manim_storyboard": manim_storyboard,
        }

    def generate_human_storyboard(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据解题步骤生成人类可读的视频分镜脚本（供开发者审阅）。

        每个解题步骤对应一个视频场景，包含场景标题、画面描述、解说词、转场方式等。
        解说词将在后续由 VoiceAgent 转换为语音。

        Args:
            reasoning_result: ReasoningAgent 输出的解题步骤

        Returns:
            格式如下的字典：
            {
                "total_scenes": int,
                "scenes": [
                    {
                        "scene_id": int,
                        "title": str,          # 场景标题
                        "description": str,    # 画面描述（告诉开发者这一幕画什么）
                        "visual_elements": str,# 关键画面元素，逗号分隔
                        "narration": str,      # 口语化解说词，不含 LaTeX
                        "transition": str      # 转场方式
                    }
                ]
            }
        """
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个数学教育视频的分镜编导，负责将解题步骤转化为视频分镜脚本。
                每个解题步骤对应一个视频场景，请为每个场景设计以下字段：
                1. scene_id: 场景编号（整数，从 1 开始）
                2. title: 场景标题（简短概括该步骤，如"利用勾股定理求斜边"）
                3. description: 画面描述（告诉开发者这一幕画面上有什么、怎么变化，例如"在画面中央逐步绘制直角三角形ABC，逐一标注三条边的长度，最后高亮斜边并写出勾股定理公式"）
                4. visual_elements: 画面关键元素，用逗号分隔的一个字符串（例如"直角三角形ABC, 边长标注6和8, 斜边AB, 勾股定理公式"）
                5. narration: 解说词，口语化、适合朗读，约30-60字。不要出现 $、\\、^ 等符号，用自然语言描述公式含义（例如把 $AB=\\sqrt{{AC^2+BC^2}}$ 说成"AB等于AC平方加BC平方的平方根"）
                6. transition: 与下一场景的转场方式，固定三选一："淡入淡出" / "直切" / "结束"（最后一个场景用"结束"）

                请严格按照以下 JSON 格式输出，不要添加任何额外内容：
                {{
                    "total_scenes": 场景总数,
                    "scenes": [
                        {{
                            "scene_id": 1,
                            "title": "场景标题",
                            "description": "画面描述",
                            "visual_elements": "元素1, 元素2, 元素3",
                            "narration": "口语化解说词",
                            "transition": "淡入淡出"
                        }}
                    ]
                }}"""
                            ),
                            (
                                "human",
                                """以下是解题步骤，请为每个步骤设计一个视频场景：
                                {reasoning_result}

                                请输出完整的分镜脚本 JSON。"""
            )
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "reasoning_result": json.dumps(reasoning_result, ensure_ascii=False, indent=2)
        })
        return self._parse_json_response(response.content)

    def generate_manim_storyboard(self, human_storyboard: Dict[str, Any]) -> Dict[str, Any]:
        """
        将人类可读分镜转化为 Manim 动画指令描述（供 AnimationAgent 生成代码使用）。
        逐场景调用 LLM，避免一次性输出过多内容导致 JSON 被截断。

        AnimationAgent 拿到此结果后，结合 VoiceAgent 返回的实际音频时长，
        用音频时长覆盖 estimated_duration 来控制 run_time，实现音画同步。

        Args:
            human_storyboard: generate_human_storyboard 的输出

        Returns:
            格式如下的字典：
            {
                "total_scenes": int,
                "scenes": [
                    {
                        "scene_id": int,
                        "background": str,
                        "manim_objects": [...],
                        "animations": [...],
                        "formula_display": str,
                        "estimated_duration": float
                    }
                ]
            }
        """
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个 Manim 动画指令设计师，负责将单个视频场景的分镜描述转化为结构化的 Manim 动画指令。
Manim 是一个 Python 数学动画库。

常用对象类：Triangle、Polygon、Line、Dot、MathTex、Text、Arrow、Arc、Circle、Square、Brace、NumberPlane、DecimalNumber
常用动画类：Create、Write、FadeIn、FadeOut、Transform、ReplacementTransform、MoveToTarget、Indicate、GrowFromCenter、DrawBorderThenFill

请为该场景输出以下字段：
1. scene_id: 场景编号（整数）
2. background: 背景色 Manim 颜色常量（教学视频统一用 "BLACK"）
3. manim_objects: 该场景需要创建的对象列表，每项包含：
   - var_name: Python 变量名（英文下划线命名，如 triangle_abc）
   - manim_class: Manim 类名（如 Polygon、MathTex、Text）
   - params: 构造参数的文字描述（如 "顶点依次为 A(ORIGIN), B(3*RIGHT), C(3*RIGHT+4*UP)"）
   - color: Manim 颜色常量（如 WHITE、YELLOW、BLUE、GREEN、RED）
   - position: 屏幕位置描述（如 "画面居中"、"左半屏"、"右上角"）
4. animations: 动画动作序列（按播放先后顺序排列），每项包含：
   - action: Manim 动画类名（如 Create、Write、FadeIn）
   - target: 作用的 var_name
   - run_time: 该动画持续时长（秒，浮点数）
   - params: 额外参数文字描述（无则填空字符串）
5. formula_display: 该场景展示的核心公式（LaTeX 字符串，不加 $ 包裹；无公式则填空字符串）
6. estimated_duration: 所有 animations 的 run_time 之和（秒）

请严格按照以下 JSON 格式输出单个场景，不要添加任何额外内容：
{{
    "scene_id": 1,
    "background": "BLACK",
    "manim_objects": [
        {{
            "var_name": "变量名",
            "manim_class": "Manim类名",
            "params": "参数描述",
            "color": "颜色常量",
            "position": "位置描述"
        }}
    ],
    "animations": [
        {{
            "action": "动画类名",
            "target": "变量名",
            "run_time": 1.5,
            "params": ""
        }}
    ],
    "formula_display": "LaTeX公式字符串",
    "estimated_duration": 5.0
}}"""
            ),
            (
                "human",
                """以下是第 {scene_id} 个场景的人类可读分镜，请将它转化为 Manim 动画指令：
{scene_data}

请输出该场景的 Manim 动画指令 JSON。"""
            )
        ])

        chain = prompt | self.llm
        scenes = human_storyboard.get("scenes", [])
        manim_scenes = []
        max_retries = 3

        for scene in scenes:
            scene_id = scene.get("scene_id", "?")
            parsed = None
            for attempt in range(1, max_retries + 1):
                print(f"    正在生成场景 {scene_id} 的 Manim 指令（第 {attempt} 次）...")
                response = chain.invoke({
                    "scene_id": scene_id,
                    "scene_data": json.dumps(scene, ensure_ascii=False, indent=2),
                })
                parsed = self._parse_json_response(response.content)
                if "error" not in parsed:
                    break
                print(f"    ⚠️  场景 {scene_id} 第 {attempt} 次解析失败：{parsed['error']}，{'重试中...' if attempt < max_retries else '已达最大重试次数'}")
            manim_scenes.append(parsed)

        return {
            "total_scenes": len(manim_scenes),
            "scenes": manim_scenes,
        }

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON 内容"""
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return {"error": "未找到 JSON 内容", "raw_content": content[:500]}

        json_str = json_match.group()
        # 替换中文引号
        json_str = json_str.replace('\u201c', '"').replace('\u201d', '"')

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {
                "error": f"无法解析 JSON: {e}",
                "raw_content": json_str[:500]
            }
