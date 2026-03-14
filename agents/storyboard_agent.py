"""
分镜 Agent - 负责设计视频分镜和解说词
"""

from .base_agent import BaseAgent
from typing import Any, Dict
from config import DEFAULT_LLM_CONFIG
from langchain_core.prompts import ChatPromptTemplate
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed # 线程池执行器，管理多个线程并发执行任务
# 迭代器，按完成顺序返回未来对象（Futures）


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
                3. description: 画面描述（告诉开发者这一幕画面上有什么、怎么变化，例如"在画面中央逐步绘制直角三角形 ABC，逐一标注三条边的长度，最后高亮斜边并写出勾股定理公式"）
                4. visual_elements: 画面关键元素，用逗号分隔的一个字符串（例如"直角三角形 ABC, 边长标注 6 和 8, 斜边 AB, 勾股定理公式"）
                5. narration: 解说词，口语化、适合朗读，约 30-60 字。不要出现 $、\\、^ 等符号，用自然语言描述公式含义（例如把 $AB=\\sqrt{{AC^2+BC^2}}$ 说成"AB 等于 AC 平方加 BC 平方的平方根"）
                6. transition: 与下一场景的转场方式，固定三选一："淡入淡出" / "直切" / "结束"（最后一个场景用"结束"）

                请严格按照以下 JSON 格式输出，不要添加任何额外内容：
                {{
                    "total_scenes": 场景总数，
                    "scenes": [
                        {{
                            "scene_id": 1,
                            "title": "场景标题",
                            "description": "画面描述",
                            "visual_elements": "元素 1, 元素 2, 元素 3",
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

**【核心原则】几何一致性（必须遵守）：**
1. 场景 1 创建的原始图形（如三角形 ABC）是整个视频的基础，后续场景不能改变其几何结构
2. 对于折叠问题：
   - 场景 1：创建原始三角形 ABC
   - 场景 2+：使用 Transform 或 ReplacementTransform 将部分图形（如三角形 BCD）沿折痕变换到新位置
   - **绝对禁止**创建新的图形（如矩形 ABCD），因为原题中没有矩形
3. 新场景只能创建：
   - 新的标注（如点 C'、虚线、大括号）
   - 新的公式（MathTex）
   - 辅助线（如折痕 BD）
   - **不能**改变原始图形的顶点坐标或形状

请为该场景输出以下字段：
1. scene_id: 场景编号（整数）
2. background: 背景色 Manim 颜色常量（教学视频统一用 "BLACK"）
3. manim_objects: 该场景需要创建的对象列表，每项包含：
   - var_name: Python 变量名（英文下划线命名，如 triangle_abc）
   - manim_class: Manim 类名（如 Polygon、MathTex、Text）
   - params: 构造参数的文字描述（如 "顶点依次为 A(ORIGIN), B(3*RIGHT), C(3*RIGHT+4*UP)"）
   - color: Manim 颜色常量（如 WHITE、YELLOW、BLUE、GREEN、RED）
   - position: 屏幕位置描述（如 "画面居中"、"左半屏"、"右上角"）
   **【重要】如果该对象是场景 1 已有对象的变换（如折叠后的三角形），请在描述中注明"由 XXX 变换而来"**
4. animations: 动画动作序列（按播放先后顺序排列），每项包含：
   - action: Manim 动画类名（如 Create、Write、FadeIn）
   - target: 作用的 var_name
   - run_time: 该动画持续时长（秒，浮点数）
   - params: 额外参数文字描述（无则填空字符串）
5. formula_display: 该场景展示的核心公式（LaTeX 字符串，不加 $ 包裹；无公式则填空字符串）
6. estimated_duration: 所有 animations 的 run_time 之和（秒）

**重要格式要求（必须遵守，否则会导致解析错误）：**
1. 必须输出标准的 JSON 格式
2. 所有标点符号必须用英文半角：逗号 , 冒号 : 引号 " 括号 ()
3. 字符串值内部的标点也必须用英文，例如：
   - ❌ 错误："坐标（-4，0）"  ← 中文括号和逗号
   - ✅ 正确："坐标 (-4, 0)"  ← 英文括号和逗号
4. 数字、字母、运算符统一用英文半角字符，如 3*RIGHT 不能写成 3×RIGHT
5. **场景 1 的三角形坐标必须是实际的坐标值（如 C(0,0), A(0,3.2), B(2.4,0)），不能用 ORIGIN/RIGHT/UP 等宏**

请严格按照以下 JSON 格式输出单个场景，不要添加任何额外内容：
{{
    "scene_id": 1,
    "background": "BLACK",
    "manim_objects": [
        {{
            "var_name": "变量名",
            "manim_class": "Manim 类名",
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
    "formula_display": "LaTeX 公式字符串",
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
        manim_scenes = [None] * len(scenes)
        max_retries = 3

        def process_scene(index: int, scene: Dict[str, Any]):
            """处理单个场景，返回 (index, result)，result 包含解析错误信息或 Manim 指令"""
            scene_id = scene.get("scene_id", "?")
            for attempt in range(1, max_retries + 1):
                print(f"    正在生成场景 {scene_id} 的 Manim 指令（第 {attempt} 次）...")
                response = chain.invoke({
                    "scene_id": scene_id,
                    "scene_data": json.dumps(scene, ensure_ascii=False, indent=2),
                })
                parsed = self._parse_json_response(response.content)
                if "error" not in parsed:
                    return index, parsed
                print(f"    ⚠️  场景 {scene_id} 第 {attempt} 次解析失败：{parsed['error']}，{'重试中...' if attempt < max_retries else '已达最大重试次数'}")
            return index, parsed

        with ThreadPoolExecutor(max_workers=len(scenes)) as executor:
            futures = {
                executor.submit(process_scene, i, scene): i
                for i, scene in enumerate(scenes)
            }
            for future in as_completed(futures):
                index, result = future.result()
                manim_scenes[index] = result # 存储到正确的位置

        return {
            "total_scenes": len(manim_scenes),
            "scenes": manim_scenes,
        }

# =====================解析格式====================================
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON 内容"""
        json_str = self._extract_json_candidate(content)
        if not json_str:
            return {"error": "未找到 JSON 内容", "raw_content": content[:500]}

        normalized = self._normalize_punctuation(json_str)
        candidates = [
            normalized,
            self._repair_common_json_issues(normalized),
        ]

        last_error = None
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                last_error = e

        return {
            "error": f"无法解析 JSON: {last_error}",
            "raw_content": candidates[-1][:500]
        }

    def _extract_json_candidate(self, content: str) -> str:
        """从模型输出中提取最可能的 JSON 主体。"""
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
        source = fence_match.group(1) if fence_match else content

        json_match = re.search(r'\{[\s\S]*\}', source)
        if json_match:
            return json_match.group()
        return ""

    def _normalize_punctuation(self, json_str: str) -> str:
        """统一中英文标点，降低 JSON 解析失败率。"""
        normalized = json_str
        normalized = normalized.replace(',', ',')   # 中文逗号 → 英文逗号
        normalized = normalized.replace(':', ':')   # 中文冒号 → 英文冒号
        normalized = normalized.replace('(', '(')   # 中文左括号 → 英文左括号
        normalized = normalized.replace(')', ')')   # 中文右括号 → 英文右括号
        normalized = normalized.replace(';', ';')   # 中文分号 → 英文分号
        normalized = normalized.replace('!', '!')   # 中文感叹号 → 英文感叹号
        normalized = normalized.replace('?', '?')   # 中文问号 → 英文问号
        normalized = normalized.replace('\u201c', '"').replace('\u201d', '"')  # 中文双引号 → 英文双引号
        normalized = normalized.replace('\u2018', "'").replace('\u2019', "'")  # 中文单引号 → 英文单引号
        return normalized

    def _repair_common_json_issues(self, text: str) -> str:
        """修复 LLM 常见 JSON 失真：非法反斜杠、尾逗号。"""
        repaired = self._escape_invalid_backslashes_in_json_strings(text)
        # 去掉对象/数组结尾前多余逗号
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        return repaired

    def _escape_invalid_backslashes_in_json_strings(self, text: str) -> str:
        """仅在 JSON 字符串内部把非法转义修复为双反斜杠。

        关键逻辑：只有反斜杠后面紧跟单个合法字符才是有效转义。
        例如：\\n、\\t、\\r、\\f、\\b、\\"、\\\\、\\uXXXX 是合法的。
        但：\\frac、\\begin、\\triangle 是非法的，因为它们是 LaTeX 命令，不是 JSON 转义。
        """
        valid_escapes = set('"\\/bfnrtu')
        out = []
        in_string = False
        i = 0

        while i < len(text):
            ch = text[i]

            if ch == '"':
                # 判断是否为被转义的引号
                bs_count = 0
                j = i - 1
                while j >= 0 and text[j] == '\\':
                    bs_count += 1
                    j -= 1
                if bs_count % 2 == 0:
                    in_string = not in_string
                out.append(ch)
                i += 1
                continue

            if in_string and ch == '\\':
                next_char = text[i + 1] if i + 1 < len(text) else ''

                # 检查是否是合法的短转义序列：\n \t \r \f \b \" \\
                is_short_escape = next_char in valid_escapes

                # 检查是否是合法的 Unicode 转义：\uXXXX
                is_valid_unicode = False
                if next_char == 'u' and i + 5 < len(text):
                    hex_part = text[i + 2:i + 6]
                    is_valid_unicode = bool(re.fullmatch(r"[0-9a-fA-F]{4}", hex_part))

                # 关键判断：如果是短转义，检查后面是否还有其他字符
                # 例如 \frac 中的 \f 后面还有 r，说明不是独立的 \f 转义，而是 LaTeX 命令
                if is_short_escape and i + 2 < len(text):
                    char_after_next = text[i + 2]
                    # 如果 \f 后面还有字母（如 \frac 的 r），说明是非法转义
                    if char_after_next.isalpha():
                        is_short_escape = False

                if not is_short_escape and not is_valid_unicode:
                    # 非法转义，需要补一个反斜杠变成 \\
                    out.append('\\\\')
                    i += 1
                    continue

            out.append(ch)
            i += 1

        return ''.join(out)
