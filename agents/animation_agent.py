"""
动画 Agent 负责根据分镜设计生成讲解动画代码块。

该 Agent 不负责最终视频渲染与音视频合成；这些职责由 FigureComposerAgent 接管。
"""

from .base_agent import BaseAgent
from typing import Any, Dict, Optional
from config import DEFAULT_LLM_CONFIG
from langchain_core.prompts import ChatPromptTemplate
import json
import re


class AnimationAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any], llm: Optional[Any] = None):
        """初始化 AnimationAgent。"""
        self.config = config
        if not llm:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=DEFAULT_LLM_CONFIG["api_key"],
                base_url=DEFAULT_LLM_CONFIG["base_url"],
                model=DEFAULT_LLM_CONFIG["model"],
                temperature=config.get("temperature", 0.1),
                max_tokens=config.get("max_tokens", 8192),
            )

        super().__init__(config, llm)

    def process(
        self,
        manim_storyboard: Dict[str, Any],
        voice_result: Dict[str, Any],
        geometry_ir: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        生成场景动画代码块与缩放后的分镜数据。

        Args:
            manim_storyboard: StoryboardAgent 输出的 manim_storyboard
            voice_result: VoiceAgent 输出的音频信息

        Returns:
            包含 scene_animation_code 和 scaled_storyboard 的字典
        """
        for scene in manim_storyboard.get("scenes", []):
            if isinstance(scene, dict) and "error" in scene:
                scene_id = scene.get("scene_id", "?")
                raise RuntimeError(
                    f"场景 {scene_id} 解析失败：{scene['error']}. "
                    "无法继续生成动画，请先修复 StoryboardAgent 的 JSON 解析问题。"
                )

        scaled_storyboard = self.scale_storyboard_to_audio(manim_storyboard, voice_result)
        scene_animation_code = self.generate_scene_animation_code(scaled_storyboard, geometry_ir)
        scene_animation_code = self.sanitize_scene_animation_code(scene_animation_code)
        return {
            "scaled_storyboard": scaled_storyboard,
            "scene_animation_code": scene_animation_code,
        }

    def scale_storyboard_to_audio(self, manim_storyboard: Dict[str, Any], voice_result: Dict[str, Any]) -> Dict[str, Any]:
        """按音频时长比例缩放每个场景中的 run_time。"""
        voice_map = {
            scene["scene_id"]: scene["duration"]
            for scene in voice_result.get("scenes", [])
        }

        scaled_scenes = []
        for scene in manim_storyboard.get("scenes", []):
            scene_copy = dict(scene)
            estimated = scene_copy.get("estimated_duration", 1.0)
            real_duration = voice_map.get(scene_copy.get("scene_id"), estimated)

            if estimated > 0:
                scale = real_duration / estimated
                scene_copy["animations"] = [
                    {**anim, "run_time": round(anim.get("run_time", 1.0) * scale, 2)}
                    for anim in scene_copy.get("animations", [])
                ]
            scene_copy["estimated_duration"] = real_duration
            scaled_scenes.append(scene_copy)

        return {
            "total_scenes": len(scaled_scenes),
            "scenes": scaled_scenes,
        }

    def generate_scene_animation_code(
        self,
        scaled_storyboard: Dict[str, Any],
        geometry_ir: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        将缩放后的分镜数据转为可嵌入 FullVideo.construct 的 Python 代码块。

        输出必须是"代码片段"，不能包含 imports、Scene 类定义或渲染逻辑。
        """
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是 Manim 动画代码生成器。请根据分镜输出可直接插入 construct(self) 的代码片段（仅代码，不要 import/class/construct）。

                硬规则：
                1) 仅输出场景代码，场景间用注释 # === 场景 X === 分隔。
                2) 不得重建或删除 problem_img/problem_group。
                3) 每个场景用 scene_objects 记录“本场景新增对象”，场景末尾只清理 scene_objects；禁止 FadeOut(*self.mobjects)。
                4) 场景1的 triangle_abc 必须用 geometry_ir points 坐标创建，并作为常驻基底：
                   - 不得加入 scene_objects；
                   - 场景2及以后必须复用，禁止重定义 triangle_abc。
                5) 布局必须图左文右：
                         - 点/线/角/三角形等几何对象放左侧（约 LEFT*1.5 区域）；
                         - 公式（formula_/eq_ 开头的 MathTex/Text）放右侧（建议 to_edge(RIGHT, buff=1.2) 或右侧纵向排布）。
                6) MathTex 禁止中文；标注优先 next_to，避免固定坐标重叠。
                7) 禁止 NumberPlane/Axes/CoordinateSystem 等多余对象。
                8) 禁止 get_part(s)_by_tex(..., index=...)。
                9) 虚线必须用 DashedLine(..., dash_length=...)，禁止 Line(..., dashed_pattern=...)。
                """
            ),
            (
                "human",
                """
                以下是已按音频时长缩放后的结构化分镜，请输出场景动画代码块：

                {scaled_storyboard}

                可选的几何 IR（若为空可忽略）：
                {geometry_ir}
                """
            )
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "scaled_storyboard": json.dumps(scaled_storyboard, ensure_ascii=False, indent=2),
            "geometry_ir": json.dumps(geometry_ir or {}, ensure_ascii=False, indent=2),
        })
        return response.content

    def sanitize_scene_animation_code(self, code: str) -> str:
        """对生成的代码块做基础安全清洗，避免 LaTeX 和格式问题。"""
        normalized = self._strip_code_fences(code)
        normalized = (
            normalized.replace("，", ",")
            .replace("：", ":")
            .replace("；", ";")
            .replace("（", "(")
            .replace("）", ")")
        )

        pattern = re.compile(r"MathTex\(\s*(r?)([\"'])(.*?)\2", re.DOTALL)

        def replace_if_chinese(match: re.Match) -> str:
            if re.search(r"[\u4e00-\u9fff]", match.group(3)):
                return match.group(0).replace("MathTex(", "Text(", 1)
            return match.group(0)

        normalized = pattern.sub(replace_if_chinese, normalized)

        # 清掉模型偶尔误加的类定义，确保输出是代码块而不是完整文件
        normalized = re.sub(r"from\s+manim\s+import\s+\*\s*", "", normalized)
        normalized = re.sub(r"class\s+\w+\(Scene\):[\s\S]*?def\s+construct\(self\):", "", normalized)

        # 修复 dashed_pattern 参数问题（新版 Manim 已废弃该参数）
        # 将 Line(..., dashed_pattern=...) 转换为 DashedLine(...)
        def fix_dashed_line(code: str) -> str:
            def replacer(m: re.Match) -> str:
                full_match = m.group(0)
                # 提取 Line(...) 括号内的完整内容
                inner_match = re.search(r'Line\s*\(([^)]+)\)', full_match)
                if not inner_match:
                    return full_match
                inner = inner_match.group(1)

                # 解析参数：处理起点可能是 [x,y,z] 列表的情况
                # 找到第一个逗号（在列表括号外的逗号）
                depth = 0
                first_comma = -1
                for i, ch in enumerate(inner):
                    if ch == '[':
                        depth += 1
                    elif ch == ']':
                        depth -= 1
                    elif ch == ',' and depth == 0:
                        first_comma = i
                        break

                if first_comma == -1:
                    return full_match

                start = inner[:first_comma].strip()
                rest = inner[first_comma+1:].strip()

                # 从 rest 中找到终点（下一个逗号之前）
                second_comma = rest.find(',')
                if second_comma == -1:
                    return full_match

                end = rest[:second_comma].strip()

                # 提取 color
                color_match = re.search(r'color\s*=\s*(\w+)', full_match)
                color = color_match.group(1) if color_match else 'WHITE'

                # 提取 dash_length（从 dashed_pattern=[x,y] 中取第一个值）
                dash_match = re.search(r'dashed_pattern\s*=\s*\[([0-9.]+)', full_match)
                dash = dash_match.group(1) if dash_match else '0.1'

                return f'DashedLine({start}, {end}, color={color}, dash_length={dash})'

            return re.sub(r'Line\s*\([^)]*dashed_pattern[^)]*\)', replacer, code)

        normalized = fix_dashed_line(normalized)
        normalized = self._enforce_left_right_layout(normalized)

        return normalized.strip() + "\n"

    def _strip_code_fences(self, code: str) -> str:
        """移除模型偶尔输出的 Markdown 代码块围栏。"""
        stripped = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
        return stripped

    def _enforce_left_right_layout(self, code: str) -> str:
        """
        对常见命名做轻量布局修正：
        - 几何对象默认左移到图形区
        - 公式对象默认靠右

        仅在代码未显式定位时追加，避免覆盖模型已有合理布局。
        """
        lines = code.splitlines()
        fixed_lines = []

        geom_name = re.compile(
            r"^\s*(point_|segment_|triangle_|fold_line_|right_angle|right_angle_mark|brace_|highlight_triangle)",
            flags=re.IGNORECASE,
        )
        geom_ctor = re.compile(r"=\s*(Dot|Line|Polygon|DashedLine|RightAngle|Brace)\(")
        formula_name = re.compile(r"^\s*(formula_|eq_)", flags=re.IGNORECASE)
        formula_ctor = re.compile(r"=\s*(MathTex|Text)\(")

        for line in lines:
            stripped = line.strip()

            if formula_name.search(stripped) and formula_ctor.search(line):
                has_pos = any(token in line for token in [".to_edge(", ".next_to(", ".move_to(", ".shift("])
                if not has_pos:
                    line = line.rstrip() + ".to_edge(RIGHT, buff=1.2)"

            if geom_name.search(stripped) and geom_ctor.search(line):
                has_pos = any(token in line for token in [".shift(", ".to_edge(", ".move_to(", ".next_to("])
                if not has_pos:
                    line = line.rstrip() + ".shift(LEFT*1.5)"

            fixed_lines.append(line)

        return "\n".join(fixed_lines)
