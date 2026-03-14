"""
FigureComposerAgent 负责把题图层与讲解动画代码块组合成最终的 FullVideo，
并执行渲染、音频拼接与成片输出。
"""

from .base_agent import BaseAgent
from typing import Any, Dict, Optional
from config import DEFAULT_LLM_CONFIG
from langchain_core.prompts import ChatPromptTemplate
import json
import os
import subprocess
import re


class FigureComposerAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any], llm: Optional[Any] = None):
        self.config = config
        if not llm:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=DEFAULT_LLM_CONFIG["api_key"],
                base_url=DEFAULT_LLM_CONFIG["base_url"],
                model=DEFAULT_LLM_CONFIG["model"],
                temperature=config.get("temperature", 0.05),
                max_tokens=config.get("max_tokens", 8192),
            )
        super().__init__(config, llm)

    def process(
        self,
        geometry_ir: Dict[str, Any],
        animation_result: Dict[str, Any],
        voice_result: Dict[str, Any],
        problem_figure_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        组合题图与讲解动画，渲染最终视频。

        Args:
            geometry_ir: VisionAgent 输出的几何 IR
            animation_result: AnimationAgent 输出的 scene_animation_code 和 scaled_storyboard
            voice_result: VoiceAgent 输出的音频信息
            problem_figure_code: VisionAgent 直出的题图代码（优先使用）

        Returns:
            包含最终视频路径与中间代码路径的字典
        """
        output_dir = self.config.get("output_dir", "output")
        show_problem_figure = bool(self.config.get("show_problem_figure", False)) # false不显示题图
        selected_path = "direct_code"
        selected_code = (problem_figure_code or "").strip()

        if show_problem_figure:
            if not self._is_valid_problem_figure_code(selected_code):
                selected_path = "ir_fallback"
                selected_code = self.generate_problem_figure_code(geometry_ir)
        else:
            selected_path = "disabled"
            selected_code = "problem_group = Group()\n"

        full_video_code = self.compose_full_video_code(selected_code, animation_result["scene_animation_code"])
        full_video_code = self.sanitize_full_video_code(full_video_code)
        code_path = self.save_code(full_video_code, output_dir)
        video_path = self.render_video(code_path, output_dir)
        audio_path = self.concat_audio_files(voice_result, output_dir)
        final_path = self.merge_audio_video(video_path, audio_path, output_dir)
        return {
            "problem_figure_code": selected_code,
            "problem_figure_code_source": selected_path,
            "full_video_code_path": code_path,
            "final_video": final_path,
        }

    def _is_valid_problem_figure_code(self, code: str) -> bool:
        """判定 Vision 直出题图代码是否可用于优先路径。"""
        sanitized = self._strip_code_fences(code).strip()
        if not sanitized:
            return False
        if "class " in sanitized or "from manim import" in sanitized:
            return False
        if "problem_group" not in sanitized:
            return False
        if "image_meta[" in sanitized:
            return False
        return True

    def generate_problem_figure_code(self, geometry_ir: Dict[str, Any]) -> str:
        """
        生成题图基础代码。
        优先依据 geometry_ir 生成 problem_group；若几何信息不足，则回退为裁剪图常驻层。
        """
        primitives = geometry_ir.get("primitives", {})
        has_structured_geometry = any(primitives.get(name) for name in ["segments", "angles", "polygons", "labels"])

        if not has_structured_geometry:
            diagram_path = geometry_ir.get("source", {}).get("diagram_image_path", "")
            normalized = os.path.abspath(diagram_path).replace("\\", "/") if diagram_path else ""

            # 根据场景尺寸动态计算题图大小（占场景高度的 15%）
            scene_height = self.config.get("scene_height", 8)  # Manim 默认场景高度
            problem_img_height = scene_height * 0.15

            return (
                f'problem_img = ImageMobject(r"{normalized}").scale_to_fit_height({problem_img_height}).to_corner(UL, buff=0.25)\n'
                'problem_group = Group(problem_img)\n'
                'self.add(problem_group)\n'
            )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                你是一个 Manim 题图构建器，负责根据几何 IR 生成基础题图代码。

                输出要求：
                1. 只输出可插入 construct(self) 内部的 Python 代码片段。
                2. 必须创建 `problem_group` 变量，类型为 VGroup。
                3. 若存在裁剪图路径，可先创建 problem_img 放左上角，再在左半屏生成结构化题图。
                4. 若几何 IR 没有精确坐标，不要伪造复杂坐标，使用稳定布局生成示意图即可。
                5. 禁止输出类定义、import、render 逻辑。
                6. 题图代码只负责基础图层，不负责讲解动画。
                7. **题图尺寸规范**：题图高度应约为场景高度的 15%，例如：
                   - 默认场景 (14x8)：题图高度约 1.2
                   - 不要用固定值如 2.2，要根据场景比例调整
                """
            ),
            (
                "human",
                """
                请根据以下 geometry_ir 生成题图基础代码：

                {geometry_ir}
                """
            )
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "geometry_ir": json.dumps(geometry_ir, ensure_ascii=False, indent=2)
        })
        return response.content.strip() + "\n"

    def compose_full_video_code(self, problem_figure_code: str, scene_animation_code: str) -> str:
        """将题图层代码和讲解动画代码块包装成完整的 FullVideo 文件。"""
        problem_figure_code = self._strip_code_fences(problem_figure_code)
        scene_animation_code = self._strip_code_fences(scene_animation_code)
        scene_animation_code = self._fix_fadeout_all_mobjects(scene_animation_code)
        return (
            "from manim import *\n"
            "import numpy as np\n\n\n"
            "class FullVideo(Scene):\n"
            "    def construct(self):\n"
            "        self.camera.background_color = BLACK\n"
            "        def _safe_get_part(tex_obj, tex, idx=0):\n"
            "            parts = tex_obj.get_parts_by_tex(tex)\n"
            "            return parts[idx] if 0 <= idx < len(parts) else VGroup()\n"
            "        persistent_objects = []\n"
            "        def _mark_persistent(*objs):\n"
            "            for obj in objs:\n"
            "                if obj is None:\n"
            "                    continue\n"
            "                if all(obj is not p for p in persistent_objects):\n"
            "                    persistent_objects.append(obj)\n"
            "        def _fade_scene_objects(scene_objects):\n"
            "            removable = []\n"
            "            for obj in scene_objects:\n"
            "                if all(obj is not p for p in persistent_objects):\n"
            "                    removable.append(obj)\n"
            "            if removable:\n"
            "                self.play(FadeOut(*removable))\n"
            + self._indent_code(problem_figure_code, 8)
            + "\n"
            + self._indent_code(scene_animation_code, 8)
            + "\n"
            + "        self.wait(2)\n"
            + "        if 'problem_group' in locals():\n"
            + "            self.play(FadeOut(problem_group), run_time=0.5)\n"
            + "        self.wait(0.5)\n"
        )

    def sanitize_full_video_code(self, code: str) -> str:
        """最终保存前的代码清洗。"""
        normalized = self._strip_code_fences(code)
        normalized = (
            normalized.replace("，", ",")
            .replace("：", ":")
            .replace("；", ";")
            .replace("（", "(")
            .replace("）", ")")
        )
        normalized = normalized.replace("problem_group = VGroup(problem_img)", "problem_group = Group(problem_img)")
        normalized = normalized.replace(
            "self.add(problem_group)",
            "self.add(problem_group)\n        problem_group.set_z_index(1000)\n        self.add_foreground_mobject(problem_group)",
        )
        normalized = re.sub(
            r"\bPoint\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)",
            r"np.array([\1, \2, \3])",
            normalized,
        )
        normalized = self._normalize_tex_part_access(normalized)
        normalized = self._normalize_2d_coordinate_tuples(normalized)
        normalized = self._normalize_vector_shift_calls(normalized)
        normalized = self._recover_broken_scale_shift_calls(normalized)
        normalized = self._normalize_tex_submobject_indexing(normalized)
        normalized = self._normalize_dashed_pattern_lines(normalized)
        normalized = self._enforce_triangle_abc_persistence(normalized)
        normalized = self._rewrite_scene_fadeouts_to_preserve_base(normalized)
        if self.config.get("show_problem_figure", False):
            normalized = self._reinforce_problem_layer_after_scene_cleanup(normalized)
        pattern = re.compile(r"MathTex\(\s*(r?)([\"'])(.*?)\2", re.DOTALL)

        def replace_if_chinese(match: re.Match) -> str:
            if re.search(r"[\u4e00-\u9fff]", match.group(3)):
                return match.group(0).replace("MathTex(", "Text(", 1)
            return match.group(0)

        return pattern.sub(replace_if_chinese, normalized)

    def _normalize_tex_part_access(self, code: str) -> str:
        """
        兼容不同 Manim 版本的 MathTex 子串访问方式。

        将不兼容写法：
            m.get_part_by_tex("x", index=1)
            m.get_parts_by_tex("x", index=1)
        改写为：
            m.get_parts_by_tex("x")[1]
        """
        # m.get_part_by_tex(tex, index=n) -> m.get_parts_by_tex(tex)[n]
        code = re.sub(
            r"([A-Za-z_][A-Za-z0-9_]*)\.get_part_by_tex\(\s*([^,\)]+?)\s*,\s*index\s*=\s*(\d+)\s*\)",
            r"\1.get_parts_by_tex(\2)[\3]",
            code,
        )

        # m.get_parts_by_tex(tex, index=n) -> m.get_parts_by_tex(tex)[n]
        code = re.sub(
            r"([A-Za-z_][A-Za-z0-9_]*)\.get_parts_by_tex\(\s*([^,\)]+?)\s*,\s*index\s*=\s*(\d+)\s*\)",
            r"\1.get_parts_by_tex(\2)[\3]",
            code,
        )

        # m.get_parts_by_tex(tex)[n] -> _safe_get_part(m, tex, n)
        code = re.sub(
            r"([A-Za-z_][A-Za-z0-9_]*)\.get_parts_by_tex\(\s*([^\)]+?)\s*\)\[(\d+)\]",
            r"_safe_get_part(\1, \2, \3)",
            code,
        )

        # x_parts = [obj1, obj2] -> x_parts = VGroup(obj1, obj2)
        code = re.sub(
            r"^([ \t]*[A-Za-z_][A-Za-z0-9_]*_parts\s*=)\s*\[(.+)\]\s*$",
            r"\1 VGroup(\2)",
            code,
            flags=re.MULTILINE,
        )

        # FadeOut(x_parts) -> FadeOut(*x_parts)
        code = re.sub(
            r"FadeOut\(\s*([A-Za-z_][A-Za-z0-9_]*_parts)\s*\)",
            r"FadeOut(*\1)",
            code,
        )
        return code

    def _normalize_2d_coordinate_tuples(self, code: str) -> str:
        """将几何构造中的 (x, y) 统一转为 np.array([x, y, 0])。"""
        tuple_pattern = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
        geometry_tokens = ("Polygon(", "Line(", "Dot(", "move_to(")

        fixed_lines = []
        for line in code.splitlines():
            if any(token in line for token in geometry_tokens):
                line = tuple_pattern.sub(r"np.array([\1, \2, 0])", line)
            fixed_lines.append(line)
        return "\n".join(fixed_lines)

    def _rewrite_scene_fadeouts_to_preserve_base(self, code: str) -> str:
        """
        将场景清场改写为“保留基底对象”的安全模式。

        - self.play(FadeOut(*scene_objects)) -> _fade_scene_objects(scene_objects)
        - 在第一次清场前自动注册常驻对象（若存在）：triangle_abc, problem_group
        """
        marker = "self.play(FadeOut(*scene_objects))"
        rewritten = code.replace(marker, "_fade_scene_objects(scene_objects)")

        call_match = re.search(r"^\s{8}_fade_scene_objects\(scene_objects\)\s*$", rewritten, flags=re.MULTILINE)
        if call_match:
            first_idx = call_match.start()
            registration = (
                "        for _name in ('triangle_abc', 'problem_group'):\n"
                "            if _name in locals():\n"
                "                _mark_persistent(locals()[_name])\n"
            )
            rewritten = rewritten[:first_idx] + registration + rewritten[first_idx:]
        return rewritten

    def _reinforce_problem_layer_after_scene_cleanup(self, code: str) -> str:
        """每次场景清场后重新挂载题图到前景层，避免题图被覆盖或误删。"""
        cleanup_line = "self.play(FadeOut(*scene_objects))"
        reinforcement = (
            "self.play(FadeOut(*scene_objects))\n"
            "        if 'problem_group' in locals():\n"
            "            self.add(problem_group)\n"
            "            self.add_foreground_mobject(problem_group)"
        )
        return code.replace(cleanup_line, reinforcement)

    def _normalize_vector_shift_calls(self, code: str) -> str:
        """
        修复向量字面量误用 .shift(...) 的写法。

        仅转换类似：
            4*UP.shift(2*LEFT) -> (4*UP + 2*LEFT)
            ORIGIN.shift(LEFT) -> (ORIGIN + LEFT)
        不处理 mobject 方法链：mobj.scale(2).shift(...)
        """
        return re.sub(
            r"\b((?:\d+(?:\.\d+)?\s*\*\s*)?(?:ORIGIN|UP|DOWN|LEFT|RIGHT))\.shift\(([^\)]+)\)",
            lambda m: f"({m.group(1).strip()} + {m.group(2).strip()})",
            code,
        )

    def _recover_broken_scale_shift_calls(self, code: str) -> str:
        """恢复误改成 '.(scale(x) + y)' 的链式写法为 '.scale(x).shift(y)'。"""
        return re.sub(
            r"\.\(scale\(([^\)]+)\)\s*\+\s*([^\)]+)\)",
            r".scale(\1).shift(\2)",
            code,
        )

    def _normalize_tex_submobject_indexing(self, code: str) -> str:
        """
        降级不稳定的 MathTex 深层索引，避免运行时 IndexError。

        例如：
            eq[0][5:7] -> eq
            eq[1][2]   -> eq
        """
        code = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\[\d+\]\[\d+:\d+\]", r"\1", code)
        code = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\[\d+\]\[\d+\]", r"\1", code)
        return code

    def _normalize_dashed_pattern_lines(self, code: str) -> str:
        """将 Line(..., dashed_pattern=[...]) 兼容改写为 DashedLine(..., dash_length=...)."""
        def replacer(m: re.Match) -> str:
            full_match = m.group(0)
            inner_match = re.search(r'Line\s*\(([^)]+)\)', full_match)
            if not inner_match:
                return full_match
            inner = inner_match.group(1)

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
            rest = inner[first_comma + 1:].strip()
            second_comma = rest.find(',')
            if second_comma == -1:
                return full_match
            end = rest[:second_comma].strip()

            color_match = re.search(r'color\s*=\s*(\w+)', full_match)
            color = color_match.group(1) if color_match else 'WHITE'
            dash_match = re.search(r'dashed_pattern\s*=\s*\[([0-9.]+)', full_match)
            dash = dash_match.group(1) if dash_match else '0.1'

            return f'DashedLine({start}, {end}, color={color}, dash_length={dash})'

        return re.sub(r'Line\s*\([^)]*dashed_pattern[^)]*\)', replacer, code)

    def _enforce_triangle_abc_persistence(self, code: str) -> str:
        """保留场景1的 triangle_abc，移除后续场景中的同名重定义。"""
        lines = code.splitlines()
        seen_first = False
        rewritten = []

        for line in lines:
            if re.match(r"^\s*triangle_abc\s*=\s*Polygon\(", line):
                if not seen_first:
                    seen_first = True
                    rewritten.append(line)
                else:
                    indent = re.match(r"^(\s*)", line).group(1)
                    rewritten.append(f"{indent}# Reuse persistent triangle_abc from scene 1")
                continue
            rewritten.append(line)

        return "\n".join(rewritten)

    def _strip_code_fences(self, code: str) -> str:
        """移除模型输出中夹带的 Markdown 代码块围栏。"""
        stripped = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
        return stripped

    def save_code(self, code: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        code_path = os.path.join(output_dir, "animation.py")
        with open(code_path, "w", encoding="utf-8") as file:
            file.write(code)
        print(f"✓ FullVideo 代码已保存：{code_path}")
        return code_path

    def render_video(self, code_path: str, output_dir: str) -> str:
        cmd = [
            "manim", "render",
            "--media_dir", output_dir,
            "-ql",
            code_path,
            "FullVideo",
        ]
        print(f"  渲染完整视频：{' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ⚠️  渲染失败：{result.stderr[-500:]}")
            raise RuntimeError(f"Manim 渲染失败: {result.stderr[-300:]}")
        video_path = os.path.join(output_dir, "videos", "animation", "480p15", "FullVideo.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"渲染完成但未找到输出文件：{video_path}")
        print(f"  ✓ 渲染完成：{video_path}")
        return video_path

    def concat_audio_files(self, voice_result: Dict[str, Any], output_dir: str) -> str:
        """拼接为完整的音频文件"""
        scenes = sorted(voice_result.get("scenes", []), key=lambda scene: scene["scene_id"])
        filelist_path = os.path.join(output_dir, "audio_filelist.txt")
        with open(filelist_path, "w", encoding="utf-8") as file:
            for scene in scenes:
                abs_path = os.path.abspath(scene["audio_path"]).replace("\\", "/")
                file.write(f"file '{abs_path}'\n")
        merged_audio = os.path.join(output_dir, "merged_audio.mp3")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", filelist_path,
            "-c", "copy",
            merged_audio,
        ]
        print("  拼接音频轨道...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"音频拼接失败: {result.stderr[-300:]}")
        print(f"  ✓ 音频拼接完成：{merged_audio}")
        return merged_audio

    def merge_audio_video(self, video_path: str, audio_path: str, output_dir: str) -> str:
        """使用 ffmpeg 将音频与视频合并为最终输出文件。"""
        final_path = os.path.join(output_dir, "final_output.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            final_path,
        ]
        print("  合并音视频...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"音视频合并失败: {result.stderr[-300:]}")
        print(f"  ✓ 最终视频已生成：{final_path}")
        return final_path

    def _indent_code(self, code: str, spaces: int) -> str:
        indent = " " * spaces
        lines = code.strip("\n").splitlines()
        return "\n".join(f"{indent}{line}" if line.strip() else "" for line in lines) + "\n"

    def _fix_fadeout_all_mobjects(self, code: str) -> str:
        """
        修复 AnimationAgent 生成的 `FadeOut(*self.mobjects)` 问题。

        将每个场景的 FadeOut(*self.mobjects) 替换为 FadeOut(*scene_X_objects)，
        其中 scene_X_objects 是该场景开头定义的对象列表。

        注意：这是一个兜底修复，最佳方案是让 AnimationAgent 直接生成正确的代码。
        """
        import re

        # 匹配场景注释和对应的 FadeOut(*self.mobjects)
        # 例如：# === 场景 1 ===\n... self.play(FadeOut(*self.mobjects))
        pattern = re.compile(
            r'(# === 场景 (\d+) ===)\s*\n([\s\S]*?)\n(\s*)self\.play\(FadeOut\(\*self\.mobjects\)(?:,\s*run_time=[\d.]+)?\)',
            re.MULTILINE
        )

        def replace_fadeout(match: re.Match) -> str:
            scene_header = match.group(1)
            scene_num = match.group(2)
            scene_content = match.group(3)
            indent = match.group(4)

            # 提取该场景中定义的所有变量名（等号左边且首字母小写的）
            var_pattern = re.compile(
                r'^\s*([a-z][a-z0-9_]*)\s*=\s*(?:Group|VGroup|MathTex|Text|Tex|'
                r'Polygon|Line|Circle|Arc|Dot|Square|Rectangle|Arrow|Brace|'
                r'Number|DecimalNumber|SurroundingRectangle|ImageMobject)',
                re.MULTILINE
            )
            variables = var_pattern.findall(scene_content)

            # 过滤掉一些常见的非对象变量
            excluded = {'offset', 'scale', 'angle', 'color', 'point', 'pos'}
            variables = [v for v in variables if v not in excluded]

            if variables:
                # 生成 scene_X_objects 列表
                scene_objects_name = f"scene_{scene_num}_objects"
                obj_list = ", ".join(variables)

                # 在场景开头插入对象列表定义
                new_content = f"{scene_header}\n{indent}{scene_objects_name} = [{obj_list}]"

                # 替换 FadeOut
                new_fadeout = f"\n{indent}self.play(FadeOut(*{scene_objects_name}))"

                return new_content + new_fadeout
            else:
                # 如果没找到变量，保留原样（或者可以移除这行 FadeOut）
                return match.group(0)

        return pattern.sub(replace_fadeout, code)
