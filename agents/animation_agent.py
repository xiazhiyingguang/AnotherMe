"""
动画 Agent 负责根据分镜设计生成 manim 动画代码，供后续渲染使用。
"""

from .base_agent import BaseAgent
from typing import Any, Dict, Optional, List
from config import DEFAULT_LLM_CONFIG
from langchain_core.prompts import ChatPromptTemplate
import json
import os
import subprocess

class AnimationAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any], llm: Optional[Any] = None):
        """
        初始化 AnimationAgent

        Args:
            param config: 配置字典，包含动画生成相关配置
        """
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

    def process(self, manim_storyboard: Dict[str, Any], voice_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        完整动画生成流水线：生成代码 → 保存 → 渲染整体视频 → 合并音频轨道 → 最终合成

        Args:
            manim_storyboard: StoryboardAgent 输出的 manim_storyboard
            voice_result: VoiceAgent 输出的音频信息

        Returns:
            包含最终视频路径的字典
        """
        output_dir = self.config.get("output_dir", "output")
        code = self.generate_manim_code(manim_storyboard, voice_result)
        code_path = self.save_code(code, output_dir)
        video_path = self.render_video(code_path, output_dir)
        audio_path = self.concat_audio_files(voice_result, output_dir)
        final_path = self.merge_audio_video(video_path, audio_path, output_dir)
        return {"final_video": final_path}


    def generate_manim_code(self, manim_storyboard: Dict[str, Any], voice_result: Dict[str, Any]) -> str:
            """
            根据 Manim 分镜设计生成完整的 Manim Python 代码。
            使用方案一：按比例缩放每个动画的 run_time，使动画总时长与音频时长对齐。

            Args:
                manim_storyboard: StoryboardAgent 输出的 manim_storyboard，包含每个场景的动画指令
                voice_result: VoiceAgent 输出的音频信息，包含每个场景的音频路径和时长

            Returns:
                完整的 Manim Python 代码字符串
            """
            # 构建 scene_id -> real_duration 的快速查找表
            voice_map = {
                s["scene_id"]: s["duration"]
                for s in voice_result.get("scenes", [])
            }

            # 方案一：按比例缩放每个场景中所有动画的 run_time
            scaled_scenes = []
            for scene in manim_storyboard.get("scenes", []):
                scene = dict(scene)  # 浅拷贝，避免修改原数据
                scene_id = scene.get("scene_id")
                estimated = scene.get("estimated_duration", 1.0)
                real = voice_map.get(scene_id, estimated)

                # 更新每一个动画的 run_time
                if estimated > 0:
                    scale = real / estimated
                    scene["animations"] = [
                        {**anim, "run_time": round(anim["run_time"] * scale, 2)}
                        for anim in scene.get("animations", [])
                    ]
                scene["estimated_duration"] = real  # 更新整体的时长为真实时长
                scaled_scenes.append(scene)

            scaled_storyboard = {
                "total_scenes": len(scaled_scenes),
                "scenes": scaled_scenes,
            }

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """你是一个 Manim 动画代码生成专家，负责根据结构化的 Manim 动画指令生成可直接运行的 Python 代码。

【核心架构要求】
整个视频只生成 **一个** 继承自 Scene 的类 `FullVideo`，所有场景都写在同一个 `construct(self)` 方法中，
用注释 `# === 场景 X ===` 分隔各段。这样对象可以跨场景持续存在，不会在场景切换时消失。

【对象生命周期管理】
- 首次出现的对象：用 `self.play(Create(...))` 或 `self.play(Write(...))` 带动画创建
- 需要在后续场景继续显示的对象：创建后不要 FadeOut，直接保留在画面中
- 需要在后续场景新增标注/高亮的对象：用 `self.play(Indicate(...))` 或改变颜色强调
- 确实不再需要的对象：用 `self.play(FadeOut(...))` 显式移除
- 每个场景结束后用 `self.wait(0.3)` 短暂定格

【场景间过渡】
每个场景段结束时，根据 transition 字段决定：
- "淡入淡出"：新增元素用 FadeIn，移除元素用 FadeOut
- "直切"：直接用 self.add() / self.remove() 不加动画
- "结束"：最后一个场景，等待后结束

代码规范：
1. 文件顶部固定写 `from manim import *`
2. 只生成一个类：`class FullVideo(Scene):`
3. 只实现 `construct(self)` 方法
4. 背景色在 construct 开头用 `self.camera.background_color = BLACK` 设置
5. 只输出纯 Python 代码，不要加任何 markdown 代码块标记或解释文字

布局与防重叠规范（必须严格遵守）：
6. 屏幕坐标系范围：x ∈ [-7, 7]，y ∈ [-4, 4]，原点 ORIGIN 在屏幕中心
7. 几何图形统一放置在屏幕左半区（x < 0），公式和文字统一放置在屏幕右半区（x > 0）或图形下方
8. 标注文字（边长、角度等）必须用 .next_to(目标对象, 方向, buff=0.2) 紧靠目标放置，禁止使用固定坐标
9. 公式块统一放在屏幕下方：使用 .to_edge(DOWN, buff=0.5) 或 .move_to(DOWN * 2.5)
10. 多行公式之间用 VGroup 组合后整体定位，避免各自独立定位导致重叠
11. 每个变量名在整个 construct 中只能定义一次，后续场景若要修改同一对象用 Transform 或 become()

禁止事项（会导致视觉错误，绝对不允许）：
12. 禁止创建 NumberPlane、Axes、CoordinateSystem 等坐标系对象，除非题目明确需要
13. 禁止创建任何用于"辅助定位"的不可见或多余几何对象
14. 直角符号只能用 RightAngle(line1, line2, length=0.2) 创建，禁止用小 Polygon 或 Square 模拟
15. 禁止创建颜色为 RED 的辅助线，除非题目图形本身需要红色元素"""
                ),
                (
                    "human",
                    """以下是已按音频时长缩放后的 Manim 动画指令 JSON，请生成完整的 Manim Python 代码。
记住：所有场景写在同一个 FullVideo(Scene) 类中，用注释分隔，对象可以跨场景持续存在。

{scaled_storyboard}

请输出可直接运行的完整 Python 代码。"""
                )
            ])

            chain = prompt | self.llm
            response = chain.invoke({
                "scaled_storyboard": json.dumps(scaled_storyboard, ensure_ascii=False, indent=2)
            })
            return response.content



    def save_code(self, code: str, output_dir: str = "output/manim") -> str:
        """
        将生成的 Manim Python 代码保存到文件。

        Args:
            code: generate_manim_code 返回的 Python 代码字符串
            output_dir: 输出目录

        Returns:
            保存的代码文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        code_path = os.path.join(output_dir, "animation.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"✓ Manim 代码已保存：{code_path}")
        return code_path

    def render_video(self, code_path: str, output_dir: str) -> str:
        """
        调用 manim 命令行渲染整体视频（单一 FullVideo 类）。

        Args:
            code_path: Manim Python 代码文件路径
            output_dir: 输出目录

        Returns:
            渲染完成的视频文件路径
        """
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
        # manim 输出路径：media/videos/animation/480p15/FullVideo.mp4
        video_path = os.path.join(output_dir, "videos", "animation", "480p15", "FullVideo.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"渲染完成但未找到输出文件：{video_path}")
        print(f"  ✓ 渲染完成：{video_path}")
        return video_path

    def concat_audio_files(self, voice_result: Dict[str, Any], output_dir: str) -> str:
        """
        用 ffmpeg 将所有场景音频按顺序拼接为一条完整音轨。

        Args:
            voice_result: VoiceAgent 输出的音频信息
            output_dir: 输出目录

        Returns:
            拼接后的音频文件路径
        """
        scenes = sorted(voice_result.get("scenes", []), key=lambda s: s["scene_id"])
        filelist_path = os.path.join(output_dir, "audio_filelist.txt")
        with open(filelist_path, "w", encoding="utf-8") as f:
            for s in scenes:
                abs_path = os.path.abspath(s["audio_path"]).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")
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
        """
        用 ffmpeg 将完整视频与完整音轨合并为最终输出。

        Args:
            video_path: 静默视频文件路径
            audio_path: 完整音频文件路径
            output_dir: 输出目录

        Returns:
            最终视频文件路径
        """
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



    
