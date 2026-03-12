"""
生成音频的 Agent - 负责将分镜脚本中的解说词转换为语音文件，供视频配音使用。
"""

import edge_tts
import asyncio
import os
from .base_agent import BaseAgent
from typing import Any, Dict, Optional, List

class VoiceAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any],llm: Optional[Any] = None):
        """
        初始化 VoiceAgent

        Args:
            param config: 配置字典，包含 TTS 模型或服务的相关配置
        """
        self.config = config
        if not llm:
            from config import VOICE_MODEL_CONFIG
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=VOICE_MODEL_CONFIG["api_key"],
                base_url=VOICE_MODEL_CONFIG["base_url"],
                model=VOICE_MODEL_CONFIG["model"],
                temperature=config.get("temperature", 0.05),
                max_tokens=config.get("max_tokens", 2048),
            )

        super().__init__(config, llm)

    def process(self, storyboard: Dict[str, Any], video_id: str = None) -> Dict[str, Any]:
        """
        完整语音生成流水线：提取解说词 → 并行生成所有场景音频 → 获取时长

        Args:
            storyboard: StoryboardAgent 输出的 human_storyboard，包含多个场景
            video_id: 视频 ID，用于创建独立的音频目录。如果为 None，则直接输出到 output_dir

        Returns:
            包含每个场景音频路径和时长的字典列表
        """
        output_dir = self.config.get("output_dir", "output/audio")
        if video_id:
            output_dir = os.path.join(output_dir, video_id)
        os.makedirs(output_dir, exist_ok=True)

        scenes = storyboard.get("scenes", [])

        async def generate_all():
            tasks = []
            for scene in scenes:
                scene_id = scene.get("scene_id")
                narration = self.extract_narration(scene)
                output_path = os.path.join(output_dir, f"scene_{scene_id}.mp3")
                tasks.append((scene_id, output_path, narration))

            # 带重试的单场景生成函数
            max_retries = 2
            async def save_with_retry(scene_id, output_path, narration):
                for attempt in range(1, max_retries + 1):
                    try:
                        await edge_tts.Communicate(narration, voice="zh-CN-XiaoxiaoNeural").save(output_path)
                        print(f"    ✓ 场景 {scene_id} 音频生成完成")
                        return
                    except Exception as e:
                        print(f"    ⚠️  场景 {scene_id} 第 {attempt} 次失败：{type(e).__name__}，{'重试中...' if attempt < max_retries else '已放弃'}")
                        if attempt < max_retries:
                            await asyncio.sleep(2 * attempt)  # 指数退避
                        else:
                            raise

            await asyncio.gather(*[
                save_with_retry(scene_id, output_path, narration)
                for scene_id, output_path, narration in tasks
            ])
            return tasks

        tasks = asyncio.run(generate_all())

        results = []
        for scene_id, output_path, _ in tasks:
            duration = self.get_audio_duration(output_path)
            results.append({
                "scene_id": scene_id,
                "audio_path": output_path,
                "duration": duration,
            })
        return {"scenes": results}




    def extract_narration(self, scene: Dict[str, Any]) -> str:
        """
        从单个场景中提取解说词

        Args:
            scene: 单个分镜场景字典，包含 narration 字段

        Returns:
            该场景的解说词文本
        """
        return scene.get("narration", "")

    def get_audio_duration(self, audio_path: str) -> float:
        """
        获取音频文件的时长

        Args:
            audio_path: 音频文件路径

        Returns:
            音频时长（秒）
        """
        from mutagen.mp3 import MP3
        audio = MP3(audio_path)
        return audio.info.length
    


    def text_to_speech(self, text: str, output_path: str) -> None:
        """
        将输入文本转换为语音文件

        Args:
            text: 要转换的文本
            output_path: 输出语音文件的路径
        """
        # 使用 edge-tts 库进行文本到语音的转换
        communicate = edge_tts.Communicate(text, voice = "zh-CN-XiaoxiaoNeural")
        asyncio.run(communicate.save(output_path))



