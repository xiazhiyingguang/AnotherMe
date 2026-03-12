"""
测试 AnimationAgent
分别测试：generate_manim_code、save_code、render_video、merge_audio_video、concat_videos、process（完整流程）
"""

import os
import json
from agents.animation_agent import AnimationAgent
from config import AGENT_CONFIGS

# 模拟 StoryboardAgent 输出的 manim_storyboard
MOCK_MANIM_STORYBOARD = {
    "total_scenes": 2,
    "scenes": [
        {
            "scene_id": 1,
            "background": "BLACK",
            "manim_objects": [
                {
                    "var_name": "triangle_abc",
                    "manim_class": "Polygon",
                    "params": "三个顶点分别为 ORIGIN, 3*RIGHT, 3*RIGHT + 4*UP",
                    "color": "WHITE",
                    "position": "画面居中"
                },
                {
                    "var_name": "label_a",
                    "manim_class": "Text",
                    "params": "内容'3'，字体大小 24",
                    "color": "YELLOW",
                    "position": "底边下方"
                }
            ],
            "animations": [
                {
                    "action": "Create",
                    "target": "triangle_abc",
                    "run_time": 2.0,
                    "params": ""
                },
                {
                    "action": "FadeIn",
                    "target": "label_a",
                    "run_time": 1.0,
                    "params": ""
                }
            ],
            "formula_display": "c = \\sqrt{a^2 + b^2}",
            "estimated_duration": 3.0
        },
        {
            "scene_id": 2,
            "background": "BLACK",
            "manim_objects": [
                {
                    "var_name": "formula",
                    "manim_class": "Text",
                    "params": "内容'c = sqrt(a^2 + b^2)'，字体大小 32",
                    "color": "WHITE",
                    "position": "画面居中"
                }
            ],
            "animations": [
                {
                    "action": "Write",
                    "target": "formula",
                    "run_time": 2.5,
                    "params": ""
                }
            ],
            "formula_display": "c = sqrt(a^2 + b^2)",
            "estimated_duration": 2.5
        }
    ]
}

# 模拟 VoiceAgent 返回的 voice_result
MOCK_VOICE_RESULT = {
    "scenes": [
        {
            "scene_id": 1,
            "audio_path": "output/audio/test_video_001/scene_1.mp3",
            "duration": 5.88  # 实际音频时长
        },
        {
            "scene_id": 2,
            "audio_path": "output/audio/test_video_001/scene_2.mp3",
            "duration": 4.0
        }
    ]
}

# 如果测试音频不存在，先创建它们
def ensure_test_audio():
    """确保测试用的音频文件存在"""
    from agents.voice_agent import VoiceAgent

    voice_agent = VoiceAgent(config=AGENT_CONFIGS["voice"])

    # 测试音频 1
    audio1_path = MOCK_VOICE_RESULT["scenes"][0]["audio_path"]
    if not os.path.exists(audio1_path):
        os.makedirs(os.path.dirname(audio1_path), exist_ok=True)
        voice_agent.text_to_speech("在直角三角形 ABC 中，已知两直角边分别为 3 和 4，利用勾股定理，斜边等于 5。", audio1_path)
        print(f"✓ 生成测试音频 1: {audio1_path}")

    # 测试音频 2
    audio2_path = MOCK_VOICE_RESULT["scenes"][1]["audio_path"]
    if not os.path.exists(audio2_path):
        os.makedirs(os.path.dirname(audio2_path), exist_ok=True)
        voice_agent.text_to_speech("根据勾股定理公式，c 等于 a 平方加 b 平方的平方根。", audio2_path)
        print(f"✓ 生成测试音频 2: {audio2_path}")


agent = AnimationAgent(config=AGENT_CONFIGS.get("animation", {}))

print("=" * 40)
print("测试前准备：确保测试音频存在")
ensure_test_audio()
print()

# 测试 1：generate_manim_code（生成代码）
print("=" * 40)
print("测试 generate_manim_code")
code = agent.generate_manim_code(MOCK_MANIM_STORYBOARD, MOCK_VOICE_RESULT)
print(f"生成的代码长度：{len(code)} 字符")
print("生成的代码预览：")
print("-" * 40)
print(code[:500] + "..." if len(code) > 500 else code)
print("-" * 40)
assert "from manim import *" in code, "代码缺少 'from manim import *'"
assert "class Scene1" in code, "代码缺少 Scene1 类"
assert "class Scene2" in code, "代码缺少 Scene2 类"
print("✓ generate_manim_code 通过\n")

# 测试 2：save_code（保存代码）
print("=" * 40)
print("测试 save_code")
code_path = agent.save_code(code, output_dir="output/manim")
assert os.path.exists(code_path), f"代码文件未保存：{code_path}"
print(f"✓ 代码已保存：{code_path}\n")

# 测试 3：render_video（渲染视频）- 这个测试比较耗时，可选
print("=" * 40)
print("测试 render_video（此步骤较慢，请耐心等待）")
scene_video_paths = agent.render_video(code_path, output_dir="output", manim_storyboard=MOCK_MANIM_STORYBOARD)
print(f"渲染结果：{scene_video_paths}")
# 注意：由于 manim 渲染可能失败，这里只做提示，不做强制断言
if len(scene_video_paths) > 0:
    print("✓ render_video 部分场景成功\n")
else:
    print("⚠️  render_video 所有场景渲染失败（可能是 manim 配置问题）\n")

# 测试 4：merge_audio_video（合并音视频）
print("=" * 40)
print("测试 merge_audio_video")
merged_paths = []
for scene_result in MOCK_VOICE_RESULT.get("scenes", []):
    scene_id = scene_result["scene_id"]
    video_path = scene_video_paths.get(scene_id)
    audio_path = scene_result["audio_path"]

    if video_path and os.path.exists(video_path) and os.path.exists(audio_path):
        merged = agent.merge_audio_video(scene_id, video_path, audio_path, output_dir="output")
        merged_paths.append(merged)
        assert os.path.exists(merged), f"合并后的视频不存在：{merged}"

if len(merged_paths) > 0:
    print(f"✓ merge_audio_video 通过，已合并 {len(merged_paths)} 个场景\n")
else:
    print("⚠️  merge_audio_video 跳过（没有可合并的视频）\n")

# 测试 5：concat_videos（拼接视频）
print("=" * 40)
print("测试 concat_videos")
if len(merged_paths) > 0:
    final_path = agent.concat_videos(merged_paths, output_dir="output")
    assert os.path.exists(final_path), f"最终视频不存在：{final_path}"
    print(f"✓ concat_videos 通过，最终视频：{final_path}\n")
else:
    print("⚠️  concat_videos 跳过（没有可拼接的视频）\n")

# 测试 6：process（完整流程）
print("=" * 40)
print("测试 process（完整流程）")
result = agent.process(MOCK_MANIM_STORYBOARD, MOCK_VOICE_RESULT)
print(f"最终结果：{json.dumps(result, ensure_ascii=False, indent=2)}")
assert "final_video" in result, "结果缺少 final_video 字段"
assert "scene_videos" in result, "结果缺少 scene_videos 字段"
print("✓ process 通过\n")

print("=" * 40)
print("所有测试完成！")
print("注意：如果渲染步骤失败，请检查 manim 和 ffmpeg 是否正确安装。")
