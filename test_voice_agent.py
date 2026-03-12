"""
测试 VoiceAgent
分别测试：extract_narration、text_to_speech、get_audio_duration、process
"""

import os
import json
from agents.voice_agent import VoiceAgent
from config import AGENT_CONFIGS

# 模拟 StoryboardAgent 输出的 human_storyboard
MOCK_STORYBOARD = {
    "scenes": [
        {
            "scene_id": 1,
            "title": "计算斜边长度",
            "description": "展示直角三角形，标注已知边长，利用勾股定理求斜边",
            "narration": "在直角三角形ABC中，已知两直角边分别为8厘米和6厘米，利用勾股定理，斜边AB等于10厘米。",
            "transition": "淡入淡出"
        },
        {
            "scene_id": 2,
            "title": "折叠性质与等量关系",
            "description": "展示折叠过程，标注对应边相等",
            "narration": "根据折叠的性质，对应边相等，所以AC'等于4厘米，且角AC'D为直角。",
            "transition": "淡入淡出"
        },
    ]
}

agent = VoiceAgent(config=AGENT_CONFIGS["voice"])

# 测试1：extract_narration
print("=" * 40)
print("测试 extract_narration")
scene = MOCK_STORYBOARD["scenes"][0]
narration = agent.extract_narration(scene)
print(f"场景1解说词：{narration}")
assert narration == scene["narration"], "extract_narration 结果不匹配"
print("✓ extract_narration 通过\n")

# 测试2：text_to_speech（单条）
print("=" * 40)
print("测试 text_to_speech")
os.makedirs("output/audio", exist_ok=True)
test_audio_path = "output/audio/test_single.mp3"
agent.text_to_speech("这是一段测试语音，用于验证语音生成功能。", test_audio_path)
assert os.path.exists(test_audio_path), "音频文件未生成"
print(f"✓ 音频文件已生成：{test_audio_path}\n")

# 测试3：get_audio_duration
print("=" * 40)
print("测试 get_audio_duration")
duration = agent.get_audio_duration(test_audio_path)
print(f"音频时长：{duration:.2f} 秒")
assert duration > 0, "音频时长应大于0"
print("✓ get_audio_duration 通过\n")

# 测试4：process（完整流程，不带 video_id）
print("=" * 40)
print("测试 process（完整流程，不带 video_id）")
result = agent.process(MOCK_STORYBOARD)
print(json.dumps(result, ensure_ascii=False, indent=2))
assert len(result["scenes"]) == 2, "场景数量不匹配"
for scene_result in result["scenes"]:
    assert os.path.exists(scene_result["audio_path"]), f"音频文件不存在：{scene_result['audio_path']}"
    assert scene_result["duration"] > 0, "时长应大于0"
print("✓ process 通过\n")

# 测试5：process（带 video_id）
print("=" * 40)
print("测试 process（带 video_id）")
result_with_id = agent.process(MOCK_STORYBOARD, video_id="test_video_001")
print(json.dumps(result_with_id, ensure_ascii=False, indent=2))
assert len(result_with_id["scenes"]) == 2, "场景数量不匹配"
for scene_result in result_with_id["scenes"]:
    assert os.path.exists(scene_result["audio_path"]), f"音频文件不存在：{scene_result['audio_path']}"
    assert scene_result["duration"] > 0, "时长应大于0"
    # 验证路径包含 video_id
    assert "test_video_001" in scene_result["audio_path"], "音频路径应包含 video_id 子目录"
print("✓ process (带 video_id) 通过\n")

print("=" * 40)
print("所有测试通过！")
