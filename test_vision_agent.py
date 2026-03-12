"""
测试 VisionAgent
"""
from agents.vision_agent import VisionAgent
from config import AGENT_CONFIGS

# 初始化 VisionAgent
vision_agent = VisionAgent(config=AGENT_CONFIGS["vision"])

# 测试图片 URL（可以替换成你的本地路径或在线图片）
test_image = "D:\\code\\learn_advio\\agents\\PixPin_2026-03-12_00-03-18.png"

print("=" * 50)
print("开始测试 VisionAgent...")
print("=" * 50)

try:
    # 调用 process 方法
    result = vision_agent.process(test_image)

    print("\n✅ 测试成功！\n")
    print("OCR 结果:")
    print(result.get("ocr_result", "无"))
    print("\n建模结果:")
    print(result.get("modeling_result", "无"))

except Exception as e:
    print(f"\n❌ 测试失败：{e}")
    import traceback
    traceback.print_exc()
