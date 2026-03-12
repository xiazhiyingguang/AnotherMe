"""
端到端流水线测试
完整流程：VisionAgent → ReasoningAgent → StoryboardAgent → VoiceAgent → AnimationAgent

使用方法：
    python test_pipeline.py                        # 使用默认测试图片
    python test_pipeline.py path/to/your_image.png # 使用指定图片
"""

import sys
import os
import json
import time

# ==================== 配置测试图片路径 ====================
# 优先使用命令行参数，否则用默认路径（替换为你的实际图片）
TEST_IMAGE = sys.argv[1] if len(sys.argv) > 1 else r"D:\code\AnotherMe\agents\PixPin_2026-03-12_00-03-18.png"

# ==================== 自动编号输出目录 ====================
def get_next_run_dir(base: str = "output") -> str:
    """扫描 output/ 下已有的 run_XXX 目录，返回下一个编号的路径"""
    os.makedirs(base, exist_ok=True)
    existing = [
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.startswith("run_")
    ]
    numbers = []
    for d in existing:
        try:
            numbers.append(int(d.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(numbers, default=0) + 1
    run_dir = os.path.join(base, f"run_{next_num:03d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

RUN_DIR = get_next_run_dir()

# ==================== 初始化所有 Agent ====================
print("=" * 60)
print("初始化 Agent...")
print("=" * 60)

from config import AGENT_CONFIGS
from agents.vision_agent import VisionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.storyboard_agent import StoryboardAgent
from agents.voice_agent import VoiceAgent
from agents.animation_agent import AnimationAgent

vision_agent    = VisionAgent(config=AGENT_CONFIGS["vision"])
reasoning_agent = ReasoningAgent(config=AGENT_CONFIGS["reasoning"])
storyboard_agent = StoryboardAgent(config=AGENT_CONFIGS["storyboard"])
# 将本次运行的输出目录注入 VoiceAgent 和 AnimationAgent
voice_config = {**AGENT_CONFIGS["voice"], "output_dir": os.path.join(RUN_DIR, "audio")}
animation_config = {**AGENT_CONFIGS["animation"], "output_dir": RUN_DIR}
voice_agent     = VoiceAgent(config=voice_config)
animation_agent = AnimationAgent(config=animation_config)
print(f"✓ 所有 Agent 初始化完成")
print(f"  本次输出目录：{RUN_DIR}\n")


def step(name: str):
    print("=" * 60)
    print(f"  {name}")
    print("=" * 60)


def save_checkpoint(name: str, data):
    """将中间结果保存到本次运行目录下，方便排查问题"""
    ckpt_dir = os.path.join(RUN_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  📄 中间结果已保存：{path}")


# ==================== Step 1: VisionAgent ====================
step("Step 1 / 5  VisionAgent — OCR 识别与数学建模")
t0 = time.time()
try:
    vision_result = vision_agent.process(TEST_IMAGE)
    print(f"  ✓ 完成（{time.time()-t0:.1f}s）")
    print(f"  OCR 结果（前200字）：{str(vision_result.get('ocr_result',''))[:200]}")
    save_checkpoint("01_vision", vision_result)
except Exception as e:
    print(f"  ❌ VisionAgent 失败：{e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ==================== Step 2: ReasoningAgent ====================
step("Step 2 / 5  ReasoningAgent — 逐步解题")
t0 = time.time()
try:
    reasoning_result = reasoning_agent.process(vision_result)
    print(f"  ✓ 完成（{time.time()-t0:.1f}s）")
    steps = reasoning_result.get("步骤", [])
    print(f"  共 {len(steps)} 个解题步骤")
    for s in steps[:3]:
        print(f"    步骤{s.get('序号')}: {s.get('描述','')[:60]}")
    if len(steps) > 3:
        print(f"    ...")
    save_checkpoint("02_reasoning", reasoning_result)
except Exception as e:
    print(f"  ❌ ReasoningAgent 失败：{e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ==================== Step 3: StoryboardAgent ====================
step("Step 3 / 5  StoryboardAgent — 生成分镜脚本")
t0 = time.time()
try:
    storyboard_result = storyboard_agent.process(reasoning_result)
    human_storyboard = storyboard_result["human_storyboard"]
    manim_storyboard = storyboard_result["manim_storyboard"]
    print(f"  ✓ 完成（{time.time()-t0:.1f}s）")
    print(f"  共 {human_storyboard.get('total_scenes')} 个场景")
    for scene in human_storyboard.get("scenes", [])[:2]:
        print(f"    场景{scene['scene_id']}「{scene['title']}」：{scene['narration'][:40]}...")
    save_checkpoint("03_human_storyboard", human_storyboard)
    save_checkpoint("03_manim_storyboard", manim_storyboard)
except Exception as e:
    print(f"  ❌ StoryboardAgent 失败：{e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ==================== Step 4: VoiceAgent ====================
step("Step 4 / 5  VoiceAgent — 生成语音音频")
t0 = time.time()
try:
    voice_result = voice_agent.process(human_storyboard)
    print(f"  ✓ 完成（{time.time()-t0:.1f}s）")
    for s in voice_result.get("scenes", []):
        print(f"    场景{s['scene_id']}：{s['audio_path']}  时长 {s['duration']:.2f}s")
    save_checkpoint("04_voice", voice_result)
except Exception as e:
    print(f"  ❌ VoiceAgent 失败：{e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ==================== Step 5: AnimationAgent ====================
step("Step 5 / 5  AnimationAgent — 生成动画并合成视频")
t0 = time.time()
try:
    animation_result = animation_agent.process(manim_storyboard, voice_result)
    print(f"  ✓ 完成（{time.time()-t0:.1f}s）")
    print(f"  最终视频：{animation_result.get('final_video')}")
    save_checkpoint("05_animation", animation_result)
except Exception as e:
    print(f"  ❌ AnimationAgent 失败：{e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ==================== 汇总 ====================
print()
print("=" * 60)
print("✅ 全流程测试完成！")
print(f"   本次输出目录：{RUN_DIR}")
print(f"   最终视频：{animation_result.get('final_video')}")
print(f"   中间结果：{os.path.join(RUN_DIR, 'checkpoints')}")
print("=" * 60)
