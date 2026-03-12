"""
测试 StoryboardAgent
输入直接使用 test_reasoning_agent_result.md 中 ReasoningAgent 的真实输出
"""

import json
from agents.storyboard_agent import StoryboardAgent
from config import AGENT_CONFIGS

# 直接使用 ReasoningAgent 的真实输出
MOCK_REASONING_RESULT = {
    "步骤": [
        {
            "序号": 1,
            "描述": "在$Rt\\triangle ABC$中，利用勾股定理计算斜边$AB$的长度",
            "公式": "$AB = \\sqrt{AC^2 + BC^2} = \\sqrt{8^2 + 6^2} = 10\\ \\text{cm}$"
        },
        {
            "序号": 2,
            "描述": "根据折叠的性质，对应边、对应角相等，计算$AC'$的长度并确定$\\triangle ADC'$为直角三角形",
            "公式": "$BC' = BC = 6\\ \\text{cm}$，$AC' = AB - BC' = 10 - 6 = 4\\ \\text{cm}$，$\\angle AC'D = \\angle C = 90^\\circ$"
        },
        {
            "序号": 3,
            "描述": "设未知量，将$AD$和$C'D$用含$x$的表达式表示",
            "公式": "设$CD = x$，则$C'D = x$，$AD = AC - CD = 8 - x$"
        },
        {
            "序号": 4,
            "描述": "在$Rt\\triangle ADC'$中应用勾股定理列方程",
            "公式": "$AC'^2 + C'D^2 = AD^2$，即$4^2 + x^2 = (8 - x)^2$"
        },
        {
            "序号": 5,
            "描述": "解方程求出$x$的值",
            "公式": "展开得$16 + x^2 = 64 - 16x + x^2$，化简得$16x = 48$，解得$x = 3$"
        },
        {
            "序号": 6,
            "描述": "利用直角三角形面积公式计算$\\triangle ADC'$的面积",
            "公式": "$S_{\\triangle ADC'} = \\frac{1}{2} \\times AC' \\times C'D = \\frac{1}{2} \\times 4 \\times 3 = 6\\ \\text{cm}^2$"
        }
    ],
    "最终答案": "$6\\ \\text{cm}^2$"
}


def print_separator(title=""):
    print("=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def test_generate_human_storyboard(agent: StoryboardAgent):
    """测试人类可读分镜生成"""
    print_separator("1. 测试 generate_human_storyboard")

    result = agent.generate_human_storyboard(MOCK_REASONING_RESULT)

    if "error" in result:
        print(f"❌ 失败：{result['error']}")
        print(f"   原始内容：{result.get('raw_content', '')}")
        return None

    print(f"✅ 调用成功！")
    print(f"   场景总数：{result.get('total_scenes', '未知')}")
    print()

    for scene in result.get("scenes", []):
        print(f"  【场景 {scene.get('scene_id')}】{scene.get('title')}")
        print(f"    画面描述：{scene.get('description')}")
        print(f"    关键元素：{scene.get('visual_elements')}")
        print(f"    解说词  ：{scene.get('narration')}")
        print(f"    转场方式：{scene.get('transition')}")
        print()

    return result


def test_generate_manim_storyboard(agent: StoryboardAgent, human_storyboard: dict):
    """测试 Manim 动画指令生成"""
    print_separator("2. 测试 generate_manim_storyboard")

    result = agent.generate_manim_storyboard(human_storyboard)

    scenes = result.get("scenes", [])
    failed = [s for s in scenes if "error" in s]
    success = [s for s in scenes if "error" not in s]

    print(f"✅ 调用完成！共 {len(scenes)} 个场景，成功 {len(success)} 个，失败 {len(failed)} 个")
    print()

    total_duration = 0.0
    for scene in scenes:
        scene_id = scene.get("scene_id", "?")
        if "error" in scene:
            print(f"  【场景 {scene_id}】❌ 解析失败：{scene['error']}")
            print(f"    原始内容：{scene.get('raw_content', '')[:200]}")
            print()
            continue

        duration = scene.get("estimated_duration", 0)
        total_duration += duration
        print(f"  【场景 {scene_id}】预估时长 {duration}s")
        print(f"    背景色  ：{scene.get('background')}")
        print(f"    公式    ：{scene.get('formula_display') or '无'}")

        objects = scene.get("manim_objects", [])
        print(f"    Manim对象（{len(objects)}个）：")
        for obj in objects:
            print(f"      - {obj.get('var_name')} [{obj.get('manim_class')}] "
                  f"颜色={obj.get('color')} 位置={obj.get('position')}")

        animations = scene.get("animations", [])
        print(f"    动画序列（{len(animations)}步）：")
        for anim in animations:
            print(f"      - {anim.get('action')}({anim.get('target')}) "
                  f"run_time={anim.get('run_time')}s")
        print()

    print(f"   全部场景预估总时长：{total_duration:.1f}s")
    return result


def test_design_storyboard_full(agent: StoryboardAgent):
    """独立测试 design_storyboard 主入口，验证完整链路健壮性"""
    print_separator("3. 测试完整流水线 design_storyboard（独立调用）")

    result = agent.design_storyboard(MOCK_REASONING_RESULT)

    has_human = "human_storyboard" in result and "error" not in result["human_storyboard"]
    manim_scenes = result.get("manim_storyboard", {}).get("scenes", [])
    failed = [s for s in manim_scenes if "error" in s]
    manim_ok = len(manim_scenes) > 0 and len(failed) == 0

    print(f"  human_storyboard: {'✅' if has_human else '❌'}")
    print(f"  manim_storyboard: {'✅ 全部通过' if manim_ok else f'❌ {len(failed)}/{len(manim_scenes)} 个场景失败'}")
    for s in failed:
        print(f"    场景 {s.get('scene_id', '?')} 失败：{s.get('error', '')}")

    if has_human and manim_ok:
        print("\n✅ 完整流水线测试通过！")
    else:
        print("\n❌ 链路存在问题，请检查上方输出")
    return result


if __name__ == "__main__":
    print_separator("开始测试 StoryboardAgent")
    print()

    agent = StoryboardAgent(config=AGENT_CONFIGS["storyboard"])

    # # 1. 测试人类可读分镜生成
    # human_result = test_generate_human_storyboard(agent)

    # # 2. 用第 1 步的结果测试 Manim 指令（复用同一份 human_result，不重复调用 LLM）
    # manim_result = None
    # if human_result:
    #     manim_result = test_generate_manim_storyboard(agent, human_result)

    # 3. 独立测试完整链路（重新发起调用，验证链路健壮性）
    test_design_storyboard_full(agent)

    print_separator("测试完成")
