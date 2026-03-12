"""
测试 ReasoningAgent
"""

from agents.reasoning_agent import ReasoningAgent
from config import AGENT_CONFIGS

# 模拟的建模结果（来自 VisionAgent）
MOCK_MODELING_RESULT = {
    "题目类型": "填空题",
    "已知条件": "在$Rt\\triangle ABC$中，$\\angle C = 90^{\\circ}$，$BC = 6cm$，$AC = 8cm$，将$\\triangle BCD$沿$BD$折叠，使点$C$落在$AB$边的点$C'$",
    "求解目标": "$\\triangle ADC'$的面积",
    "约束条件": "折叠前后对应边相等，对应角相等",
    "变量定义": "设$CD = x$，则$C'D = x$，$AD = 8 - x$，$BC' = BC = 6$，$AC' = AB - BC'$，先根据勾股定理求出$AB=\\sqrt{AC^{2}+BC^{2}}=\\sqrt{8^{2}+6^{2}} = 10$，所以$AC' = 10 - 6 = 4$",
    "方程组": "在$Rt\\triangle ADC'$中，根据勾股定理$AC'^{2}+C'D^{2}=AD^{2}$，即$4^{2}+x^{2}=(8 - x)^{2}$"
}


def test_reasoning_agent():
    """测试 ReasoningAgent 生成解题步骤"""
    print("=" * 50)
    print("开始测试 ReasoningAgent...")
    print("=" * 50)

    # 初始化 Agent
    reasoning_agent = ReasoningAgent(config=AGENT_CONFIGS["reasoning"])

    # 测试 deduce_solution
    print("\n1. 测试 deduce_solution 函数...")
    try:
        result = reasoning_agent.deduce_solution(MOCK_MODELING_RESULT)
        print("✅ deduce_solution 调用成功！")
        print(f"\n返回类型：{type(result)}")
        print(f"\n返回内容:\n{result}")
    except Exception as e:
        print(f"❌ deduce_solution 调用失败：{e}")
        return False

    # 测试 check_style
    print("\n2. 测试 check_style 函数...")
    try:
        # 如果 result 是字符串，尝试解析成 JSON
        import json
        if isinstance(result, str):
            solution_data = json.loads(result)
        else:
            solution_data = result

        is_valid = reasoning_agent.check_style(solution_data)
        print(f"✅ check_style 调用成功！结果：{is_valid}")
    except Exception as e:
        print(f"❌ check_style 调用失败：{e}")

    # 验证输出格式
    print("\n3. 验证输出格式...")
    try:
        if isinstance(result, str):
            solution_data = json.loads(result)
        else:
            solution_data = result

        # 检查必要字段
        assert "步骤" in solution_data, "缺少'步骤'字段"
        assert "最终答案" in solution_data, "缺少'最终答案'字段"
        assert isinstance(solution_data["步骤"], list), "'步骤'应该是数组"

        # 检查每个步骤的字段
        for i, step in enumerate(solution_data["步骤"]):
            assert "序号" in step, f"步骤{i+1}缺少'序号'字段"
            assert "描述" in step, f"步骤{i+1}缺少'描述'字段"
            assert "公式" in step, f"步骤{i+1}缺少'公式'字段"

        print("✅ 输出格式验证通过！")
        print(f"\n解题步骤数量：{len(solution_data['步骤'])}")
        print(f"最终答案：{solution_data['最终答案']}")

        # 打印步骤详情
        print("\n步骤详情:")
        for step in solution_data["步骤"]:
            print(f"  步骤{step['序号']}: {step['描述']}")
            if step['公式']:
                print(f"           公式：{step['公式']}")

    except Exception as e:
        print(f"❌ 输出格式验证失败：{e}")

    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)

    return True


if __name__ == "__main__":
    test_reasoning_agent()
