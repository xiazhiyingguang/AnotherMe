"""
解题 Agent - 用来生成完整的解题步骤
"""

from .base_agent import BaseAgent
from typing import Any, Dict, Optional
from config import DEFAULT_LLM_CONFIG
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import re


class ReasoningAgent(BaseAgent):
    """解题 Agent - 用来生成完整的解题步骤"""

    def __init__(self, config: Dict[str, Any], llm=None):
        """
        初始化 ReasoningAgent
        Args:
            config: ReasoningAgent 的配置字典
            llm: 可选的 LLM 实例，如果不传则内部创建
        """
        if not llm:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=DEFAULT_LLM_CONFIG["api_key"],
                base_url=DEFAULT_LLM_CONFIG["base_url"],
                model=DEFAULT_LLM_CONFIG["model"],
                temperature=config.get("temperature", 0.1),
                max_tokens=config.get("max_tokens", 4096),
            )

        super().__init__(config, llm)

    def process(self, input_data: Any) -> Any:
        return self.deduce_solution(input_data)

    def deduce_solution(self, modeling_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入的题目描述，返回完整的解题步骤

        Args:
            modeling_result: 数学建模结果

        Returns:
            包含完整解题步骤的字典
        """
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个数学解题专家，负责根据题目描述生成完整的解题步骤。
                要求：
                1. 详细分析题目中的已知条件和求解目标
                2. 列出解题思路和步骤，使用清晰的数学语言和符号，使用 LaTeX 表示
                3. 每一步都要有清晰的解释和逻辑推理
                4. 最后给出完整的解答过程和结果
                5. 不要添加与解题无关的内容或解释
                6. 输出格式要清晰，便于理解和后续生成视频分镜
                请严格按照以下 JSON 格式输出：
                {{
                    "步骤": [
                        {{
                            "序号": 1,
                            "描述": "步骤描述文字",
                            "公式": "$LaTeX 公式$"
                        }},
                        ...
                    ],
                    "最终答案": "最终结果"
                }}
                注意：
                - "步骤"必须是一个数组，包含多个步骤对象
                - 每个步骤对象必须包含"序号"、"描述"、"公式"三个字段
                - 公式使用 LaTeX 格式，用 $ 包裹
                """
            ),
            (
                "human",
                """
                以下是通过分析得到的数学建模结果，请根据这些信息生成完整的解题步骤：
                {modeling_result}
                """
            )
        ])

        chain = prompt | self.llm
        response = chain.invoke({"modeling_result": modeling_result})

        return self._parse_json_response(response.content)

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON 内容"""
        # 提取 JSON 块
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return {"error": "未找到 JSON 内容", "raw_content": content[:500]}

        json_str = json_match.group()

        # 替换中文引号
        json_str = json_str.replace('"', '"').replace('"', '"').replace('"', '"').replace('"', '"')

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # 如果还是失败，尝试用 ast.literal_eval 处理
            import ast
            try:
                return ast.literal_eval(json_str)
            except:
                pass

            return {
                "error": f"无法解析 JSON: {e}",
                "raw_content": json_str[:500]
            }

    def check_style(self, solution_steps: Dict[str, Any]) -> bool:
        """
        检查解题步骤的风格和格式是否符合要求

        Args:
            solution_steps: 包含解题步骤的字典

        Returns:
            是否符合要求的布尔值
        """
        # 检查必要字段
        if "步骤" not in solution_steps:
            return False
        if "最终答案" not in solution_steps:
            return False

        # 检查步骤格式
        steps = solution_steps.get("步骤", [])
        if not isinstance(steps, list):
            return False

        for step in steps:
            if "序号" not in step or "描述" not in step or "公式" not in step:
                return False

        return True
