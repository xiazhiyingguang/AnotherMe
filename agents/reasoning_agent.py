"""
解题 Agent - 用来生成完整的解题步骤
"""

from .base_agent import BaseAgent
from typing import Any, Dict, Optional, Tuple
from config import DEFAULT_LLM_CONFIG
from langchain_core.messages import HumanMessage
import json
import re
import base64
import os


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

    def deduce_solution(self, input_data: Any) -> Dict[str, Any]:
        """
        处理输入并返回完整的解题步骤。

        Args:
            input_data: 题目图片路径，或 VisionAgent 输出字典

        Returns:
            包含完整解题步骤的字典
        """
        image_path, ocr_result, modeling_result, geometry_ir = self._resolve_reasoning_inputs(input_data)

        content = []
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    },
                }
            )

        # 将 Vision 输出作为强上下文，避免仅靠图像推理丢失结构信息。
        context_text = (
            "请结合以下信息进行解题：\n"
            f"OCR: {ocr_result or '未提供'}\n"
            f"建模: {json.dumps(modeling_result or {}, ensure_ascii=False)}\n"
            f"几何IR: {json.dumps(geometry_ir or {}, ensure_ascii=False)}\n"
        )

        message = HumanMessage(
            content=content + [
                {
                    "type": "text",
                    "text": context_text + """你是数学解题专家，请根据题图与结构化信息生成完整的解题步骤。
                    要求：
                    1. 详细分析题目中的已知条件和求解目标
                    2. 列出解题思路和步骤，使用清晰的数学语言和符号，使用 LaTeX 表示
                    3. 每一步都要有清晰的解释和逻辑推理
                    4. 最后给出完整的解答过程和结果
                    5. 不要添加与解题无关的内容或解释
                    6. 输出格式要清晰，便于理解和后续生成视频分镜

                    请严格按照以下 JSON 格式输出：
                    {
                        "步骤": [
                            {
                                "序号": 1,
                                "描述": "步骤描述文字",
                                "公式": "$LaTeX 公式$"
                            }
                        ],
                        "最终答案": "最终结果"
                    }
                    注意：
                    - "步骤"必须是一个数组，包含多个步骤对象
                    - 每个步骤对象必须包含"序号"、"描述"、"公式"三个字段
                    - 公式使用 LaTeX 格式，用 $ 包裹
                    """
                }
            ]
        )

        response = self.llm.invoke([message])
        return self._parse_json_response(response.content)

    def _resolve_reasoning_inputs(self, input_data: Any) -> Tuple[Optional[str], str, Dict[str, Any], Dict[str, Any]]:
        """统一解析 Reasoning 输入，兼容路径与 Vision 字典。"""
        if isinstance(input_data, str):
            return input_data, "", {}, {}

        if isinstance(input_data, dict):
            image_meta = input_data.get("image_meta", {}) if isinstance(input_data.get("image_meta", {}), dict) else {}
            geometry_ir = input_data.get("geometry_ir", {}) if isinstance(input_data.get("geometry_ir", {}), dict) else {}
            source = geometry_ir.get("source", {}) if isinstance(geometry_ir.get("source", {}), dict) else {}

            image_path = (
                image_meta.get("diagram_image_path")
                or image_meta.get("original_image_path")
                or source.get("diagram_image_path")
                or source.get("original_image_path")
            )

            modeling_result = input_data.get("modeling_result", {})
            if not isinstance(modeling_result, dict):
                modeling_result = {"modeling_result": str(modeling_result)}

            return (
                str(image_path) if image_path else None,
                str(input_data.get("ocr_result", "")),
                modeling_result,
                geometry_ir,
            )

        raise TypeError(f"ReasoningAgent 不支持的输入类型: {type(input_data)}")

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
