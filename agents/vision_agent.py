"""
视觉 Agent - 负责 OCR 识别和理解题目图片
"""

from .base_agent import BaseAgent
from typing import Any, Dict, Optional
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import re
import base64 

class VisionAgent(BaseAgent):
    """视觉 Agent - 负责 OCR 识别和理解题目图片"""

    def __init__(self, config: Dict[str, Any], llm: Optional[Any] = None):
        """
        初始化 VisionAgent

        Args:
            config: VisionAgent 的配置字典
            llm: 可选的 LLM 实例，如果不传则内部创建
        """
        if not llm:
            from config import VISION_MODEL_CONFIG
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=VISION_MODEL_CONFIG["api_key"],
                base_url=VISION_MODEL_CONFIG["base_url"],
                model=VISION_MODEL_CONFIG["model"],
                temperature=config.get("temperature", 0.05),
                max_tokens=config.get("max_tokens", 2048),
            )

        super().__init__(config, llm)


    def process(self, input_data: str) -> Dict[str, Any]:
        """
        处理输入的题目图片 URL，返回 OCR 结果和理解结果

        Args:
            input_data: 题目图片的 URL 或本地文件路径

        Returns:
            包含 OCR 结果和理解结果的字典
        """
        # 1. 使用 OCR 模型识别图片中的文本
        ocr_result = self.ocr_recognize(input_data)

        # 2. 使用 LLM 理解图片等信息，进行数学建模，生成对题目的理解
        modeling_result = self.model_problem(ocr_result)

        return {
            "ocr_result": ocr_result,
            "modeling_result": modeling_result
        }

    def ocr_recognize(self, image_path: str) -> str:
        """
        使用 OCR 模型识别图片中的文本

        Args:
            image_path: 题目图片的 URL 或本地文件路径

        Returns:
            图片中识别出的文本
        """
        # 判断是本地文件还是 URL
        if image_path.startswith(("http://", "https://")):
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_path, "detail": "high"}
            }
        else:
            # 本地文件转 base64
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            }

        message = HumanMessage(
            content=[
                image_content,
                {
                    "type": "text",
                    "text": """请识别这张图片中题目的所有文字内容
                    要求：
                    1. 准确提取所有文字，包括标点符号
                    2. 数学公式用 LaTeX 格式表示
                    3. 按原格式排版（换行、缩进等）
                    直接输出识别的文字，不要添加其他解释。"""
                }
            ]
        )
        response = self.llm.invoke([message])

        return response.content




    def model_problem(self, ocr_result: str) -> Dict[str, Any]:
        """
        使用 LLM 理解图片等信息，进行数学建模，生成对题目的理解

        Args:
            ocr_result: OCR 识别结果

        Returns:
            对题目数学建模方面的理解结果
        """
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                你是一个数学建模专家，负责根据题目图片进行建模分析。
                请根据以下 OCR 识别结果，提取题目的关键信息，并进行数学建模。
                要求：
                1. 识别题目的类型（选择题/填空题/解答题）
                2. 提取题目的已知条件、求解目标和约束条件
                3. 将题目中的实体和关系抽象成数学符号和公式,使用LaTeX表示
                4. 输出清晰的数学模型，包括变量定义、方程组等,使用LaTeX表示
                5. 不要添加任何与建模无关的解释或内容
                6. 所有字段都用字符串表示，不要用数组/列表
                请用 json 格式输出
                """
            ),
            (
                "human",
                """
                以下是 OCR 识别的题目文本，请根据这些信息进行数学建模：{ocr_result}
                请分析并建立数学模型。
                """
            )
        ])

        chain = prompt | self.llm
        response = chain.invoke({"ocr_result": ocr_result})

        return self.parse_json_response(response.content)


    def parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON 内容"""
        # 提取 JSON 块
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return {"error": "未找到 JSON 内容", "raw_content": content[:500]}

        json_str = json_match.group()

        # 替换中文引号
        json_str = json_str.replace('"', '"').replace('"', '"').replace('"', '"').replace('"', '"')

        # 处理 LaTeX 转义：\command 形式需要变成 \\command
        # 但要注意不要重复转义已经是 \\ 的
        def escape_latex(match):
            cmd = match.group(0)
            if cmd.startswith('\\\\'):
                return cmd  # 已经是双斜杠
            return '\\' + cmd  # 单斜杠变双斜杠

        # 匹配 \ 开头的 LaTeX 命令
        json_str = re.sub(r'\\[a-zA-Z]+', escape_latex, json_str)

        try:
            result = json.loads(json_str)
            # 后处理：把公式中的 \\\\ 清理成 \\（方便后续使用）
            self._clean_latex_backslashes(result)
            return result
        except json.JSONDecodeError as e:
            # 如果还是失败，尝试用 ast.literal_eval 处理
            import ast
            try:
                # 把 JSON 当 Python 字典处理（支持单引号等）
                return ast.literal_eval(json_str)
            except:
                pass

            return {
                "error": f"无法解析 JSON: {e}",
                "raw_content": json_str[:500]
            }

    def _clean_latex_backslashes(self, data: Any) -> None:
        """递归清理嵌套结构中的 LaTeX 反斜杠，并把列表转成字符串"""
        if isinstance(data, dict):
            for key, value in list(data.items()):
                if isinstance(value, list):
                    # 把列表拼接成字符串
                    data[key] = ', '.join(str(v) for v in value)
                elif isinstance(value, str):
                    # 把 \\\\ 变成 \
                    data[key] = value.replace('\\\\', '\\')
                else:
                    self._clean_latex_backslashes(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    data[i] = item.replace('\\\\', '\\')
                else:
                    self._clean_latex_backslashes(item)


    def _flatten_list_to_string(self, data: Any) -> None:
        """递归把嵌套结构中的列表转成字符串"""
        if isinstance(data, dict):
            for key, value in list(data.items()):
                if isinstance(value, list):
                    data[key] = ', '.join(str(v) for v in value)
                elif isinstance(value, (dict, list)):
                    self._flatten_list_to_string(value)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._flatten_list_to_string(item)


