"""
Agent 抽象基类 - 定义所有 Agent 的统一接口
"""
from abc import ABC, abstractmethod # 导入抽象基类相关模块
from typing import Any, Dict, Optional
from langchain_openai import ChatOpenAI


class BaseAgent(ABC):
    """所有 Agent 的基类"""

    def __init__(self, config: Dict[str, Any], llm: Optional[ChatOpenAI] = None):
        """
        初始化 Agent

        Args:
            config: Agent 配置字典
            llm: 可选的 LLM 实例，如果不传则内部创建
        """
        self.config = config
        self.name = config.get("name", "BaseAgent") # Agent 的名称，默认为 "BaseAgent"
        self.description = config.get("description", "")

        # 初始化 LLM
        if llm:
            self.llm = llm
        else:
            from config import DEFAULT_LLM_CONFIG
            self.llm = ChatOpenAI(
                api_key=DEFAULT_LLM_CONFIG["api_key"],
                base_url=DEFAULT_LLM_CONFIG["base_url"],
                model=DEFAULT_LLM_CONFIG["model"],
                temperature=config.get("temperature", 0.1),
                max_tokens=config.get("max_tokens", 4096),
            )

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        处理输入数据，返回输出结果

        Args:
            input_data: 输入数据，类型由子类定义

        Returns:
            输出结果，类型由子类定义
        """
        pass

    # 可选的字符串表示方法，方便调试和日志记录
    def __repr__(self) -> str:
        return f"{self.name}({self.description})"
