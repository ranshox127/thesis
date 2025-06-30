import os
from openai import OpenAI
from groq import Groq
import logging


class Agent:
    def __init__(self, llm):
        """
        通用 Agent 基底類別
        :param llm: gpt-4o, gpt-4o-mini, llama 3.3 70B
        """
        self.llm = llm

        # 初始化 LLM API 客戶端
        if llm in ["gpt-4o", "gpt-4o-mini"]:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif llm == "llama 3.3 70B":
            self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        else:
            raise ValueError(f"Unsupported LLM: {llm}")

    def generate_response(self, conversations, tools=None, tool_choice="required"):
        """
        與 LLM 交互，生成回應
        conversations: List[Dict]，對話記錄
        tools: 可用的工具函式列表
        tool_choice: 是否強制使用工具 (預設為 "required")

        回傳:
        - message: LLM 的回應 (可能是工具呼叫，或最終回覆)
        - usage: 該次請求的 token 統計數據
        """
        
        # **如果 tools 被設定，則強制使用 tool_choice**
        kwargs = {}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice  # 預設為 "required"，確保每次回應都使用工具
    
        if self.llm in ["gpt-4o", "gpt-4o-mini"]:
            completion = self.client.chat.completions.create(
                model=self.llm,
                messages=conversations,
                parallel_tool_calls=False,
                **kwargs
            )
            return completion.choices[0].message, completion.usage
        elif self.llm == "llama 3.3 70B":
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=conversations,
                parallel_tool_calls=False,
                **kwargs
            )
            return completion.choices[0].message, completion.usage
        else:
            raise NotImplementedError(f"LLM {self.llm} is not yet integrated.")
