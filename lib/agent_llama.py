import os
import instructor
from pydantic import BaseModel, Field
from groq import Groq
import logging


class ToolCall(BaseModel):
    tool_name: str = Field(description="The name of the tool to call")
    tool_parameters: str = Field(description="JSON string of tool parameters")


class Search(BaseModel):
    tool_name: str = Field(description="The name of the tool to call")
    reason: str = Field(description="The reason why this search is needed")
    query: str = Field(description="The search query string")


class Summary(BaseModel):
    tool_name: str = Field(description="The name of the tool to call")
    summary: str = Field(description="The summary of retrieved data")


class CriticalEvaluation(BaseModel):
    critical_evaluation: str = Field(
        description="Your reflection after gathering information")


class FinalAnswer(BaseModel):
    final_answer: str = Field(
        description="The final answer based on your cognition")


class Agent_llama:
    def __init__(self, llm="llama 3.3 70B"):
        """
        通用 Agent 基底類別
        :param llm: llama 3.3 70B
        """
        self.llm = llm

        # 初始化 LLM API 客戶端
        if llm == "llama 3.3 70B":
            self.client = instructor.from_groq(
                Groq(api_key=os.environ.get("GROQ_API_KEY")), mode=instructor.Mode.JSON)
        else:
            raise ValueError(f"Unsupported LLM: {llm}")

    def generate_response(self, conversations, action="TC"):
        """
        與 LLM 交互，生成回應
        conversations: List[Dict]，對話記錄
        action: LLaMA 3.3 70B 的回覆類型
        - TC: 工具呼叫
        - CE: 批判性評估
        - FA: 最終回答

        - Summary: 總結
        - Search: 搜尋

        回傳:
        - completion: LLM 的回應 (可能是工具呼叫，或最終回覆)
        """

        response_model: BaseModel

        if action == "TC":
            response_model = ToolCall
        elif action == "CE":
            response_model = CriticalEvaluation
        elif action == "FA":
            response_model = FinalAnswer
        elif action == "Search":
            response_model = Search
        elif action == "Summary":
            response_model = Summary

        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_model=response_model,
            messages=conversations
        )
        return completion
