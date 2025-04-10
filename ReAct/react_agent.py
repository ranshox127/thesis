import json
import logging
import os

from dotenv import load_dotenv
from lib.agent import Agent
from lib.tools import search

REACT_SYSTEM_PROMPT = """You are an AI Agent based on the ReAct framework, and your task is to answer questions through the "observation -> reasoning -> action" cycle.

## **Role and Mission**
You are an AI assistant designed to answer human questions. Your task is to strictly adhere to the conversation context and integrate information to respond to inquiries. You must use structured reasoning to determine whether additional information is required before answering.

## **Reasoning and Action Rules**
1. **Reasoning (Reason)**
    - Before taking any action, you must **explain your reasoning**.
    - Your reasoning must be concise but informative, summarizing the current situation and justifying the next step.
    - If more information is needed, explain why.
    - If enough information has been gathered, justify why a final answer can be given.
    - **Your reasoning must be included in the `reason` parameter** of the tool call.

2. **Action (Act)**
    - You have access to the following tools:
        - **`search(reason: str, query: str) -> List[Dict]`**
            - Use this tool when additional information is needed.
            - `reason`: Explain why this search is necessary.
            - `query`: The keyword or phrase to search.
        - **`final_answer(reason: str) -> str`**
            - Use this tool when you have gathered sufficient information to answer the question.
            - `reason`: Explain why the current information is sufficient to generate an answer.

## **Observation and Response Rules**
- You will receive the most recent search result after each action.
- You should carefully analyze the new information and decide:
    - Whether another search is required?
    - Whether you have gathered enough information to generate the final answer?

## **Example Usage**

### **Using `search` to obtain additional information**
'''
Based on the question, I need to search for "Who discovered dark energy?" to obtain relevant information.
'''
_(At this point, you should use the `search` tool to perform the search.)_

**Correct Tool Call**
```json
{
  "tool_call": {
    "name": "search",
    "arguments": {
      "reason": "To answer the question about dark energy, I need to find out who discovered it.",
      "query": "Who discovered dark energy?"
    }
  }
}

```

"""

load_dotenv()


class ReAct_agent:
    def __init__(self, llm, system_prompt, max_turns=9, debug_log="react_debug.log", summary_json="react_summary.json"):
        self.agent = Agent(llm=llm)
        self.max_turns = max_turns
        self.conversation = [{"role": "developer", "content": system_prompt}]
        self.conversation_log = []  # 用於詳細記錄每一條訊息，不做傳入模型用
        self.total_tokens = []  # 用於詳細記錄每個resopnse的tokens數量

        # Setup detailed debug logging
        logging.basicConfig(filename=debug_log, level=logging.DEBUG,
                            format="%(asctime)s [%(levelname)s] %(message)s", encoding="utf-8")
        logging.info("\n=== New ReAct Execution Started ===\n")

        # Summary log file
        self.summary_json = summary_json

        # Initialize JSON file if it doesn't exist
        if not os.path.exists(self.summary_json):
            with open(self.summary_json, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=4)

        # Define Function Calling Tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Retrieve relevant web search results for a given query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason why this search is needed."
                            },
                            "query": {
                                "type": "string",
                                "description": "Search query string."
                            }
                        },
                        "required": ["reason", "query"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": "Generate a final answer based on the conversation history.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason why the final answer can now be generated."
                            }
                        },
                        "required": ["reason"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]

    def handle_tool_call(self, tool_call):
        """Executes the function requested by OpenAI's function calling system."""
        function_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logging.warning(f"JSON Decode Error in tool_call arguments: {e}")
            return "retry", None

        logging.info(f"Tool called: {function_name} with args: {arguments}")

        if function_name == "search":
            query = arguments.get("query")

            if not query:
                logging.warning(
                    "Missing 'query' parameter in search function call.")
                return "retry", None

            logging.info(f"Executing search for: {query}")
            return "search", search(query, max_results=10)

        elif function_name == "final_answer":
            logging.info("Generating final answer...")
            return "answer", arguments.get("reason")

        else:
            logging.warning(f"Unknown function requested: {function_name}")
            return "retry", None

    def final_answer(self):
        """
        Generates a final answer based on the entire conversation history.
        """
        logging.info("Generating final answer...")

        response, usage = self.agent.generate_response(
            self.conversation, tools=self.tools, tool_choice={"type": "function", "function": {"name": "final_answer"}})

        assistant_response = {"agent": "planner", "role": "assistant",
                              "tool_calls": response.tool_calls}

        self.conversation.append(assistant_response)
        self.conversation_log.append(assistant_response)

        logging.info(f"Final Answer: {response.content}")

        self.total_tokens.append({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        })

        tool_call = response.tool_calls[0]
        state, feedback = self.handle_tool_call(tool_call)

        tool_response = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": feedback,
        }

        self.conversation.append(tool_response)
        self.conversation_log.append(tool_response)

    def _save_summary(self):
        """Saves the ReAct session to JSON with ordered retrieved data."""

        if not os.path.exists(self.summary_json) or os.stat(self.summary_json).st_size == 0:
            data = []
        else:
            try:
                with open(self.summary_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                logging.warning("JSON file is corrupted. Resetting to empty.")
                data = []

        # 保證所有 log 可序列化
        def make_json_safe(entry):
            if isinstance(entry, dict):
                safe_entry = entry.copy()
                if "tool_calls" in safe_entry:
                    safe_entry["tool_calls"] = [
                        getattr(tc, "model_dump", lambda: str(tc))() for tc in safe_entry["tool_calls"]]
                return safe_entry
            return str(entry)

        serializable_log = [make_json_safe(msg)
                            for msg in self.conversation_log]

        session_summary = {
            "question": self.conversation_log[1]["content"].replace("Question: ", ""),
            "conversations": serializable_log,
            "token_usage": self.total_tokens
        }

        data.append(session_summary)

        with open(self.summary_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def merge_assistant_message(self, conversation, new_message):
        """
        合併 Assistant 的回應，確保不會產生過多獨立的 Assistant 訊息。
        """
        if conversation and conversation[-1]["role"] == "assistant":
            # 直接合併到最後一個 assistant 回應中
            conversation[-1]["content"] += "\n" + new_message["content"]
        else:
            conversation.append(new_message)

    def run(self, question):
        """
        Executes the ReAct loop:
        - Generates Thought, Action
        - Executes Action, gets Observation
        - Retries if needed
        """
        logging.info(f"Starting new session with question: {question}")

        user_question = {"role": "user", "content": f"Question: {question}"}

        self.conversation.append(user_question)

        self.conversation_log = self.conversation.copy()

        turn = 1

        while turn <= self.max_turns:
            logging.info(f"Turn {turn}: Generating Thought & Action...")

            # 1. Generate Thought + Action
            response, usage = self.agent.generate_response(
                self.conversation, tools=self.tools, tool_choice="required")
            logging.info(f"LLM Response:\n{response}")

            self.total_tokens.append({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            })

            # 2. Check if the LLM requested a function call
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    state, feedback = self.handle_tool_call(tool_call)

                    # 3. Retry if the tool call was invalid
                    if state == "retry":
                        logging.warning(f"Retrying Turn {turn}...")
                        continue

                    assistant_response = {
                        "role": "assistant",
                        "content": f"Using tool: {tool_call.function.name} with arguments {tool_call.function.arguments}",
                    }

                    self.conversation_log.append(assistant_response)

                    # 4. Update Conversations
                    if state == "search":
                        search_observation = {
                            "role": "user",
                            "content": f"<retrieved data>\n{str(feedback)}\n</retrieved data>"
                        }

                        self.conversation_log.append(search_observation)

                        if self.conversation and self.conversation[-1]["role"] == "user":
                            self.conversation.pop()
                        self.merge_assistant_message(
                            self.conversation, assistant_response)
                        self.conversation.append(search_observation)

                        logging.info(
                            f"Search Results for Turn {turn}: {feedback}\n")

                    elif state == "answer":
                        self.conversation.append(assistant_response)

                        final_answer_request = {
                            "role": "user",
                            "content": (
                                "Please organize the information you have gathered and write a complete and comprehensive answer to the original question.")}

                        self.conversation.append(final_answer_request)
                        self.conversation_log.append(final_answer_request)

                        response, usage = self.agent.generate_response(
                            self.conversation, tools=self.tools, tool_choice="none")

                        logging.info(f"LLM Response:\n{response.content}")

                        self.total_tokens.append({"prompt_tokens": usage.prompt_tokens,
                                                  "completion_tokens": usage.completion_tokens,
                                                  "total_tokens": usage.total_tokens})

                        self.conversation_log.append(
                            {"role": "assistant", "content": response.content})

                        self._save_summary()
                        logging.info("Final Answer Reached.")
                        return response.content

            turn += 1

        logging.warning("Max turns reached. No definitive answer found.")
        self.final_answer()

        final_answer_request = {
            "role": "user",
            "content": (
                "Please organize the information you have gathered and write a complete and comprehensive answer to the original question.")}

        self.conversation.append(final_answer_request)
        self.conversation_log.append(final_answer_request)

        response, usage = self.agent.generate_response(
            conversations=self.conversation, tools=self.tools, tool_choice="none")

        self.total_tokens.append({"prompt_tokens": usage.prompt_tokens,
                                  "completion_tokens": usage.completion_tokens,
                                  "total_tokens": usage.total_tokens})

        self.conversation_log.append(
            {"role": "assistant", "content": response.content})

        self._save_summary()
        return response.content
