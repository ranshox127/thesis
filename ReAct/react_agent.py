from dotenv import load_dotenv
from lib.agent import Agent
from lib.tools import search
import logging
import json
import os

react_system_prompt = """You are an AI Agent based on the ReAct framework, and your task is to answer questions through the "observation -> reasoning -> action" cycle.

## **Reasoning and Action Rules**
1. **Reasoning (Reason)**
    - You should think about why you are performing an action before each action.
    - You should make a brief summary of the current situation and decide what to do next.
    - Your reasoning should be based on existing information and clearly explain why this is the best next step.
    - Your reasoning should be embedded in the `reason` parameter and executed within the `search` or `final_answer` tools.  

2. **Action (Act)**
    - You **cannot simply respond with text**, but must **use Function Calling to perform actions**.
    - You have the following tools available:
        - **`search(reason: str, query: str) -> List[Dict]`**
            - Use the `search` tool when you need additional information.
            - The `reason` parameter should explain why this search is necessary.
            - The `query` parameter is the keyword you want to search for.
        - **`final_answer(reason: str) -> str`**
            - Use the `final_answer` tool when you have collected enough information to answer the question.
            - The `reason` parameter should explain why you believe the current information is sufficient to answer the question.

## **Observation and Response Rules**
- You will receive the last search result each time, and you need to update your reasoning based on this information.
- You should analyze this information and decide:
    - Whether further search is needed?
    - Whether there is enough information to answer the question?

## **Example**

### **Using `search` to obtain additional information**
'''
Based on the question, I need to search "Who discovered dark energy?" to obtain background knowledge.
'''
_(At this point, you should use the `search` tool to perform the search)_

**Correct Usage**
```json
{
    "tool_call": {
        "name": "search",
        "arguments": {
            "reason": "Based on the question, I need to search 'Who discovered dark energy?' to obtain background knowledge.",
            "query": "Who discovered dark energy?"
        }
    }
}
```

### Using final_answer to provide the final answer
'''
Based on all the information collected, I can now answer the question.
'''
_(At this point, you should use the final_answer tool to respond)_

**Correct Usage**
```json
{
  "tool_call": {
    "name": "final_answer",
    "arguments": {
      "reason": "Based on all the information collected, I can now answer the question."
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
        self.history = [{"role": "developer", "content": system_prompt}]
        self.conversation_log = []
        self.total_tokens = []

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
        self.tools = [{
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
                    "required": ["reason", "query"]
                }
            }
        }, {
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
                    "required": ["reason"]
                }
            }
        }]

    def handle_tool_call(self, tool_call):
        """Executes the function requested by OpenAI's function calling system."""
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        logging.info(f"Tool called: {function_name} with args: {arguments}")

        if function_name == "search":
            query = arguments["query"]  # 只傳遞 query
            logging.info(f"Executing search for: {query}")
            return "search", search(query, max_results=15)

        elif function_name == "final_answer":
            logging.info("Generating final answer...")
            return "answer", self.final_answer()

        else:
            logging.warning(f"Unknown function requested: {function_name}")
            return "retry", None

    def final_answer(self):
        """
        Generates a final answer based on the entire conversation history.
        """
        logging.info("Generating final answer...")

        conversations = [
            {"role": "developer", "content": "You are an AI assistant designed to answer human questions. Your task is to strictly adhere to the conversation context and integrate information to respond to inquiries."}]
        conversations += self.history[1:]
        conversations += [{"role": "user",
                           "content": "Please respond to the question based on the conversation content above."}]

        response, usage = self.agent.generate_response(
            conversations, tools=None, tool_choice=None)

        # Append final answer to history
        self.conversation_log.append(
            {"role": "assistant", "content": response.content})

        logging.info(f"Final Answer: {response.content}")

        self.total_tokens.append({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        })

        self._save_summary()

        return response.content

    def _save_summary(self):
        """Saves the ReAct session to JSON with ordered retrieved data."""
        # If file is empty or invalid, initialize as empty list
        if not os.path.exists(self.summary_json) or os.stat(self.summary_json).st_size == 0:
            data = []
        else:
            try:
                with open(self.summary_json, "r", encoding="utf-8") as f:
                    data = json.load(f)  # Load existing data
            except json.JSONDecodeError:
                logging.warning("JSON file is corrupted. Resetting to empty.")
                data = []  # Reset JSON if it's corrupted

        session_summary = {
            "question": self.history[1]["content"].replace("Question: ", ""),
            "conversations": self.conversation_log,
            "token_usage": self.total_tokens
        }

        data.append(session_summary)

        with open(self.summary_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False,
                      indent=4)  # Save updated data

    def run(self, question):
        """
        Executes the ReAct loop:
        - Generates Thought, Action
        - Executes Action, gets Observation
        - Retries if needed
        """
        logging.info(f"Starting new session with question: {question}")

        user_question = {"role": "user", "content": f"Question: {question}"}

        self.history.append(user_question)
        self.conversation_log.append(user_question)

        conversations = self.history.copy()
        turn = 1

        while turn <= self.max_turns:
            logging.info(f"Turn {turn}: Generating Thought & Action...")

            # 1. Generate Thought + Action
            response, usage = self.agent.generate_response(
                conversations, tools=self.tools, tool_choice="required")
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

                    # 4. Update Conversations
                    if state == "search":
                        assistant_response = {
                            "role": "assistant",
                            # 直接當作回應
                            "content": f"Using tool: {tool_call.function.name} with arguments {tool_call.function.arguments}",
                        }
                        search_observation = {
                            "role": "user",
                            # 確保是字串
                            "content": f"<retrieved data>\n{str(feedback)}\n</retrieved data>"
                        }

                        # Append both LLM response and retrieved data in order
                        self.history.append(assistant_response)
                        self.conversation_log.append(assistant_response)
                        self.conversation_log.append(search_observation)

                        conversations = self.history.copy() + \
                            [search_observation]

                        logging.info(
                            f"Search Results for Turn {turn}: {feedback}\n")

                    elif state == "answer":
                        logging.info("Final Answer Reached.")
                        return feedback

            turn += 1

        logging.warning("Max turns reached. No definitive answer found.")
        return self.final_answer()
