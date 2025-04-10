import json
import logging
import os

from dotenv import load_dotenv
from lib.agent import Agent
from lib.tools import search

PLANNER_SYSTEM_PROMPT = """You are an AI Planner Agent designed to handle complex questions using a **decomposition-first strategy**.

## Your Goal:
Make decisions on how to break down complex questions or provide final answers when appropriate.

---

## Available Actions

1. **question_decompose**  
   - Use this when a question is **too broad, multi-faceted, abstract, or contains multiple sub-goals**.
   - You may decompose **one or more existing questions** — including the original question or any previous sub-questions.
   - **Important:** You can **only decompose questions that already exist** in the conversation context. Do not introduce new parent questions that have not appeared before.
   - Each decomposition must map a `parent_q` to a list of `sub_questions`.
   - You must provide a global `reason` explaining **why** decomposition is needed.
   - Format:
     ```json
     {
       "reason": "These questions require finer-grained analysis.",
       "mapping": [
         {
           "parent_q": "How does climate change affect agriculture?",
           "sub_questions": [
             "How does temperature rise affect crop yield?",
             "How does drought impact livestock?"
           ]
         },
         {
           "parent_q": "What are the social impacts of climate change?",
           "sub_questions": [
             "How does climate change affect migration patterns?",
             "What mental health issues are linked to climate change?"
           ]
         }
       ]
     }
     ```

2. **final_answer**
   - Use this only when you believe the question can be directly answered, either based on prior knowledge or based on the `<sub-answers>` returned by previous decompositions.
   - Provide a `reason` explaining **why** a final answer can now be given.
   - Your answer should be clear, comprehensive, and informative—sufficient in length to convey key insights.

---

## Observation Rules

- You may receive a block like:
  ```
  <sub-answers>
  [ ... structured list of sub-questions and their summarized answers ... ]
  </sub-answers>
  ```
  This means earlier decomposed questions have been resolved. You should:
  - Consider whether these provide enough context to synthesize a final answer.
  - Or decide whether **further decomposition** is needed for any sub-question.

---

## Reasoning Format

For every action:
1. First, **explain your reasoning** clearly in natural language.
2. Then, **invoke one of the tools** using structured output (function call).

---

## Examples

### Decomposing Questions

'''
The original question is too broad, spanning environmental, economic, and social dimensions. I will decompose it into manageable parts for better analysis.

```json
{
  "reason": "The question requires analysis across multiple domains.",
  "mapping": [
    {
      "parent_q": "How can we reduce global carbon emissions?",
      "sub_questions": [
        "What are the biggest sources of carbon emissions?",
        "What policies have proven effective in reducing emissions?",
        "What role does individual behavior play in emission reduction?"
      ]
    }
  ]
}
```
'''

### Providing a Final Answer

'''
I have received sufficient sub-answers, and they together form a complete picture. I can now provide a final answer to the original question.

```json
{
  "reason": "The provided sub-answers cover all aspects of the original question.",
}
```
'''"""

load_dotenv()


class MindSearch:
    def __init__(self, llm, system_prompt, max_turns=9, debug_log="mindsearch_debug.log", summary_json="mindsearch_summary.json"):
        self.planner = Agent(llm=llm)
        self.searcher = Agent(llm=llm)
        self.max_turns = max_turns
        self.planner_conversation = [
            {"role": "developer", "content": system_prompt}]
        self.conversation_log = []  # 用於詳細記錄每一條訊息，不做傳入模型用
        self.total_tokens = []  # 用於詳細記錄每個resopnse的tokens數量
        self.questions = []

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

        self.tools = [{
            "type": "function",
            "function": {
                "name": "question_decompose",
                "description": "Decompose one or more complex questions into sub-questions with reasoning.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why the decomposition is needed"
                        },
                        "mapping": {
                            "type": "array",
                            "description": "List of parent questions and their sub-questions",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "parent_q": {
                                        "type": "string",
                                        "description": "The original question being broken down"
                                    },
                                    "sub_questions": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of sub-questions derived from the parent question"
                                    }
                                },
                                "required": ["parent_q", "sub_questions"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["reason", "mapping"],
                    "additionalProperties": False
                }
            }
        }, {
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": (
                    "Based on the current information and your confidence, explain why a final answer can be generated."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Explain why you now have enough information to provide a final answer."
                        }
                    },
                    "required": ["reason"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]

    def _normalize(self, q: str) -> str:
        return q.lower().strip().rstrip("?。！？")

    def handle_tool_call(self, tool_call):
        """Executes the function requested by OpenAI's function calling system."""
        function_name = tool_call.function.name

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logging.warning(f"JSON Decode Error in tool_call arguments: {e}")
            return "retry", None

        logging.info(f"Tool called: {function_name} with args: {arguments}")

        if function_name == "question_decompose":
            mapping = arguments.get("mapping")

            valid_new_questions = []
            sub_answers = []

            for item in mapping:
                parent_q = item.get("parent_q")
                sub_qs = item.get("sub_questions", [])

                if self._normalize(parent_q) not in [self._normalize(q) for q in self.questions]:
                    logging.warning(f"Parent question not found: {parent_q}")
                    continue

                sub_answer = self.get_sub_answers(parent_q, sub_qs)
                sub_answers.append(sub_answer)

                valid_new_questions.extend(sub_qs)

            if len(sub_answers) == 0:  # 只要有一個 parent_q 存在，sub_answers 就會有東西。反之，就會是空的。
                logging.warning("No valid existed questions found.")
                return "retry", None

            self.questions.extend(valid_new_questions)

            return "decompose", sub_answers

        elif function_name == "search":
            query = arguments.get("query")

            logging.info(f"Executing search for: {query}")
            return "search", search(query, max_results=5)

        elif function_name == "summary":
            summary = arguments.get("summary")

            logging.info(f"summary: {summary}")
            return "summary", summary

        elif function_name == "final_answer":
            logging.info("Generating final answer...")
            return "answer", arguments.get("reason")

        else:
            logging.warning(f"Unknown function requested: {function_name}")
            return "retry", None

    def get_sub_answers(self, question, sub_questions):
        sub_answers = []
        searcher_system_prompt = """You are an AI assistant designed to answer sub-questions.
        You will be presented with original questions and sub-questions that you need to answer using a search engine.
        Your task is to:
        1. come up with the most appropriate query based on the original question and the sub-question.
        2. summarize the search results to answer the sub-question based on the original question and the sub-question."""

        for sub_question in sub_questions:

            searcher_conversation = [{"role": "developer", "content": searcher_system_prompt},
                                     {"role": "user", "content": f"The parent question: {question}, the sub-question: {sub_question}"}]

            tool = [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Retrieve relevant web search results for a given query.",
                        "strict": True,
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
                            "required": [
                                "reason",
                                "query"
                            ],
                            "additionalProperties": False
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "summary",
                        "description": "summary retrieved data to answer the sub-question.",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "summary": {
                                    "type": "string",
                                    "description": "summary of the retrieved data."
                                }
                            },
                            "required": ["summary"],
                            "additionalProperties": False
                        }
                    }
                }
            ]
            # search
            response, usage = self.searcher.generate_response(
                conversations=searcher_conversation, tools=tool, tool_choice={"type": "function", "function": {"name": "search"}})

            self.total_tokens.append({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            })

            search_call = response.tool_calls[0]

            search_result = self.handle_tool_call(search_call)[1]

            assistant_response = {"role": "assistant",
                                  "tool_calls": response.tool_calls}

            tool_response = {
                "role": "tool",
                "tool_call_id": search_call.id,
                "content": str(search_result)
            }

            searcher_conversation.append(assistant_response)
            searcher_conversation.append(tool_response)

            # summary

            response, usage = self.searcher.generate_response(
                conversations=searcher_conversation, tools=tool, tool_choice={"type": "function", "function": {"name": "summary"}})

            self.total_tokens.append({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            })

            summary_call = response.tool_calls[0]
            summary_result = self.handle_tool_call(summary_call)[1]

            self.conversation_log.extend([
                {"agent": "searcher", **msg} for msg in searcher_conversation
            ])

            assistant_response = {"agent": "searcher", "role": "assistant",
                                  "tool_calls": response.tool_calls}

            tool_response = {
                "agent": "searcher",
                "role": "tool",
                "tool_call_id": summary_call.id,
                "content": str(summary_result)
            }

            self.conversation_log.append(assistant_response)
            self.conversation_log.append(tool_response)

            sub_answers.append(
                {"sub_q": sub_question, "answer": summary_result})

        return {
            "parent_q": question,
            "sub_answers": sub_answers
        }

    def final_answer(self):
        logging.info("Generating final answer...")

        response, usage = self.planner.generate_response(
            self.planner_conversation, tools=self.tools, tool_choice={"type": "function", "function": {"name": "final_answer"}})

        self.total_tokens.append({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        })

        assistant_response = {"role": "assistant",
                              "tool_calls": response.tool_calls}

        self.planner_conversation.append(assistant_response)
        self.conversation_log.append(assistant_response)

        tool_call = response.tool_calls[0]
        state, feedback = self.handle_tool_call(tool_call)

        tool_response = {
            "agent": "planner",
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": feedback,
        }
        self.planner_conversation.append(tool_response)
        self.conversation_log.append(tool_response)

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

        # 把 message 中任何非可序列化的欄位轉換為字串或 dict
        serializable_log = []

        for msg in self.conversation_log:
            if isinstance(msg, dict):
                msg_copy = msg.copy()
                if "tool_calls" in msg_copy:
                    msg_copy["tool_calls"] = [tc.model_dump() if hasattr(tc, "model_dump") else str(tc)
                                              for tc in msg_copy["tool_calls"]]
                serializable_log.append(msg_copy)
            else:
                serializable_log.append(str(msg))

        session_summary = {
            "question": self.conversation_log[1]["content"].replace("Question: ", ""),
            "conversations": serializable_log,
            "token_usage": self.total_tokens
        }

        data.append(session_summary)

        with open(self.summary_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False,
                      indent=4)  # Save updated data

    def run(self, question):
        logging.info(f"Starting new session with question: {question}")

        user_question = {"role": "user", "content": f"Question: {question}"}

        self.planner_conversation.append(user_question)

        self.conversation_log = [{"agent": "planner", **msg}
                                 for msg in self.planner_conversation]

        self.questions.append(question)

        turn = 1

        while turn <= self.max_turns:
            logging.info(f"Turn {turn}: Planner's action.")

            # 1. Generate Thought + Action
            response, usage = self.planner.generate_response(
                self.planner_conversation, tools=self.tools, tool_choice="required")
            logging.info(f"LLM Response:\n{response}")

            self.total_tokens.append({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            })

            # 2. Check if the LLM requested a function call
            if response.tool_calls:
                assistant_response = {"agent": "planner", "role": "assistant",
                                      "tool_calls": response.tool_calls}

                for tool_call in response.tool_calls:
                    state, feedback = self.handle_tool_call(tool_call)

                    if state == "retry":
                        logging.warning(f"Retrying Turn {turn}...")
                        turn -= 1
                        continue

                    self.planner_conversation.append(assistant_response)
                    self.conversation_log.append(assistant_response)

                    if state == "decompose":
                        tool_response = {
                            "agent": "planner",
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(feedback, ensure_ascii=False),
                        }
                        self.planner_conversation.append(tool_response)
                        self.conversation_log.append(tool_response)

                        logging.info(f"sub-answers: {str(feedback)}")

                    elif state == "answer":
                        tool_response = {
                            "agent": "planner",
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": feedback,
                        }
                        self.planner_conversation.append(tool_response)
                        self.conversation_log.append(tool_response)

                        final_answer_request = {
                            "role": "user",
                            "content": (
                                "Please organize the information you have gathered and write a complete and comprehensive answer to the original question.")}

                        self.planner_conversation.append(final_answer_request)
                        self.conversation_log.append(final_answer_request)

                        response, usage = self.planner.generate_response(
                            self.planner_conversation, tools=self.tools, tool_choice="none")

                        logging.info(f"LLM Response:\n{response.content}")

                        self.total_tokens.append({"prompt_tokens": usage.prompt_tokens,
                                                  "completion_tokens": usage.completion_tokens,
                                                  "total_tokens": usage.total_tokens})

                        self.conversation_log.append(
                            {"agent": "planner", "role": "assistant", "content": response.content})

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

        self.planner_conversation.append(final_answer_request)
        self.conversation_log.append(final_answer_request)

        response, usage = self.planner.generate_response(
            self.planner_conversation, tools=self.tools, tool_choice="none")

        self.total_tokens.append({"prompt_tokens": usage.prompt_tokens,
                                  "completion_tokens": usage.completion_tokens,
                                  "total_tokens": usage.total_tokens})

        self.conversation_log.append(
            {"agent": "planner", "role": "assistant", "content": response.content})

        self._save_summary()
        return response.content
