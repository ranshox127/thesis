import json
import logging
import os

from dotenv import load_dotenv
from lib.agent_llama import Agent_llama
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
   
   Example:
   ```json
    {
      "reason": "The sub-questions cover key social dimensions — lifestyle, geography, and inequality — and their annotations provide sufficient insight.",
    }
    ```

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
2. Then, **invoke one of the tools** using structured output (function call)."""

load_dotenv()


class MindSearch:
    def __init__(self, llm, system_prompt, max_turns=9, debug_log="llama_mindsearch_debug.log", summary_json="llama_mindsearch_summary.json"):
        self.planner = Agent_llama(llm=llm)
        self.searcher = Agent_llama(llm=llm)
        self.max_turns = max_turns
        self.planner_conversation = [
            {"role": "system", "content": system_prompt}]
        self.conversation_log = []  # 用於詳細記錄每一條訊息，不做傳入模型用
        self.questions = []

        # Setup detailed debug logging
        logging.basicConfig(filename=debug_log, level=logging.DEBUG,
                            format="%(asctime)s [%(levelname)s] %(message)s", encoding="utf-8")
        logging.info("\n=== New mindsearch Execution Started ===\n")

        # Summary log file
        self.summary_json = summary_json

        # Initialize JSON file if it doesn't exist
        if not os.path.exists(self.summary_json):
            with open(self.summary_json, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=4)

    def _normalize(self, q: str) -> str:
        return q.lower().strip().rstrip("?。！？")

    def handle_tool_call(self, tool_call):
        """Executes the function requested by OpenAI's function calling system."""
        function_name = tool_call.tool_name

        if function_name == "search":
            query = tool_call.query

            logging.info(f"Executing search for: {query}")
            return "search", search(query, max_results=5)

        elif function_name == "summary":
            summary = tool_call.summary

            logging.info(f"summary: {summary}")
            return "summary", summary

        try:
            arguments = json.loads(tool_call.tool_parameters)
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

        elif function_name == "final_answer":
            logging.info("Generating final answer...")
            return "answer", arguments.get("reason")

        else:
            logging.warning(f"Unknown function requested: {function_name}")
            return "retry", None

    def get_sub_answers(self, question, sub_questions):
        sub_answers = []
        searcher_system_prompt = """You are an AI assistant designed to answer sub-questions. You will be presented with original questions and sub-questions that you need to answer using a search engine.
        Your task is to.
        1. come up with the most appropriate query based on the original question and the sub-question.
        2. summarize the search results to answer the sub-question based on the original question and the sub-question.
        
        Use the available tools:
        - `search(query, reason)`: to retrieve relevant information
          Example:
          ```json
          {
            "reason": "To answer the question about dark energy, I need to find out who discovered it.",
            "query": "Who discovered dark energy?"
          }
           ```
        - `summary(text)`: to return your synthesized summary
          Example:
          ```json
          {
            "summary": "Dark energy was discovered by astronomer Edwin Hubble in 1929."
          }
          ```
        """

        for sub_question in sub_questions:

            searcher_conversation = [{"role": "system", "content": searcher_system_prompt},
                                     {"role": "user", "content": f"The parent question: {question}, the sub-question: {sub_question}"}]

            # search
            response = self.searcher.generate_response(
                conversations=searcher_conversation, action="Search")

            search_result = self.handle_tool_call(response)[1]

            assistant_response = {
                "role": "assistant", "content": f"Tool's name:{response.tool_name}\nreason:{response.reason}\nquery:{response.query}"}
            searcher_conversation.append(assistant_response)

            tool_response = {"role": "user", "content": str(search_result)}
            searcher_conversation.append(tool_response)

            # summary

            response = self.searcher.generate_response(
                conversations=searcher_conversation, action="Summary")

            summary_result = self.handle_tool_call(response)[1]

            self.conversation_log.extend([
                {"agent": "searcher", **msg} for msg in searcher_conversation
            ])

            assistant_response = {"agent": "searcher", "role": "assistant",
                                  "content": f"Tool's name:{response.tool_name}"}
            self.conversation_log.append(assistant_response)

            tool_response = {"agent": "searcher",
                             "role": "user", "content": summary_result}
            self.conversation_log.append(tool_response)

            sub_answers.append(
                {"sub_q": sub_question, "answer": summary_result})

        return {
            "parent_q": question,
            "sub_answers": sub_answers
        }

    def final_answer(self):
        logging.info("Generating final answer...")

        final_answer_request = {
            "role": "user", "content": "Please organize the information you have gathered and write a complete and comprehensive answer to the original question."}
        self.planner_conversation.append(final_answer_request)
        self.conversation_log.append(
            {**final_answer_request, "agent": "planner"})

        response = self.planner.generate_response(
            self.planner_conversation, action="FA")

        return response.final_answer

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
            response = self.planner.generate_response(
                self.planner_conversation, action="TC")
            logging.info(f"LLM Response:\n{response}")

            # 2. process agents function call

            # 2.1 record the assistant
            assistant_response = {
                "role": "assistant", "content": f"Tool's name:{response.tool_name}\nTool's parameters:{response.tool_parameters}"}
            self.planner_conversation.append(assistant_response)
            self.conversation_log.append(
                {**assistant_response, "agent": "planner"})

            state, feedback = self.handle_tool_call(response)

            if state == "retry":
                logging.warning(f"Retrying Turn {turn}...")
                turn -= 1
                self.planner_conversation.pop()
                self.conversation_log.pop()
                continue

            if state == "decompose":
                tool_response = {"role": "user", "content": json.dumps(
                    feedback, ensure_ascii=False)}
                self.planner_conversation.append(tool_response)
                self.conversation_log.append(
                    {**tool_response, "agent": "planner"})

                logging.info(f"sub-answers: {str(feedback)}")

            elif state == "answer":
                tool_response = {
                    "role": "user", "content": "Please organize the information you have gathered and write a complete and comprehensive answer to the original question."}
                self.planner_conversation.append(tool_response)
                self.conversation_log.append(
                    {**tool_response, "agent": "planner"})

                response = self.planner.generate_response(
                    self.planner_conversation, action="FA")

                logging.info(f"LLM Response:\n{response.final_answer}")

                self.conversation_log.append(
                    {"agent": "planner", "role": "assistant", "content": response.final_answer})

                self._save_summary()
                logging.info("Final Answer Reached.")
                return response.final_answer

            turn += 1

        logging.warning("Max turns reached. No definitive answer found.")
        final_answer = self.final_answer()

        self.conversation_log.append(
            {"agent": "planner", "role": "assistant", "content": final_answer})

        self._save_summary()
        return final_answer
