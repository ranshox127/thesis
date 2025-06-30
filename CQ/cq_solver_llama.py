import json
import logging
import os
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from lib.agent_llama import Agent_llama
from lib.tools import search


class Node(TypedDict):
    node_id: str
    question: str
    annotation: str


class Edge(TypedDict):
    start: str
    end: str
    annotation: str


class Q_DAG:
    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []

    def add_root(self, question: str) -> None:
        root_node = Node(node_id="Q", question=question, annotation="")
        self.nodes.append(root_node)

    def get_node_question(self, node_id: str) -> Optional[Node]:
        for node in self.nodes:
            if node["node_id"] == node_id:
                return node["question"]
        return None

    def derive_question(self, new_sub_question: str, edge_annotation: str, parent_id: str) -> tuple[str, str]:
        """
        根據 parent_id，派生出一個新的子節點（question）。
        - 自動產生子節點 id，格式為 Q.1、Q.1.1 等（依據 parent_id）。
        - 如果該子問題已存在，就不重複建立節點，但可以重複建立「新的父邊」。
        - 檢查是否已有相同的邊；如果有就跳過。
        - 檢查是否會造成循環（違反 DAG）。

        回傳子節點 id（不論新建或重用）。
        """
        existing_ids = {node["node_id"] for node in self.nodes}
        parent_ids = {node["node_id"] for node in self.nodes}

        # 檢查 parent_id 是否存在
        if parent_id not in parent_ids:
            raise ValueError(f"Parent node '{parent_id}' does not exist.")

        # 想添加的子問題是否存在
        existing_node_id = None
        for node in self.nodes:
            if self._normalize(node["question"]) == self._normalize(new_sub_question):
                existing_node_id = node["node_id"]
                break

        # 若子問題不存在，則建立新節點
        if existing_node_id is None:
            # 完全新節點，可直接加入，不需 cycle 檢查
            base = parent_id
            i = 1
            while True:
                new_id = f"{base}.{i}"
                if new_id not in existing_ids:
                    break
                i += 1

            self.nodes.append(
                Node(node_id=new_id, question=new_sub_question, annotation=""))
            self.edges.append(
                Edge(start=parent_id, end=new_id, annotation=edge_annotation))
            return new_id, "new node added"

        # 如果子問題已存在，則使用現有的 ID
        else:
            new_id = existing_node_id
            # 檢查是否已有這條邊
            for edge in self.edges:
                if edge["start"] == parent_id and edge["end"] == new_id:
                    raise ValueError(
                        f"Edge from '{parent_id}' to '{new_id}' already exists.")

            # 子節點已存在，需檢查加入這條邊會不會形成環
            temp_edges = self.edges + \
                [Edge(start=parent_id, end=new_id, annotation=edge_annotation)]
            if self._has_cycle(temp_edges):
                raise ValueError(
                    f"Adding edge from '{parent_id}' to '{new_id}' would create a cycle.")

            self.edges.append(
                Edge(start=parent_id, end=new_id, annotation=edge_annotation))
            return new_id, "new edge added"

    def _has_cycle(self, edges: List[Edge]) -> bool:
        """簡單的 DFS 來偵測是否有循環。"""

        from collections import defaultdict

        graph = defaultdict(list)
        for edge in edges:
            graph[edge["start"]].append(edge["end"])

        visited = set()
        in_path = set()

        def dfs(node):
            if node in in_path:
                return True
            if node in visited:
                return False
            visited.add(node)
            in_path.add(node)
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            in_path.remove(node)
            return False

        for node in graph:
            if dfs(node):
                return True
        return False

    def _normalize(self, q: str) -> str:
        return q.lower().strip().rstrip("?。！？")

    def update_node(self, node_id: str, new_annotation: str) -> None:
        """
        根據 node_id 更新該節點的 annotation。
        - 檢查 node_id 是否存在。
        """
        for node in self.nodes:
            if node["node_id"] == node_id:
                node["annotation"] = new_annotation
                return

        raise ValueError(f"Node with id '{node_id}' not found.")

    def export_DAG(self) -> str:
        """
        將目前 DAG（nodes 與 edges）以 dict 形式包裝並轉為 JSON 字串。
        回傳格式：
        {
            "nodes": [...],
            "edges": [...]
        }
        """
        dag_dict = {
            "nodes": self.nodes,
            "edges": self.edges
        }
        return json.dumps(dag_dict, ensure_ascii=False, indent=2)


CQ_SYSTEM_PROMPT = """You are a Planner Agent designed to reason through complex, open-ended, or ambiguous questions by constructing, reflecting on, and expanding a directed acyclic graph (DAG) of interrelated sub-questions. Your task is not simply to retrieve answers, but to actively explore the question space, refine your understanding, and make informed decisions about when the original question has been sufficiently addressed.

---

## Problem Space Representation: The Question DAG

The DAG is your evolving internal model of the problem. It represents your reasoning process — how the main question relates to sub-questions, intermediate knowledge, and reflections.
Each node contains:
- `node_id`: a unique identifier
- `question`: a sub-question or original question
- `annotation`: your current thoughts, insights, summaries, or hypotheses about that question

Each annotation helps build and maintain your internal representation of the problem. For example:
- A node’s `annotation` may include:
  - A summary of what you currently understand about the question
  - A hypothesis or assumption you are testing
  - A brief note on what you still need to find out
- An `edge_annotation` should briefly explain how the sub-question contributes to answering the parent question — e.g., cause-effect, component, condition, clarification, definition, comparison, or implication.
---

## Input Format

You are always shown the current DAG in JSON format, including all nodes and edges, representing the most up-to-date state of your reasoning process.

---

## Key Reasoning Guidelines

- You cannot delete nodes or edges. Even if a previous path turns out to be incorrect or irrelevant, leave it intact and revise your understanding through `update()`. This mimics how humans preserve earlier lines of thought for traceability, reflection, and learning from missteps.
- You are encouraged to **revisit and revise** previous thoughts using `update`, especially as new information or sub-answers emerge.
- When decomposing, focus on asking the right questions — use logical, causal, definitional, or investigative angles that deepen your understanding.
- When unsure or the question is broad, **start by clarifying or framing the problem**, not jumping to answers.
- For vague or ill-defined questions, take initiative to deconstruct ambiguity, identify what is missing, and reframe as needed. You shape the problem space.

---

## Your Tools

You have three core actions to build and navigate the problem space:

1. **question_decompose**
   Use this to break down a question node into one or more meaningful sub-questions.
   - You may decompose multiple nodes at once.
   - Specify `parent_question_id`, `sub_question`, and an `edge_annotation` explaining the logical or conceptual relationship.
   - Multiple parents pointing to the same sub-question are allowed.
   - Keep the graph acyclic.

   Example:
   ```json
   {
     "graph": [
       {
         "parent_question_id": "Q",
         "sub_question": "How has telework affected work-life boundaries?",
         "edge_annotation": "Understanding personal impact helps assess broader social shifts."
       },
       {
         "parent_question_id": "Q.1",
         "sub_question": "Does telework reinforce or reduce social inequality?",
         "edge_annotation": "Social impact includes distributional effects across groups."
       }
     ]
   }
    ```

2.  **update**
    Use this to revise or expand the annotation of existing nodes.
    - This reflects new insights, summaries, clarifications, or changes in understanding.
    - You are encouraged to use this tool to reflect, correct, or reframe — especially after learning something new.
    - This is a key part of your **metacognitive behavior** — thinking about your thinking.
    
    Example:
    ```json
    {
      "nodes": [
        {
          "question_id": "Q.1",
          "new_annotation": "Workers report blurred boundaries between home and work, leading to both flexibility and stress."
        },
        {
          "question_id": "Q.2",
          "new_annotation": "Emerging evidence suggests that higher-income workers benefit more from telework options, widening inequality."
        }
      ]
    }
    ```

3.  **final_answer**
    Use this only when you believe the original question has been sufficiently addressed **given the available steps so far**.  
    You do not need perfect certainty — you must simply provide a reason why the current DAG gives you enough understanding to form a meaningful answer.
    - Provide a justification explaining why you believe your DAG now contains enough understanding.
    - Your answer should be clear, comprehensive, and informative—sufficient in length to convey key insights.
    - You may use paragraph or bullet point format as appropriate.
    - Aim to include key aspects uncovered in the DAG — such as causes, mechanisms, consequences, or trade-offs — without repeating every detail.
    
    Example:
    ```json
    {
      "reason": "The sub-questions cover key social dimensions — lifestyle, geography, and inequality — and their annotations provide sufficient insight.",
    }
    ```

---

## Metacognitive Expectations

This is not a static search task — it is an evolving thinking process.

- Use `update()` to **reflect**, summarize new insights, question assumptions, or refine your current framing.
- Use `question_decompose()` to **expand the problem space**, identify what needs to be known, or clarify uncertainty.
- Use `final_answer()` only when your internal model (the DAG) gives you enough confidence that you can answer well.
- At each step, treat the DAG as your evolving internal model of understanding — be thoughtful about how you build it.

- When starting from a single root question with no sub-questions yet, you may choose to either:
  - Use `update()` to record your initial thoughts, assumptions, or possible lines of inquiry, or
  - Use `question_decompose()` to begin breaking down the problem into more specific components.
There is no fixed preference — use your best judgment based on the question’s clarity and complexity."""

load_dotenv()


class CQ_Solver_llama:
    def __init__(self, llm, system_prompt, max_turns=9, debug_log="llama_debug.log", summary_json="llama_summary.json"):
        self.planner = Agent_llama(llm=llm)
        self.searcher = Agent_llama(llm=llm)
        self.max_turns = max_turns
        self.planner_conversation = [
            {"role": "system", "content": system_prompt}]
        self.conversation_log = []  # 用於詳細記錄每一條訊息，不做傳入模型用
        self.DAG = Q_DAG()

        # Setup detailed debug logging
        logging.basicConfig(filename=debug_log, level=logging.DEBUG,
                            format="%(asctime)s [%(levelname)s] %(message)s", encoding="utf-8")
        logging.info("\n=== New llama CQ Execution Started ===\n")

        # Summary log file
        self.summary_json = summary_json

        # Initialize JSON file if it doesn't exist
        if not os.path.exists(self.summary_json):
            with open(self.summary_json, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=4)

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
            graph_entries = arguments.get("graph", [])
            new_ids = []

            for entry in graph_entries:
                parent_id = entry.get("parent_question_id")
                sub_question = entry.get("sub_question")
                edge_annotation = entry.get("edge_annotation")

                try:
                    new_id, status = self.DAG.derive_question(
                        new_sub_question=sub_question,
                        edge_annotation=edge_annotation,
                        parent_id=parent_id
                    )
                except ValueError as e:
                    logging.warning(str(e))
                    continue

                # 確定新增成功後再做後續
                new_ids.append(new_id)

                if status == "new node added":

                    sub_node_info = self.get_sub_node_info(
                        root_question=self.DAG.get_node_question("Q"),
                        parent_question=self.DAG.get_node_question(parent_id),
                        annotation=edge_annotation,
                        target_question=sub_question)

                    self.DAG.update_node(new_id, sub_node_info)

            if not new_ids:
                return "retry", None

            return "decompose", self.DAG.export_DAG()

        elif function_name == "update":
            nodes = arguments.get("nodes", [])

            for entry in nodes:
                question_id = entry.get("question_id")
                new_annotation = entry.get("new_annotation")
                try:
                    self.DAG.update_node(
                        node_id=question_id, new_annotation=new_annotation)
                except ValueError as e:
                    logging.warning(str(e))
                    continue

            return "update", self.DAG.export_DAG()

        elif function_name == "final_answer":
            logging.info("Generating final answer...")
            return "answer", arguments.get("reason")

        else:
            logging.warning(f"Unknown function requested: {function_name}")
            return "retry", None

    def get_sub_node_info(self, root_question, parent_question, annotation, target_question):
        searcher_system_prompt = """You are a research assistant in a multi-agent system designed to answer complex and ambiguous questions. Your role is to assist in answering sub-questions by generating search queries and summarizing relevant results.

        You will be given:
        - A root question: the user's original, high-level question
        - A parent sub-question: a more specific inquiry derived from the root
        - A target sub-question: the current question to be addressed
        - An edge annotation: an explanation of how the target sub-question connects to its parent (i.e. the reasoning for asking it)

        Your job consists of two steps:

        1. **Query Generation**  
        Based on the context (root question, parent question, edge annotation), write the most focused and effective search query to help retrieve useful information to address the target sub-question.  
        - You are not merely rewriting the question. You must *interpret* the intent, especially if the edge annotation implies a deeper or more specific angle.
        - For example, if the annotation indicates causal reasoning or a historical background is needed, reflect that in your query.

        2. **Summary Generation**  
        After receiving retrieved results, produce a concise but contextually appropriate summary that helps address the target sub-question.  
        - The summary should match the *type of information* implied by the edge annotation.  
        - Sometimes this may be a factual list, a comparison, a causal explanation, or a brief definition.  
        - Avoid general or vague summaries; tailor the content to the sub-question's intent.

        Be flexible: the edge annotation may imply different kinds of answers (e.g. factual, explanatory, evaluative), and your output should reflect that.

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

        searcher_conversation = [{"role": "system", "content": searcher_system_prompt},
                                 {"role": "user",
                                  "content": (
                                      f"Root question: {root_question}\n"
                                      f"Parent question: {parent_question}\n"
                                      f"Target sub-question: {target_question}\n"
                                      f"Edge annotation: {annotation}"
                                  )
                                  }]

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

        tool_response = {"agent": "searcher",
                         "role": "user", "content": summary_result}

        self.conversation_log.append(assistant_response)
        self.conversation_log.append(tool_response)

        return summary_result

    def final_answer(self):
        logging.info("Generating final answer...")

        if self.conversation_log[-1]["content"] == "After reflecting, please continue to act.":
            self.planner_conversation[-1]["content"] = ("You have reached the maximum number of rounds of action and you must force a final answer.\n\n"
                                                        "Please organize the information you have gathered and write a complete and comprehensive answer to the original question.")
            self.conversation_log[-1]["content"] = ("You have reached the maximum number of rounds of action and you must force a final answer.\n\n"
                                                    "Please organize the information you have gathered and write a complete and comprehensive answer to the original question.")
        else:
            self.planner_conversation[-1]["content"] += "\n\n" + ("You have reached the maximum number of rounds of action and you must force a final answer.\n\n"
                                                                  "Please organize the information you have gathered and write a complete and comprehensive answer to the original question.")
            self.conversation_log[-1]["content"] += "\n\n" + ("You have reached the maximum number of rounds of action and you must force a final answer.\n\n"
                                                              "Please organize the information you have gathered and write a complete and comprehensive answer to the original question.")

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

        self.DAG.add_root(question)

        user_question = {"role": "user", "content": self.DAG.export_DAG()}

        self.planner_conversation.append(user_question)

        self.conversation_log = [{"agent": "planner", **msg}
                                 for msg in self.planner_conversation]

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

            # 2.2 record the tool

            if state == "decompose":
                logging.info(f"Graph: {feedback}")

                Critical_Evaluation_request = {
                    "role": "user",
                    "content": (
                        "You have just decomposed part of the problem into new sub-questions. Now, take a moment to reflect on your current understanding and planning:\n\n"
                        "1. Have the new sub-questions changed or expanded your understanding of the original question or any part of the problem space?\n"
                        "    - If yes, consider using `update()` to revise or refine your current annotations.\n\n"
                        "2. Are there any remaining uncertainties, vague concepts, or areas that seem underdeveloped?\n"
                        "    - If yes, you may want to continue decomposing or exploring before concluding.\n\n"
                        "3. If you believe you are ready to answer the original question, pause and verify your confidence:\n"
                        "    - Formulate **a few critical questions** that would challenge or test your current answer.\n"
                        "    - If your answer still holds after these checks, then proceed with `final_answer()`.\n"
                        "    - Otherwise, revise your thinking or explore further as needed.\n\n"
                        "Choose your next tool based on your reflection."
                    )
                }

                tool_response = {"role": "user", "content": feedback +
                                 '\n\n' + Critical_Evaluation_request["content"]}
                self.planner_conversation.append(tool_response)
                self.conversation_log.append(
                    {**tool_response, "agent": "planner"})

                # self.planner_conversation.append(Critical_Evaluation_request)
                # self.conversation_log.append(
                #     {**Critical_Evaluation_request, "agent": "planner"})

                response = self.planner.generate_response(
                    self.planner_conversation, action="CE")

                Critical_Evaluation_response = {
                    "role": "assistant", "content": response.critical_evaluation}
                self.planner_conversation.append(Critical_Evaluation_response)
                self.conversation_log.append(
                    {**Critical_Evaluation_response, "agent": "planner"})

                continue_request = {"role": "user", "content": (
                    "After reflecting, please continue to act.")}

                self.planner_conversation.append(continue_request)
                self.conversation_log.append(
                    {**continue_request, "agent": "planner"})

            elif state == "update":
                logging.info(f"Graph: {feedback}")

                tool_response = {"role": "user", "content": feedback}
                self.planner_conversation.append(tool_response)
                self.conversation_log.append(
                    {**tool_response, "agent": "planner"})

            elif state == "answer":
                final_answer_request = {"role": "user", "content": (
                    "Please organize the information you have gathered and write a complete and comprehensive answer to the original question.")}

                tool_response = {
                    "role": "user", "content": final_answer_request["content"]}
                self.planner_conversation.append(tool_response)
                self.conversation_log.append(
                    {**tool_response, "agent": "planner"})

                # self.planner_conversation.append(final_answer_request)
                # self.conversation_log.append(
                #     {**final_answer_request, "agent": "planner"})

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
