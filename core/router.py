"""
core/router.py — RouterAgent: Bộ điều phối đa luồng (Fan-out / Fan-in).

Nguyên lý hoạt động:
  PHA 1 (Song song): Kích hoạt đồng thời các Worker Agent trong ThreadPool.
  BARRIER: Gom tất cả kết quả, đồng bộ hóa an toàn vào AgentState.
  PHA 2 (Tuần tự): Chuyển sang Planner → Validator → ResponseGenerator.

Router chịu trách nhiệm:
  1. Phân tích intent để quyết định kích hoạt những Agent nào.
  2. Chạy song song với ThreadPoolExecutor (max_workers=6).
  3. Bọc timeout độc lập (5 giây) cho mỗi worker.
  4. Thu thập kết quả và timing từ tất cả các worker.
"""

import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from .agents.base import AgentBase
from .agents.rag import KnowledgeBaseAgent
from .agents.math_agent import CalculatorAgent
from .agents.datetime_agent import DateTimeAgent
from .agents.search import WebSearchAgent
from .agents.analyst import AnalystAgent


# Thời gian chờ tối đa cho mỗi Worker Agent (giây)
WORKER_TIMEOUT = 8


class RouterAgent(AgentBase):
    """
    Bộ điều phối trung tâm — Fan-out / Fan-in.
    Kích hoạt các Worker Agent song song dựa theo intent,
    sau đó gom kết quả trả về cho pha tuần tự tiếp theo.
    """

    NAME = "Router"

    def __init__(self):
        super().__init__(called_by="GraphNode")

    def run(
        self,
        query: str,
        intent: str,
        kb_context: str = "",
    ) -> dict:
        """
        Phân tích intent và kích hoạt các Worker Agent tương ứng.

        Returns:
            dict chứa kết quả từ tất cả các Worker + timing + call_graph.
        """
        self._calls = []
        results = {
            "kb_context": kb_context,
            "calc_result": "",
            "datetime_result": "",
            "web_results": "",
            "analysis": "",
            "agent_timings": [],
            "call_graph": [],
            "log": [],
        }

        # ── Xác định danh sách Worker cần kích hoạt ──────────────────────
        workers = self._select_workers(query, intent, kb_context)

        if not workers:
            results["log"].append(self._log(f"Không có worker nào được kích hoạt cho intent='{intent}'."))
            return results

        results["log"].append(
            self._log(f"Kích hoạt {len(workers)} worker song song cho intent='{intent}': "
                       f"{[w[0] for w in workers]}")
        )

        # ── PHA 1: Fan-out — Chạy song song ─────────────────────────────
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {}
            for worker_key, agent, kwargs in workers:
                self._record_call(agent.NAME)
                future = pool.submit(agent.safe_run, **kwargs)
                futures[future] = (worker_key, agent)

            # ── BARRIER: Gom kết quả (Fan-in) ───────────────────────────
            for future in as_completed(futures, timeout=WORKER_TIMEOUT + 2):
                worker_key, agent = futures[future]
                try:
                    worker_result = future.result(timeout=WORKER_TIMEOUT)
                except TimeoutError:
                    worker_result = {
                        "error": "timeout",
                        "message": f"[{agent.NAME}] Quá thời gian ({WORKER_TIMEOUT}s).",
                    }
                    results["log"].append(
                        self._log(f"⚠ {agent.NAME} bị timeout ({WORKER_TIMEOUT}s).")
                    )
                except Exception as exc:
                    worker_result = {
                        "error": str(exc),
                        "message": f"[{agent.NAME}] Lỗi: {exc}",
                    }
                    results["log"].append(self._log(f"⚠ {agent.NAME} lỗi: {exc}"))

                # Ghi nhận timing và call graph
                results["agent_timings"].append(agent.get_timing())
                results["call_graph"].extend(agent.get_call_edges())

                # Gán kết quả vào đúng vùng dữ liệu trong State
                self._merge_worker_result(results, worker_key, worker_result)

                results["log"].append(
                    self._log(f"✓ {agent.NAME} hoàn thành ({agent._duration:.3f}s)")
                )

        return results

    def _select_workers(
        self, query: str, intent: str, kb_context: str
    ) -> list[tuple[str, AgentBase, dict]]:
        """
        Xác định danh sách Worker Agent cần kích hoạt dựa trên intent.

        Returns:
            List of (worker_key, agent_instance, kwargs_dict)
        """
        workers = []

        if intent == "calculate":
            workers.append(("calc", CalculatorAgent(called_by=self.NAME), {"query": query}))

        elif intent == "datetime":
            workers.append(("dt", DateTimeAgent(called_by=self.NAME), {"query": query}))

        elif intent == "search":
            workers.append(("web", WebSearchAgent(called_by=self.NAME), {"query": query}))
            workers.append(("analyst", AnalystAgent(called_by=self.NAME), {"topic": query, "context": kb_context}))

        elif intent == "rag":
            workers.append(("rag", KnowledgeBaseAgent(called_by=self.NAME), {"query": query, "intent": intent}))
            workers.append(("analyst", AnalystAgent(called_by=self.NAME), {"topic": query, "context": kb_context}))

        elif intent == "analyze":
            # Phân tích toàn diện: Kích hoạt 4 Worker song song
            workers.append(("rag", KnowledgeBaseAgent(called_by=self.NAME), {"query": query, "intent": intent}))
            workers.append(("web", WebSearchAgent(called_by=self.NAME), {"query": query}))
            workers.append(("analyst", AnalystAgent(called_by=self.NAME), {"topic": query, "context": kb_context}))
            workers.append(("calc", CalculatorAgent(called_by=self.NAME), {"query": query}))

        elif intent == "plan":
            workers.append(("rag", KnowledgeBaseAgent(called_by=self.NAME), {"query": query, "intent": intent}))
            workers.append(("web", WebSearchAgent(called_by=self.NAME), {"query": query}))
            workers.append(("analyst", AnalystAgent(called_by=self.NAME), {"topic": query, "context": kb_context}))

        elif intent == "greet":
            # Chào hỏi đơn giản: Chỉ cần Analyst tạo phản hồi
            workers.append(("analyst", AnalystAgent(called_by=self.NAME), {"topic": "greeting", "context": query}))

        else:  # general
            workers.append(("rag", KnowledgeBaseAgent(called_by=self.NAME), {"query": query, "intent": intent}))
            workers.append(("web", WebSearchAgent(called_by=self.NAME), {"query": query}))
            workers.append(("analyst", AnalystAgent(called_by=self.NAME), {"topic": query, "context": kb_context}))

        return workers

    @staticmethod
    def _merge_worker_result(results: dict, worker_key: str, worker_result: dict) -> None:
        """Gán kết quả của từng worker vào đúng trường trong State."""
        if worker_key == "rag":
            results["kb_context"] = worker_result.get("kb_context", "")
        elif worker_key == "calc":
            results["calc_result"] = worker_result.get("calc_result", "")
        elif worker_key == "dt":
            results["datetime_result"] = worker_result.get("datetime_result", "")
        elif worker_key == "web":
            results["web_results"] = worker_result.get("web_results", "")
        elif worker_key == "analyst":
            results["analysis"] = worker_result.get("analysis", "")

