from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from .agents.base import AgentBase
from .agents.rag import KnowledgeBaseAgent
from .agents.math_agent import CalculatorAgent
from .agents.datetime_agent import DateTimeAgent
from .agents.search import WebSearchAgent
from .agents.analyst import AnalystAgent

WORKER_TIMEOUT = 8

class RouterAgent(AgentBase):
    NAME = "Router"

    # Khởi tạo tác tử điều phối trung tâm giữa đồ thị và các tác tử thành phần.
    def __init__(self):
        super().__init__(called_by="GraphNode")

    # Điều phối thực thi song song các tác tử thành phần qua ThreadPool và thu thập kết quả tại rào cản đồng bộ.
    def run(self, query: str, intent: str, kb_context: str = "") -> dict:
        self._calls = []
        results = {
            "kb_context": kb_context, "calc_result": "", "datetime_result": "",
            "web_results": "", "analysis": "", "agent_timings": [], "call_graph": [], "log": [],
        }
        workers = self._select_workers(query, intent, kb_context)
        if not workers:
            results["log"].append(self._log(f"Không có worker nào được kích hoạt cho intent='{intent}'."))
            return results
        results["log"].append(self._log(f"Kích hoạt {len(workers)} worker song song cho intent='{intent}': {[w[0] for w in workers]}"))
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {}
            for worker_key, agent, kwargs in workers:
                self._record_call(agent.NAME)
                futures[pool.submit(agent.safe_run, **kwargs)] = (worker_key, agent)
            for future in as_completed(futures, timeout=WORKER_TIMEOUT + 2):
                worker_key, agent = futures[future]
                try:
                    worker_result = future.result(timeout=WORKER_TIMEOUT)
                except TimeoutError:
                    worker_result = {"error": "timeout", "message": f"[{agent.NAME}] Quá thời gian ({WORKER_TIMEOUT}s)."}
                    results["log"].append(self._log(f"⚠ {agent.NAME} bị timeout ({WORKER_TIMEOUT}s)."))
                except Exception as exc:
                    worker_result = {"error": str(exc), "message": f"[{agent.NAME}] Lỗi: {exc}"}
                    results["log"].append(self._log(f"⚠ {agent.NAME} lỗi: {exc}"))
                results["agent_timings"].append(agent.get_timing())
                results["call_graph"].extend(agent.get_call_edges())
                self._merge_worker_result(results, worker_key, worker_result)
                results["log"].append(self._log(f"✓ {agent.NAME} hoàn thành ({agent._duration:.3f}s)"))
        return results

    # Lựa chọn danh sách các tác tử thực thi phù hợp với ý định truy vấn đã được phân loại.
    def _select_workers(self, query: str, intent: str, kb_context: str) -> list[tuple[str, AgentBase, dict]]:
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
            workers.append(("rag", KnowledgeBaseAgent(called_by=self.NAME), {"query": query, "intent": intent}))
            workers.append(("web", WebSearchAgent(called_by=self.NAME), {"query": query}))
            workers.append(("analyst", AnalystAgent(called_by=self.NAME), {"topic": query, "context": kb_context}))
            workers.append(("calc", CalculatorAgent(called_by=self.NAME), {"query": query}))
        elif intent == "plan":
            workers.append(("rag", KnowledgeBaseAgent(called_by=self.NAME), {"query": query, "intent": intent}))
            workers.append(("web", WebSearchAgent(called_by=self.NAME), {"query": query}))
            workers.append(("analyst", AnalystAgent(called_by=self.NAME), {"topic": query, "context": kb_context}))
        elif intent == "greet":
            workers.append(("analyst", AnalystAgent(called_by=self.NAME), {"topic": "greeting", "context": query}))
        else:
            workers.append(("rag", KnowledgeBaseAgent(called_by=self.NAME), {"query": query, "intent": intent}))
            workers.append(("web", WebSearchAgent(called_by=self.NAME), {"query": query}))
            workers.append(("analyst", AnalystAgent(called_by=self.NAME), {"topic": query, "context": kb_context}))
        return workers

    # Gộp kết quả đầu ra từ tác tử thành phần vào cấu trúc dữ liệu kết quả chung.
    @staticmethod
    def _merge_worker_result(results: dict, worker_key: str, worker_result: dict) -> None:
        key_mapping = {
            "rag": ("kb_context", ""),
            "calc": ("calc_result", ""),
            "dt": ("datetime_result", ""),
            "web": ("web_results", ""),
            "analyst": ("analysis", ""),
        }
        if worker_key in key_mapping:
            field, default = key_mapping[worker_key]
            results[field] = worker_result.get(field, default)
