from typing import Optional
from .storage import ChatHistoryStore


class ProcessInspector:
    def __init__(self, store: Optional[ChatHistoryStore] = None):
        self.store = store or ChatHistoryStore()

    def get_process(self, session_id: str, turn_id: Optional[int] = None):
        return self.store.get_process(session_id, turn_id)

    def render_html(self, session_id: str, turn_id: Optional[int] = None) -> str:
        report = self.get_process(session_id, turn_id)
        if not report:
            return f"<h1>Session not found: {session_id}</h1>"

        title = f"Multi-Agent Process for session {session_id}"
        html = [
            "<html><head><meta charset=\"utf-8\"><title>Multi-Agent Process</title>",
            "<style>body{font-family:Segoe UI,Arial,sans-serif;padding:24px;color:#111;line-height:1.6;}",
            "table{border-collapse:collapse;width:100%;margin:16px 0;}td,th{border:1px solid #ccc;padding:8px;text-align:left;}",
            "th{background:#f3f4f6;}pre{background:#f7f7f7;padding:12px;border-radius:8px;overflow-x:auto;}",
            "h1,h2,h3{color:#222;}li{margin-bottom:6px;}",
            "</style></head><body>",
            f"<h1>{title}</h1>",
            f"<p><strong>User ID:</strong> {report.get('user_id') or 'N/A'} &nbsp;|&nbsp; <strong>Turn:</strong> {report.get('turn_id')} &nbsp;|&nbsp; <strong>Timestamp:</strong> {report.get('timestamp')}</p>",
            f"<p><strong>User Query:</strong> {report.get('user_query') or 'N/A'}</p>",
            f"<p><strong>Intent:</strong> {report.get('intent') or 'N/A'}</p>",
            "<h2>Final Answer</h2>",
            f"<pre>{report.get('final_answer') or ''}</pre>",
            "<h2>Agent Timings</h2>",
            "<table><thead><tr><th>#</th><th>Agent</th><th>Called By</th><th>Duration (s)</th><th>Calls</th></tr></thead><tbody>",
        ]
        for idx, timing in enumerate(report.get("agent_timings", []), 1):
            calls = ", ".join(timing.get("calls", [])) or "-"
            html.append(
                f"<tr><td>{idx}</td><td>{timing.get('agent')}</td><td>{timing.get('called_by')}</td>"
                f"<td>{timing.get('duration')}</td><td>{calls}</td></tr>"
            )
        html.append("</tbody></table>")
        html.append("<h2>Call Graph</h2>")
        cg = report.get("call_graph", [])
        if cg:
            html.append("<table><thead><tr><th>#</th><th>From</th><th>To</th><th>Timestamp</th></tr></thead><tbody>")
            for idx, edge in enumerate(cg, 1):
                html.append(
                    f"<tr><td>{idx}</td><td>{edge.get('from')}</td><td>{edge.get('to')}</td><td>{edge.get('ts')}</td></tr>"
                )
            html.append("</tbody></table>")
        else:
            html.append("<p>No call graph data available.</p>")

        html.append("<h2>Log</h2>")
        html.append("<ul>")
        for line in report.get("log", []):
            html.append(f"<li>{line}</li>")
        html.append("</ul>")
        html.append("</body></html>")
        return "".join(html)
