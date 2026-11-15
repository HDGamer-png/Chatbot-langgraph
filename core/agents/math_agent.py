import re
import ast
import math
import operator
from .base import AgentBase

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "pi": math.pi,
    "e": math.e,
}

# Đánh giá biểu thức toán học an toàn bằng phân tích cú pháp AST.
def _safe_eval(expr_str: str) -> float:
    expr_str = expr_str.strip()
    if not expr_str:
        raise ValueError("Biểu thức rỗng.")
    tree = ast.parse(expr_str, mode="eval")

    def _eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[type(node.op)](_eval_node(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.func.id](*[_eval_node(arg) for arg in node.args])
        if isinstance(node, ast.Name) and node.id in _SAFE_FUNCTIONS:
            val = _SAFE_FUNCTIONS[node.id]
            if isinstance(val, (int, float)):
                return val
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        raise ValueError(f"Thành phần không được phép: {ast.dump(node)}")

    return _eval_node(tree)

class CalculatorAgent(AgentBase):
    NAME = "Calculator"

    # Khởi tạo tác tử tính toán toán học.
    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)

    # Phân tích câu hỏi, trích xuất biểu thức và tính toán kết quả số học hoặc giải phương trình.
    def run(self, query: str = "", **kwargs) -> dict:
        query = query.strip()
        if not query:
            return {"calc_result": "", "expression": "", "message": "Không có biểu thức."}
        expr = self._extract_expression(query)
        if not expr:
            return {"calc_result": "", "expression": "", "message": "Không tìm thấy biểu thức toán trong câu hỏi."}
        if "=" in expr and re.search(r'[a-zA-Z]', expr):
            res = self._solve_equation(expr)
            if res:
                return res
        try:
            val = _safe_eval(expr)
            formatted = str(int(val)) if isinstance(val, float) and val == int(val) else (f"{val:.6g}" if isinstance(val, float) else str(val))
            return {"calc_result": formatted, "expression": expr, "message": f"Kết quả: {expr} = {formatted}"}
        except Exception as e:
            res = self._sympy_eval(expr)
            return res if res else {"calc_result": "", "expression": expr, "message": f"Không thể tính: {e}"}

    # Trích xuất biểu thức toán học chuẩn từ câu văn bản tự nhiên tiếng Việt.
    def _extract_expression(self, query: str) -> str:
        q = query.lower()
        q = re.sub(r'\b(giải|phương|trình|tìm|nghiệm|tính|giá|trị|biểu|thức|toán|của|là|bằng|cho|tôi|hãy|hỏi|số|kết quả)\b', ' ', q)
        q = q.replace(':', ' ').replace('?', ' ')
        replacements = [
            (r'\bcộng\b', '+'), (r'\btrừ\b', '-'), (r'\bnhân\b', '*'), (r'\bchia\b', '/'),
            (r'\bmũ\b', '**'), (r'\blũy thừa\b', '**'), (r'\bbình phương\b', '**2'), (r'\blập phương\b', '**3'),
            (r'\bcăn bậc hai\b', 'sqrt'), (r'\bcăn bậc 2\b', 'sqrt'), (r'\bcăn\b', 'sqrt'),
            (r'\bphần trăm\b', '/100'), (r'\b%\b', '/100'),
            (r'\bsin\b', 'sin'), (r'\bcos\b', 'cos'), (r'\btan\b', 'tan'), (r'\bgiai thừa\b', 'factorial'),
        ]
        for pattern, repl in replacements:
            q = re.sub(pattern, repl, q)
        q = re.sub(r'\b(sqrt|sin|cos|tan|log|factorial)\s+(\d+(?:\.\d+)?)', r'\1(\2)', q)
        if '=' in q:
            m = re.search(r'([0-9a-zA-Z\s\+\-\*\/\^\(\)\.]+\s*=\s*[0-9a-zA-Z\s\+\-\*\/\^\(\)\.]+)', q)
            if m:
                cand = " ".join([t for t in m.group(1).strip().split() if re.match(r'^[0-9a-zA-Z\+\-\*\/\^\(\)\.\=]+$', t) and (len(t) <= 3 or any(c in t for c in "+-*/^=0123456789"))])
                if '=' in cand and re.search(r'[a-zA-Z]', cand) and re.search(r'\d', cand):
                    return cand
        m_func = re.search(r'((?:sqrt|sin|cos|tan|log|factorial)\s*\([^)]+\))', q)
        if m_func:
            return m_func.group(1).replace(' ', '')
        m_math = re.search(r'([\d\.\s\+\-\*\/\%\^\(\)]+)', q)
        if m_math:
            clean_expr = re.sub(r'\s+', '', m_math.group(1).strip())
            if re.search(r'\d', clean_expr) and len(clean_expr) > 1:
                return clean_expr
        return ""

    # Giải phương trình đại số chứa biến bằng thư viện SymPy.
    def _solve_equation(self, expr: str) -> dict | None:
        try:
            import sympy
            vars_found = set(re.findall(r'[a-zA-Z]', expr)) - {'e', 'E'}
            if not vars_found:
                return None
            var_name = vars_found.pop()
            var = sympy.Symbol(var_name)
            parts = expr.split('=')
            if len(parts) != 2:
                return None
            left = sympy.sympify(re.sub(r'(\d)([a-zA-Z])', r'\1*\2', parts[0].strip()))
            right = sympy.sympify(re.sub(r'(\d)([a-zA-Z])', r'\1*\2', parts[1].strip()))
            solutions = sympy.solve(left - right, var)
            if solutions:
                sol_str = ", ".join(str(s) for s in solutions)
                return {"calc_result": sol_str, "expression": expr, "message": f"Giải phương trình {expr}: {var_name} = {sol_str}"}
        except Exception:
            pass
        return None

    # Tính toán giá trị số học biểu thức bằng SymPy khi cần dự phòng.
    def _sympy_eval(self, expr: str) -> dict | None:
        try:
            import sympy
            res = float(sympy.sympify(expr).evalf())
            formatted = f"{res:.6g}"
            return {"calc_result": formatted, "expression": expr, "message": f"Kết quả (SymPy): {expr} = {formatted}"}
        except Exception:
            return None
