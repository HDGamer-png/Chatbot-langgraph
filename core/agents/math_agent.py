"""
core/agents/math_agent.py — CalculatorAgent
Tính toán an toàn bằng SymPy & Python AST. Không cần API Key.

Hỗ trợ:
  - Biểu thức số học: cộng, trừ, nhân, chia, lũy thừa, căn bậc
  - Giải phương trình: 2x + 5 = 15
  - Hàm lượng giác: sin, cos, tan
  - Phần trăm, giai thừa, logarit
"""

import re
import ast
import math
import operator
from .base import AgentBase


# Các toán tử cho phép trong AST eval
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

# Hàm toán học cho phép
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


def _safe_eval(expr_str: str) -> float:
    """
    Đánh giá biểu thức toán học AN TOÀN bằng AST.
    Chỉ cho phép: số, toán tử +−*/**, ngoặc (), và hàm toán học cơ bản.
    CHẶN hoàn toàn mọi code injection.
    """
    expr_str = expr_str.strip()
    if not expr_str:
        raise ValueError("Biểu thức rỗng.")

    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Cú pháp không hợp lệ: {e}")

    def _eval_node(node):
        # Số
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        # Toán tử đơn (-x, +x)
        if isinstance(node, ast.UnaryOp):
            op = _SAFE_OPERATORS.get(type(node.op))
            if op:
                return op(_eval_node(node.operand))
        # Toán tử hai ngôi (a + b, a * b, ...)
        if isinstance(node, ast.BinOp):
            op = _SAFE_OPERATORS.get(type(node.op))
            if op:
                return op(_eval_node(node.left), _eval_node(node.right))
        # Gọi hàm (sqrt(x), sin(x), ...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _SAFE_FUNCTIONS.get(node.func.id)
            if fn:
                args = [_eval_node(arg) for arg in node.args]
                return fn(*args)
        # Tên biến (pi, e)
        if isinstance(node, ast.Name) and node.id in _SAFE_FUNCTIONS:
            val = _SAFE_FUNCTIONS[node.id]
            if isinstance(val, (int, float)):
                return val
        # Expression wrapper
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)

        raise ValueError(f"Thành phần không được phép: {ast.dump(node)}")

    return _eval_node(tree)


class CalculatorAgent(AgentBase):
    """
    Tác tử tính toán — Giải toán bằng SymPy và Python AST.
    Chính xác 100%, an toàn (chống code injection), không cần API.
    """

    NAME = "Calculator"

    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)

    def run(self, query: str = "", **kwargs) -> dict:
        """
        Phân tích câu hỏi, trích xuất biểu thức toán và tính kết quả.

        Returns:
            {"calc_result": str, "expression": str, "message": str}
        """
        query = query.strip()
        if not query:
            return {"calc_result": "", "expression": "", "message": "Không có biểu thức."}

        # Trích xuất biểu thức toán từ câu hỏi tiếng Việt
        expr = self._extract_expression(query)
        if not expr:
            return {
                "calc_result": "",
                "expression": "",
                "message": "Không tìm thấy biểu thức toán trong câu hỏi.",
            }

        # Thử giải phương trình trước (nếu có dấu =)
        if "=" in expr and re.search(r'[a-zA-Z]', expr):
            result = self._solve_equation(expr)
            if result:
                return result

        # Tính biểu thức số học
        try:
            value = _safe_eval(expr)
            # Định dạng kết quả đẹp
            if isinstance(value, float) and value == int(value):
                formatted = str(int(value))
            elif isinstance(value, float):
                formatted = f"{value:.6g}"
            else:
                formatted = str(value)

            return {
                "calc_result": formatted,
                "expression": expr,
                "message": f"Kết quả: {expr} = {formatted}",
            }
        except Exception as e:
            # Thử dùng SymPy nếu AST eval thất bại
            result = self._sympy_eval(expr)
            if result:
                return result
            return {
                "calc_result": "",
                "expression": expr,
                "message": f"Không thể tính: {e}",
            }

    def _extract_expression(self, query: str) -> str:
        """Trích xuất biểu thức toán từ câu hỏi tiếng Việt."""
        q = query.lower()

        # Xóa các từ phụ tiếng Việt và ký tự đặc biệt
        q = re.sub(r'\b(giải|phương|trình|tìm|nghiệm|tính|giá|trị|biểu|thức|toán|của|là|bằng|cho|tôi|hãy|hỏi|số|kết quả)\b', ' ', q)
        q = q.replace(':', ' ').replace('?', ' ')

        # Thay thế từ tiếng Việt thành ký hiệu toán
        replacements = [
            (r'\bcộng\b', '+'), (r'\btrừ\b', '-'),
            (r'\bnhân\b', '*'), (r'\bchia\b', '/'),
            (r'\bmũ\b', '**'), (r'\blũy thừa\b', '**'),
            (r'\bbình phương\b', '**2'), (r'\blập phương\b', '**3'),
            (r'\bcăn bậc hai\b', 'sqrt'), (r'\bcăn bậc 2\b', 'sqrt'), (r'\bcăn\b', 'sqrt'),
            (r'\bphần trăm\b', '/100'), (r'\b%\b', '/100'),
            (r'\bsin\b', 'sin'), (r'\bcos\b', 'cos'), (r'\btan\b', 'tan'),
            (r'\bgiai thừa\b', 'factorial'),
        ]
        for pattern, repl in replacements:
            q = re.sub(pattern, repl, q)

        # Định dạng hàm có ngoặc: sqrt 144 -> sqrt(144)
        q = re.sub(r'\b(sqrt|sin|cos|tan|log|factorial)\s+(\d+(?:\.\d+)?)', r'\1(\2)', q)

        # Kiểm tra phương trình (có dấu =)
        if '=' in q:
            eq_match = re.search(r'([0-9a-zA-Z\s\+\-\*\/\^\(\)\.]+\s*=\s*[0-9a-zA-Z\s\+\-\*\/\^\(\)\.]+)', q)
            if eq_match:
                candidate = eq_match.group(1).strip()
                tokens = candidate.split()
                clean_tokens = [t for t in tokens if re.match(r'^[0-9a-zA-Z\+\-\*\/\^\(\)\.\=]+$', t) and (len(t) <= 3 or any(c in t for c in "+-*/^=0123456789"))]
                candidate = " ".join(clean_tokens)
                if '=' in candidate and re.search(r'[a-zA-Z]', candidate) and re.search(r'\d', candidate):
                    return candidate

        # Tìm biểu thức có hàm toán học (sqrt, sin, cos...) hoặc số học
        func_match = re.search(r'((?:sqrt|sin|cos|tan|log|factorial)\s*\([^)]+\))', q)
        if func_match:
            return func_match.group(1).replace(' ', '')

        # Tìm biểu thức số học thông thường
        math_match = re.search(
            r'([\d\.\s\+\-\*\/\%\^\(\)]+)',
            q
        )
        if math_match:
            expr = math_match.group(1).strip()
            expr = re.sub(r'\s+', '', expr)
            if re.search(r'\d', expr) and len(expr) > 1:
                return expr

        return ""

    def _solve_equation(self, expr: str) -> dict | None:
        """Giải phương trình bằng SymPy (ví dụ: 2x + 5 = 15)."""
        try:
            import sympy
        except ImportError:
            return None

        try:
            # Tìm biến (chữ cái đơn trong biểu thức)
            variables = set(re.findall(r'[a-zA-Z]', expr))
            variables -= {'e', 'E'}  # loại bỏ hằng số e
            if not variables:
                return None

            var_name = variables.pop()
            var = sympy.Symbol(var_name)

            # Phân tách phương trình qua dấu =
            parts = expr.split('=')
            if len(parts) != 2:
                return None

            # Chèn dấu nhân ngầm: 2x -> 2*x
            left = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', parts[0].strip())
            right = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', parts[1].strip())

            left_expr = sympy.sympify(left)
            right_expr = sympy.sympify(right)

            solutions = sympy.solve(left_expr - right_expr, var)

            if solutions:
                sol_str = ", ".join(str(s) for s in solutions)
                return {
                    "calc_result": sol_str,
                    "expression": expr,
                    "message": f"Giải phương trình {expr}: {var_name} = {sol_str}",
                }
        except Exception:
            pass
        return None

    def _sympy_eval(self, expr: str) -> dict | None:
        """Thử tính biểu thức bằng SymPy khi AST eval thất bại."""
        try:
            import sympy
            result = sympy.sympify(expr)
            evaluated = float(result.evalf())
            formatted = f"{evaluated:.6g}"
            return {
                "calc_result": formatted,
                "expression": expr,
                "message": f"Kết quả (SymPy): {expr} = {formatted}",
            }
        except Exception:
            return None

