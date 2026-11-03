"""Thư viện tiện ích tính toán ma trận và đại số tuyến tính."""

def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Tính tích vô hướng của hai vector cùng chiều."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Hai vector phải có cùng số chiều!")
    return sum(a * b for a, b in zip(vec_a, vec_b))


def solve_linear_equation(a: float, b: float) -> float:
    """Giải phương trình bậc nhất: a * x + b = 0 => x = -b / a."""
    if a == 0:
        raise ValueError("Hệ số a phải khác 0!")
    return -b / a
