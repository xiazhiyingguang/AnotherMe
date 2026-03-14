"""
几何 IR（Intermediate Representation）定义与辅助函数。

该结构用于在 Vision -> Animation/FigureComposer 之间稳定传递题图信息，
避免直接依赖像素图片或自由生成的图形代码。

支持图形类型：线段、任意多边形、圆、向量、角度关系、平行/垂直关系等。
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

GEOMETRY_IR_SCHEMA_VERSION = "1.0"


def create_default_geometry_ir(image_meta: Dict[str, Any]) -> Dict[str, Any]:
    """创建默认几何 IR，作为识别失败时的安全回退。"""
    return {
        "schema_version": GEOMETRY_IR_SCHEMA_VERSION,
        "problem_type": "geometry",
        "source": {
            "original_image_path": image_meta.get("original_image_path", ""),
            "diagram_image_path": image_meta.get("diagram_image_path", ""),
            "diagram_bbox": image_meta.get("diagram_bbox", {}),
        },
        "canvas": {
            "coordinate_system": "manim_2d",
            "x_range": [-7, 7],
            "y_range": [-4, 4],
            "suggested_anchor": "LEFT",
        },
        "primitives": {
            "points": [],
            "segments": [],
            "angles": [],
            "polygons": [],
            "circles": [],
        },
        "relations": [],
        "render_hints": {
            "keep_problem_figure_visible": True,
            "preferred_layout": {
                "figure_region": "left",
                "text_region": "right",
                "diagram_anchor": "UL",
            },
            "use_cropped_diagram_as_fallback": True,
        },
    }


# ==========================================================================
# 基础向量 / 坐标工具
# ==========================================================================

def pt(x: float, y: float, z: float = 0.0) -> np.ndarray:
    """创建 Manim 兼容的 3D 坐标数组。"""
    return np.array([float(x), float(y), float(z)])


def to_pt(coord) -> np.ndarray:
    """将任意可迭代坐标（2D/3D 列表/元组）转换为 3D numpy 数组。"""
    c = list(coord)
    if len(c) == 2:
        return np.array([float(c[0]), float(c[1]), 0.0])
    return np.array([float(c[0]), float(c[1]), float(c[2])])


def vec(A, B) -> np.ndarray:
    """从 A 指向 B 的向量。"""
    return np.asarray(B, dtype=float) - np.asarray(A, dtype=float)


def dist(p1, p2) -> float:
    """计算两点间欧氏距离。"""
    return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)))


def midpoint(A, B) -> np.ndarray:
    """线段 AB 的中点。"""
    return (np.asarray(A, dtype=float) + np.asarray(B, dtype=float)) / 2.0


def unit_vec(A, B) -> np.ndarray:
    """从 A 指向 B 的单位向量；若 A == B 返回零向量。"""
    v = vec(A, B)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def angle_at(p_vertex, p1, p2) -> float:
    """以 p_vertex 为顶点，p1→顶点←p2 构成的角（弧度，[0, π]）。"""
    v1 = np.asarray(p1, dtype=float) - np.asarray(p_vertex, dtype=float)
    v2 = np.asarray(p2, dtype=float) - np.asarray(p_vertex, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def angle_between_lines(A, B, C, D) -> float:
    """直线 AB 与直线 CD 之间的锐角（弧度，[0, π/2]）。"""
    v1 = vec(A, B)
    v2 = vec(C, D)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_a = abs(np.dot(v1, v2) / (n1 * n2))
    return float(np.arccos(np.clip(cos_a, 0.0, 1.0)))


# ==========================================================================
# 基础几何关系判断
# ==========================================================================

def is_collinear(A, B, C, tol: float = 1e-6) -> bool:
    """判断三点是否共线（用叉积面积）。"""
    ab = vec(A, B)[:2]
    ac = vec(A, C)[:2]
    return abs(float(np.cross(ab, ac))) < tol


def are_parallel(A, B, C, D, tol: float = 1e-6) -> bool:
    """判断线段/直线 AB 与 CD 是否平行（叉积接近零）。"""
    v1 = vec(A, B)[:2]
    v2 = vec(C, D)[:2]
    return abs(float(np.cross(v1, v2))) < tol * (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)


def are_perpendicular(A, B, C, D, tol: float = 1e-6) -> bool:
    """判断线段/直线 AB 与 CD 是否垂直（点积接近零）。"""
    v1 = vec(A, B)
    v2 = vec(C, D)
    return abs(float(np.dot(v1, v2))) < tol * (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)


def foot_of_perpendicular(P, A, B) -> np.ndarray:
    """点 P 在直线 AB 上的投影（垂足）。"""
    ab = vec(A, B).astype(float)
    ap = vec(A, P).astype(float)
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-30)
    return np.asarray(A, dtype=float) + t * ab


def line_intersection(A, B, C, D) -> Optional[np.ndarray]:
    """
    求直线 AB 与直线 CD 的交点（2D）。
    若平行（含重合）则返回 None。
    """
    x1, y1 = float(A[0]), float(A[1])
    x2, y2 = float(B[0]), float(B[1])
    x3, y3 = float(C[0]), float(C[1])
    x4, y4 = float(D[0]), float(D[1])
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1), 0.0])


# ==========================================================================
# 面积 / 周长
# ==========================================================================

def polygon_area(pts_list) -> float:
    """
    Shoelace 公式计算任意简单多边形面积（顶点按序排列，可正可负）。
    返回绝对值。
    """
    pts = [np.asarray(p, dtype=float) for p in pts_list]
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def perimeter(pts_list) -> float:
    """计算多边形周长（按顺序首尾相连）。"""
    n = len(pts_list)
    return sum(dist(pts_list[i], pts_list[(i + 1) % n]) for i in range(n))


def polygon_interior_angles(pts_list) -> List[float]:
    """
    计算简单多边形各顶点处的内角（弧度）。
    顶点需按逆时针顺序给出；如为顺时针，返回结果相同（angle_at 取绝对值）。
    """
    pts = [np.asarray(p, dtype=float) for p in pts_list]
    n = len(pts)
    return [
        angle_at(pts[i], pts[(i - 1) % n], pts[(i + 1) % n])
        for i in range(n)
    ]


# ==========================================================================
# 圆
# ==========================================================================

def circumscribed_circle(A, B, C) -> Optional[Dict[str, Any]]:
    """
    求三角形 ABC 的外接圆（圆心 + 半径）。
    若三点共线返回 None。
    """
    ax, ay = float(A[0]), float(A[1])
    bx, by = float(B[0]), float(B[1])
    cx, cy = float(C[0]), float(C[1])
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-12:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D
    center = np.array([ux, uy, 0.0])
    radius = dist(center, np.array([ax, ay, 0.0]))
    return {"center": center.tolist(), "radius": round(radius, 6)}


def inscribed_circle(A, B, C) -> Dict[str, Any]:
    """
    求三角形 ABC 的内切圆（圆心 + 半径）。
    """
    a = dist(B, C)
    b = dist(A, C)
    c = dist(A, B)
    s_perim = a + b + c
    if s_perim < 1e-12:
        return {"center": [0, 0, 0], "radius": 0.0}
    Ap = np.asarray(A, dtype=float)
    Bp = np.asarray(B, dtype=float)
    Cp = np.asarray(C, dtype=float)
    center = (a * Ap + b * Bp + c * Cp) / s_perim
    area = polygon_area([A, B, C])
    radius = 2 * area / s_perim
    return {"center": center.tolist(), "radius": round(radius, 6)}


def circle_area(radius: float) -> float:
    return float(np.pi * radius ** 2)


def arc_length(radius: float, angle_rad: float) -> float:
    """弧长。"""
    return float(radius * angle_rad)


def sector_area(radius: float, angle_rad: float) -> float:
    """扇形面积。"""
    return float(0.5 * radius ** 2 * angle_rad)


# ==========================================================================
# 通用多边形 / 线段 IR 构建
# ==========================================================================

def build_polygon_ir(pts_dict: Dict[str, Any], labels: Dict[str, str] = None) -> Dict[str, Any]:
    """
    从有序顶点坐标字典构建任意多边形几何 IR（派生量全部用 numpy 计算）。

    Args:
        pts_dict: 顶点名称 → 坐标，按多边形顶点顺序排列，例如
                  {"A": [0,3], "B": [-2,0], "C": [2,0]}        三角形
                  {"A": [0,2], "B": [-2,0], "C": [0,-2], "D": [2,0]}  四边形
        labels:   dict，可选标注，键格式：
                  "side_AB" / "angle_A" 等

    Returns:
        包含 points / segments / angles / checks 的结构化字典。
    """
    labels = labels or {}
    names = list(pts_dict.keys())
    n = len(names)
    np_pts = {name: to_pt(coords) for name, coords in pts_dict.items()}
    pts_seq = [np_pts[name] for name in names]

    # ---- 边（按顺序相邻，不连所有组合）---- #
    segments = []
    for i in range(n):
        na, nb = names[i], names[(i + 1) % n]
        length = dist(np_pts[na], np_pts[nb])
        label_key = f"side_{''.join(sorted([na, nb]))}"
        segments.append({
            "name": f"{na}{nb}",
            "from": na,
            "to": nb,
            "length": round(length, 6),
            "label": labels.get(label_key, ""),
        })

    # ---- 内角 ---- #
    angle_rads = polygon_interior_angles(pts_seq)
    angles = []
    for i, rad in enumerate(angle_rads):
        name = names[i]
        prev_name = names[(i - 1) % n]
        next_name = names[(i + 1) % n]
        deg = round(float(np.degrees(rad)), 4)
        angles.append({
            "vertex": name,
            "from": prev_name,
            "to": next_name,
            "radians": round(rad, 6),
            "degrees": deg,
            "label": labels.get(f"angle_{name}", ""),
        })

    # ---- 面积 / 周长 / 合法性 ---- #
    area = polygon_area(pts_seq)
    peri = perimeter(pts_seq)
    angle_sum = sum(a["degrees"] for a in angles)
    expected_angle_sum = (n - 2) * 180.0

    checks = {
        "vertex_count": n,
        "area": round(area, 6),
        "perimeter": round(peri, 6),
        "angle_sum_degrees": round(angle_sum, 2),
        "expected_angle_sum": round(expected_angle_sum, 2),
        "angle_sum_valid": abs(angle_sum - expected_angle_sum) < 1.0,
        "is_degenerate": area < 1e-6,
    }

    return {
        "points": {n: np_pts[n].tolist() for n in names},
        "segments": segments,
        "angles": angles,
        "checks": checks,
    }


def build_segment_ir(name_A: str, coord_A, name_B: str, coord_B,
                     label: str = "") -> Dict[str, Any]:
    """构建单条线段的派生信息（长度、中点、方向角）。"""
    A = to_pt(coord_A)
    B = to_pt(coord_B)
    length = dist(A, B)
    mid = midpoint(A, B)
    direction_deg = float(np.degrees(np.arctan2(float(B[1] - A[1]), float(B[0] - A[0]))))
    return {
        "name": f"{name_A}{name_B}",
        "from": name_A,
        "to": name_B,
        "coord_from": A.tolist(),
        "coord_to": B.tolist(),
        "length": round(length, 6),
        "midpoint": mid.tolist(),
        "direction_deg": round(direction_deg, 4),
        "label": label,
    }


def build_circle_ir(center, radius: float, label: str = "") -> Dict[str, Any]:
    """构建圆的派生信息（面积、周长等）。"""
    c = to_pt(center) if not isinstance(center, np.ndarray) else center
    r = float(radius)
    return {
        "center": c.tolist(),
        "radius": round(r, 6),
        "area": round(circle_area(r), 6),
        "circumference": round(2 * np.pi * r, 6),
        "label": label,
    }

def normalize_geometry_ir(raw_ir: Dict[str, Any], image_meta: Dict[str, Any]) -> Dict[str, Any]:
    """对视觉模型返回的 IR 做字段补齐，保证下游稳定可用。"""
    ir = create_default_geometry_ir(image_meta)
    if not isinstance(raw_ir, dict):
        return ir

    for key in ["schema_version", "problem_type", "relations"]:
        if key in raw_ir:
            ir[key] = raw_ir[key]

    if isinstance(raw_ir.get("source"), dict):
        ir["source"].update(raw_ir["source"])
    if isinstance(raw_ir.get("canvas"), dict):
        ir["canvas"].update(raw_ir["canvas"])
    if isinstance(raw_ir.get("primitives"), dict):
        for name in ir["primitives"]:
            value = raw_ir["primitives"].get(name)
            if isinstance(value, list):
                ir["primitives"][name] = value
    if isinstance(raw_ir.get("render_hints"), dict):
        preferred_layout = raw_ir["render_hints"].get("preferred_layout")
        if isinstance(preferred_layout, dict):
            ir["render_hints"]["preferred_layout"].update(preferred_layout)
        for name, value in raw_ir["render_hints"].items():
            if name != "preferred_layout":
                ir["render_hints"][name] = value

    if not ir["source"].get("diagram_image_path"):
        ir["source"]["diagram_image_path"] = image_meta.get("diagram_image_path", "")
    if not ir["source"].get("original_image_path"):
        ir["source"]["original_image_path"] = image_meta.get("original_image_path", "")
    if not ir["source"].get("diagram_bbox"):
        ir["source"]["diagram_bbox"] = image_meta.get("diagram_bbox", {})

    return ir
