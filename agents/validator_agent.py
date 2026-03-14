"""
验证 Agent - 负责校验中间结果结构与基础完整性。
"""

from .base_agent import BaseAgent
from .geometry_ir import (
    build_polygon_ir, build_segment_ir, build_circle_ir,
    to_pt, dist, are_parallel, are_perpendicular,
    circumscribed_circle, foot_of_perpendicular, polygon_area,
)
from typing import Any, Dict, List, Optional
from config import DEFAULT_LLM_CONFIG
import numpy as np


class ValidatorAgent(BaseAgent):
    """验证 Agent（当前为基础实现，后续可扩展更严格规则）。"""

    def __init__(self, config: Dict[str, Any], llm: Optional[Any] = None):
        if not llm:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=DEFAULT_LLM_CONFIG["api_key"],
                base_url=DEFAULT_LLM_CONFIG["base_url"],
                model=DEFAULT_LLM_CONFIG["model"],
                temperature=config.get("temperature", 0.05),
                max_tokens=config.get("max_tokens", 2048),
            )

        super().__init__(config, llm)

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        对输入做基础结构验证。

        Returns:
            { "is_valid": bool, "issues": list[str], "input": Any }
        """
        issues = []
        if input_data is None:
            issues.append("input_data 为空")
        if isinstance(input_data, dict) and "error" in input_data:
            issues.append(f"检测到错误字段: {input_data.get('error')}")
        return {"is_valid": len(issues) == 0, "issues": issues, "input": input_data}

    # ------------------------------------------------------------------ #
    #  通用几何 IR 校验入口                                                 #
    # ------------------------------------------------------------------ #

    def validate_geometry_ir(self, geometry_ir: Dict[str, Any]) -> Dict[str, Any]:
        """
        对几何 IR 中的坐标与关系做 numpy 断言校验。
        支持点、线段、任意多边形（n 边形）、圆、平行/垂直关系。

        Returns:
            {
              "is_valid": bool,
              "warnings": list[str],   # 严重程度不足以阻断流程
              "errors":   list[str],   # 会阻断流程的错误
              "derived":  dict,        # numpy 计算的派生量
            }
        """
        warnings: List[str] = []
        errors: List[str] = []
        derived: Dict[str, Any] = {}

        primitives = geometry_ir.get("primitives", {})

        # ---- 1. 解析 points ---- #
        np_pts = self._parse_points(primitives.get("points", []), errors)
        derived["points_count"] = len(np_pts)

        # ---- 2. 检查重合点 ---- #
        self._check_coincident_points(np_pts, errors)

        # ---- 3. 校验线段 ---- #
        derived["segments"] = self._validate_segments(
            primitives.get("segments", []), np_pts, warnings
        )

        # ---- 4. 校验任意多边形（通用 n 边形） ---- #
        derived["polygons"] = self._validate_polygons(
            primitives.get("polygons", []), np_pts, warnings, errors
        )

        # ---- 5. 校验圆 ---- #
        derived["circles"] = self._validate_circles(
            primitives.get("circles", []), np_pts, warnings, errors
        )

        # ---- 6. 校验 relations（平行/垂直/共线等声明） ---- #
        self._validate_relations(
            geometry_ir.get("relations", []), np_pts, warnings
        )

        # ---- 7. 校验题意完整性（点覆盖、折叠映射） ---- #
        self._validate_semantic_completeness(
            geometry_ir=geometry_ir,
            np_pts=np_pts,
            warnings=warnings,
            errors=errors,
        )

        return {
            "is_valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
            "derived": derived,
        }

    # ------------------------------------------------------------------ #
    #  内部辅助方法                                                         #
    # ------------------------------------------------------------------ #

    def _parse_points(self, raw_points, errors: List[str]) -> Dict[str, np.ndarray]:
        """
        解析 points 字段，支持两种格式：
        - dict: {"A": [x,y], ...}
        - list: [{"name":"A", "coord":[x,y]}, ...]
        """
        np_pts: Dict[str, np.ndarray] = {}
        if isinstance(raw_points, dict):
            items = raw_points.items()
        elif isinstance(raw_points, list):
            items = []
            for item in raw_points:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("label") or item.get("id")
                coord = (item.get("coord") or item.get("position")
                         or item.get("coordinates") or item.get("location"))
                if name and coord:
                    items.append((name, coord))
        else:
            return np_pts

        for name, coord in items:
            try:
                np_pts[name] = to_pt(coord)
            except Exception as exc:
                errors.append(f"点 {name} 坐标解析失败: {exc}")
        return np_pts

    def _check_coincident_points(self, np_pts: Dict[str, np.ndarray], errors: List[str]):
        names = list(np_pts.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                na, nb = names[i], names[j]
                if dist(np_pts[na], np_pts[nb]) < 1e-6:
                    errors.append(f"点 {na} 与 {nb} 重合，坐标无效")

    def _validate_segments(self, segments, np_pts, warnings) -> List[Dict]:
        result = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            p_from = seg.get("from") or seg.get("start")
            p_to = seg.get("to") or seg.get("end")
            if not (p_from and p_to and p_from in np_pts and p_to in np_pts):
                continue
            ir = build_segment_ir(
                p_from, np_pts[p_from], p_to, np_pts[p_to],
                label=seg.get("label", "")
            )
            if ir["length"] < 1e-6:
                warnings.append(f"线段 {p_from}{p_to} 长度接近零")
            result.append(ir)
        return result

    def _validate_polygons(self, polygons, np_pts, warnings, errors) -> List[Dict]:
        """
        校验任意多边形。若 primitives 中没有 polygons 但恰好有 3 个点，
        则自动视为三角形。
        """
        result = []

        # 自动识别三角形
        if not polygons and len(np_pts) == 3:
            polygons = [{"type": "triangle", "vertices": list(np_pts.keys())}]

        for poly in polygons:
            if not isinstance(poly, dict):
                continue
            verts = poly.get("vertices", [])
            n = len(verts)
            if n < 3:
                warnings.append(f"多边形顶点数 {n} < 3，跳过")
                continue

            missing = [v for v in verts if v not in np_pts]
            if missing:
                errors.append(f"多边形顶点 {missing} 缺少坐标，跳过")
                continue

            pts_dict = {v: np_pts[v].tolist() for v in verts}
            poly_ir = build_polygon_ir(pts_dict)

            checks = poly_ir["checks"]
            # 内角和检查（对所有多边形通用）
            if not checks["angle_sum_valid"]:
                warnings.append(
                    f"{''.join(verts)} 内角和 = {checks['angle_sum_degrees']}°，"
                    f"期望 {checks['expected_angle_sum']}°（{n} 边形），坐标可能有误"
                )
            if checks["is_degenerate"]:
                errors.append(f"{''.join(verts)} 面积接近零，图形退化")

            result.append({
                "vertices": verts,
                "type": poly.get("type", f"{n}gon"),
                **poly_ir,
            })
        return result

    def _validate_circles(self, circles, np_pts, warnings, errors) -> List[Dict]:
        result = []
        for circ in circles:
            if not isinstance(circ, dict):
                continue

            # 支持 center 用坐标直接给，也支持用点名引用
            center_raw = circ.get("center")
            radius = circ.get("radius")

            if isinstance(center_raw, str) and center_raw in np_pts:
                center = np_pts[center_raw]
            elif center_raw is not None:
                try:
                    center = to_pt(center_raw)
                except Exception:
                    warnings.append(f"圆心坐标解析失败: {center_raw}")
                    continue
            else:
                warnings.append("圆缺少 center 字段，跳过")
                continue

            # 三点定圆
            if radius is None:
                on_circle = circ.get("on_circle", [])
                if len(on_circle) >= 3:
                    pts_on = [np_pts[n] for n in on_circle[:3] if n in np_pts]
                    if len(pts_on) == 3:
                        cc = circumscribed_circle(*pts_on)
                        if cc:
                            radius = cc["radius"]
                        else:
                            errors.append(f"圆上三点 {on_circle[:3]} 共线，无法确定圆")
                            continue

            if radius is None or float(radius) <= 0:
                warnings.append(f"圆半径无效: {radius}")
                continue

            ir = build_circle_ir(center, float(radius), label=circ.get("label", ""))
            result.append(ir)
        return result

    def _validate_relations(self, relations, np_pts, warnings):
        """
        校验 relations 中声明的几何关系（平行、垂直、共线等）是否与坐标一致。
        relations 中每条记录格式示例：
          {"type": "parallel", "elements": ["AB", "CD"]}
          {"type": "perpendicular", "elements": ["AB", "CD"]}
          {"type": "collinear", "elements": ["A", "B", "C"]}
        """
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            rel_type = rel.get("type", "").lower()
            elements = rel.get("elements", [])

            try:
                if rel_type == "parallel" and len(elements) == 2:
                    seg1, seg2 = elements
                    A, B = self._seg_endpoints(seg1, np_pts)
                    C, D = self._seg_endpoints(seg2, np_pts)
                    if A is not None and not are_parallel(A, B, C, D):
                        warnings.append(
                            f"声明 {seg1} ∥ {seg2}，但坐标计算不满足平行条件"
                        )

                elif rel_type == "perpendicular" and len(elements) == 2:
                    seg1, seg2 = elements
                    A, B = self._seg_endpoints(seg1, np_pts)
                    C, D = self._seg_endpoints(seg2, np_pts)
                    if A is not None and not are_perpendicular(A, B, C, D):
                        warnings.append(
                            f"声明 {seg1} ⊥ {seg2}，但坐标计算不满足垂直条件"
                        )

                elif rel_type == "collinear" and len(elements) >= 3:
                    pts = [np_pts[e] for e in elements if e in np_pts]
                    if len(pts) >= 3:
                        # 用叉积检查连续三点
                        for i in range(len(pts) - 2):
                            ab = (pts[i + 1] - pts[i])[:2]
                            ac = (pts[i + 2] - pts[i])[:2]
                            if abs(float(np.cross(ab, ac))) > 1e-4:
                                warnings.append(
                                    f"声明 {elements} 共线，但点 {elements[i]}、"
                                    f"{elements[i+1]}、{elements[i+2]} 不共线"
                                )
                                break

            except Exception as exc:
                warnings.append(f"关系 {rel} 校验异常: {exc}")

    def _seg_endpoints(self, seg_name: str, np_pts) -> tuple:
        """
        从线段名称（如 'AB'）提取两端点坐标。
        若点不存在返回 (None, None)。
        """
        if len(seg_name) >= 2:
            na, nb = seg_name[0], seg_name[1]
            if na in np_pts and nb in np_pts:
                return np_pts[na], np_pts[nb]
        return None, None

    def _normalize_point_name(self, name: str) -> str:
        if not isinstance(name, str):
            return ""
        normalized = name.strip().upper()
        normalized = normalized.replace("’", "'").replace("′", "'")
        return normalized

    def _validate_semantic_completeness(
        self,
        geometry_ir: Dict[str, Any],
        np_pts: Dict[str, np.ndarray],
        warnings: List[str],
        errors: List[str],
    ) -> None:
        """根据 semantic_hints 检查题意关键实体是否在 IR 中覆盖。"""
        hints = geometry_ir.get("semantic_hints", {})
        if not isinstance(hints, dict):
            return

        required_points_raw = hints.get("required_points", [])
        if isinstance(required_points_raw, list) and required_points_raw:
            required_points = {
                self._normalize_point_name(p)
                for p in required_points_raw
                if self._normalize_point_name(p)
            }
            present_points = {
                self._normalize_point_name(name)
                for name in np_pts.keys()
                if self._normalize_point_name(name)
            }
            missing = sorted(required_points - present_points)
            if missing:
                errors.append(f"题干关键点未覆盖: {missing}")

        fold_detected = bool(hints.get("fold_detected"))
        if not fold_detected:
            return

        relations = geometry_ir.get("relations", [])
        has_fold_relation = False
        if isinstance(relations, list):
            for rel in relations:
                if not isinstance(rel, dict):
                    continue
                rel_type = str(rel.get("type", "")).lower()
                if rel_type in {"fold", "reflection", "mirror", "symmetry", "fold_map"}:
                    has_fold_relation = True
                    break
                if any(k in rel for k in ["fold_axis", "mapped_from", "mapped_to"]):
                    has_fold_relation = True
                    break

        if not has_fold_relation:
            warnings.append("题干包含折叠/映射语义，但 relations 中未找到明确折叠映射描述")

