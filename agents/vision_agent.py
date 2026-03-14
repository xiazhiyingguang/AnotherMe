"""
视觉 Agent - 负责 OCR 识别和理解题目图片
"""

from .base_agent import BaseAgent
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from .geometry_ir import create_default_geometry_ir, normalize_geometry_ir
import json
import re
import base64
import os
from urllib.request import urlretrieve

class VisionAgent(BaseAgent):
    """视觉 Agent - 负责 OCR 识别和理解题目图片"""

    def __init__(self, config: Dict[str, Any], llm: Optional[Any] = None):
        """
        初始化 VisionAgent

        Args:
            config: VisionAgent 的配置字典
            llm: 可选的 LLM 实例，如果不传则内部创建
        """
        if not llm:
            from config import VISION_MODEL_CONFIG
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=VISION_MODEL_CONFIG["api_key"],
                base_url=VISION_MODEL_CONFIG["base_url"],
                model=VISION_MODEL_CONFIG["model"],
                temperature=config.get("temperature", 0.05),
                max_tokens=config.get("max_tokens", 2048),
            )

        super().__init__(config, llm)


    def process(self, input_data: str) -> Dict[str, Any]:
        """
        处理输入的题目图片 URL，返回 OCR、建模、图形元信息与几何 IR。

        Args:
            input_data: 题目图片的 URL 或本地文件路径

        Returns:
            包含 OCR 结果、数学建模、图像元信息和几何 IR 的字典
        """
        local_image_path = self._prepare_local_image(input_data)
        diagram_info = self.extract_diagram_region(local_image_path)
        ocr_result = self.ocr_recognize(input_data) # 识别题目
        modeling_result = self.model_problem(diagram_info.get("diagram_image_path", local_image_path)) # 数学建模

        image_meta = {
            "original_image_path": local_image_path,
            "diagram_image_path": diagram_info.get("diagram_image_path", local_image_path),
            "diagram_bbox": diagram_info.get("diagram_bbox", {}),
        }
        geometry_ir = self.build_geometry_ir(ocr_result, modeling_result, image_meta)
        geometry_ir = self.enrich_geometry_ir_with_vision_points(
            geometry_ir=geometry_ir,
            diagram_image_path=image_meta.get("diagram_image_path", local_image_path),
            ocr_result=ocr_result,
            modeling_result=modeling_result,
        )
        required_points = sorted(self._extract_required_point_names(ocr_result, modeling_result))
        geometry_ir.setdefault("semantic_hints", {})
        geometry_ir["semantic_hints"]["required_points"] = required_points
        geometry_ir["semantic_hints"]["fold_detected"] = self._text_mentions_fold(ocr_result, modeling_result)

        # 用 ValidatorAgent 计算 numpy 派生量，并回写到 IR 的 derived 字段
        try:
            from .validator_agent import ValidatorAgent
            validator = ValidatorAgent(config={})
            validation = validator.validate_geometry_ir(geometry_ir)
            geometry_ir["derived"] = validation.get("derived", {})
            geometry_ir["validation"] = {
                "is_valid": validation["is_valid"],
                "warnings": validation["warnings"],
                "errors": validation["errors"],
            }
        except Exception as exc:
            geometry_ir["validation"] = {"is_valid": True, "warnings": [], "errors": [str(exc)]}

        geometry_ir = self._to_jsonable(geometry_ir)
        problem_figure_code = self.generate_problem_figure_code(geometry_ir, image_meta)
        code_paths = self.save_problem_figure_code(problem_figure_code)

        return {
            "ocr_result": ocr_result,
            "modeling_result": modeling_result,
            "image_meta": image_meta,
            "geometry_ir": geometry_ir,
            "problem_figure_code": problem_figure_code,
            "problem_figure_code_path": code_paths["snippet_path"],
            "problem_figure_preview_path": code_paths["preview_path"],
        }

    def save_problem_figure_code(self, code: str) -> Dict[str, str]:
        """将 Vision 直出题图代码落盘，便于独立检查和预览渲染。"""
        output_dir = self.config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)

        snippet_path = os.path.join(output_dir, "problem_figure_snippet.py")
        preview_path = os.path.join(output_dir, "problem_figure_preview.py")

        sanitized_code = code.strip() + "\n"
        sanitized_code = re.sub(r"^```(?:python)?\s*", "", sanitized_code, flags=re.IGNORECASE)
        sanitized_code = re.sub(r"\s*```\s*$", "", sanitized_code)

        with open(snippet_path, "w", encoding="utf-8") as f:
            f.write(sanitized_code)

        preview_code = (
            "from manim import *\n\n"
            "class ProblemFigurePreview(Scene):\n"
            "    def construct(self):\n"
            "        self.camera.background_color = BLACK\n"
            + self._indent_code(sanitized_code, 8)
            + "\n"
            + "        self.wait(2)\n"
        )
        with open(preview_path, "w", encoding="utf-8") as f:
            f.write(preview_code)

        return {
            "snippet_path": snippet_path,
            "preview_path": preview_path,
        }

    def generate_problem_figure_code(self, geometry_ir: Dict[str, Any], image_meta: Dict[str, Any]) -> str:
        """
        直接生成可插入 Manim construct(self) 的题图代码。

        这是优先路径；若模型生成失败或质量不佳，下游可回退到 geometry_ir 方案。
        """
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                你是 Manim 题图代码生成器。

                请输出“可直接插入 construct(self) 内部”的 Python 代码片段，用于创建题图基础层。
                约束：
                1. 必须定义 `problem_group = VGroup(...)`，并 `self.add(problem_group)`。
                2. 可选地在左半屏补充结构化几何对象，但禁止创建多余辅助线和坐标系。
                3. 禁止输出 import、class、render 逻辑。
                4. 代码应尽量简洁稳定，避免复杂坐标魔法。
                5. 只输出代码，不要解释。
                """
            ),
            (
                "human",
                """
                geometry_ir:
                {geometry_ir}

                image_meta:
                {image_meta}

                请生成题图代码片段。
                """
            )
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "geometry_ir": json.dumps(geometry_ir, ensure_ascii=False, indent=2),
            "image_meta": json.dumps(image_meta, ensure_ascii=False, indent=2),
        })
        code = response.content.strip() + "\n"

        # 基础守卫：若没有生成 problem_group，回退到最小可用代码
        if "problem_group" not in code:
            diagram_path = image_meta.get("diagram_image_path", image_meta.get("original_image_path", ""))
            normalized = os.path.abspath(diagram_path).replace("\\", "/") if diagram_path else ""
            return (
                f'problem_img = ImageMobject(r"{normalized}").scale_to_fit_height(2.2).to_corner(UL, buff=0.25)\n'
                'problem_group = VGroup(problem_img)\n'
                'self.add(problem_group)\n'
            )
        return code

    def build_geometry_ir(
        self,
        ocr_result: str,
        modeling_result: Dict[str, Any],
        image_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        基于 OCR、数学建模和图片元信息生成几何 IR，供后续动画编排使用。

        Args:
            ocr_result: OCR 文本结果
            modeling_result: 数学建模结果
            image_meta: 图片元信息

        Returns:
            标准化的几何 IR 字典
        """
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                你是一个几何图形结构分析器，负责把数学题中的图形信息抽取为结构化几何 IR。
                                请严格输出 JSON，顶层字段必须包含：
                                - schema_version: 固定为 "1.0"
                                - problem_type: 固定为 "geometry"
                                - source: 包含 original_image_path, diagram_image_path, diagram_bbox
                                - canvas: 包含 coordinate_system, x_range, y_range, suggested_anchor
                                - primitives: 包含 points, segments, angles, polygons, labels（全部是数组）
                                - relations: 数组
                                - render_hints: 包含 keep_problem_figure_visible, preferred_layout, use_cropped_diagram_as_fallback

                                其中 diagram_bbox 必须包含 x1, y1, x2, y2 四个数字字段。

                要求：
                1. 若无法可靠恢复精确坐标，不要伪造坐标，只输出图元关系。
                2. points / segments / angles / polygons / labels 都必须是数组。
                3. 所有字段都必须保留，缺失时填空数组或默认值。
                4. 必须优先覆盖题干和建模中出现的所有命名点，有几个就要识别几个
                5. 若题干出现折叠、对称、镜像、旋转、平移等变换，请在 relations 中显式表达（例如 fold_axis, mapped_from, mapped_to）。
                5. 只输出 JSON，不要解释。
                """
            ),
            (
                "human",
                """
                OCR 结果：
                {ocr_result}

                数学建模结果：
                {modeling_result}

                图片元信息：
                {image_meta}

                请输出几何 IR JSON。
                """
            )
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "ocr_result": ocr_result,
            "modeling_result": json.dumps(modeling_result, ensure_ascii=False, indent=2),
            "image_meta": json.dumps(image_meta, ensure_ascii=False, indent=2),
        })
        raw_ir = self.parse_json_response(response.content)
        return normalize_geometry_ir(raw_ir, image_meta)

    def enrich_geometry_ir_with_vision_points(
        self,
        geometry_ir: Dict[str, Any],
        diagram_image_path: str,
        ocr_result: str,
        modeling_result: str,
    ) -> Dict[str, Any]:
        """若 geometry_ir 未覆盖题干关键点，则从题图裁剪图强制补点并回写。"""
        primitives = geometry_ir.get("primitives", {})
        points = primitives.get("points", [])
        required_points = self._extract_required_point_names(ocr_result, modeling_result)
        existing_names = self._collect_point_names(points if isinstance(points, list) else [])
        has_required_coverage = required_points.issubset(existing_names) if required_points else False

        if isinstance(points, list) and len(points) >= 3 and has_required_coverage:
            return geometry_ir

        extracted = self.extract_geometry_primitives_from_image(
            diagram_image_path=diagram_image_path,
            ocr_result=ocr_result,
            modeling_result=modeling_result,
        )
        if not extracted:
            extracted = self.synthesize_geometry_primitives_from_text(ocr_result, modeling_result)
        if not extracted:
            return geometry_ir

        merged = dict(geometry_ir)
        merged_primitives = dict(merged.get("primitives", {}))
        merged_primitives["points"] = self._merge_point_lists(
            points if isinstance(points, list) else [],
            extracted.get("points", []),
        )
        merged_primitives["segments"] = self._merge_relation_like_list(
            primitives.get("segments", []) if isinstance(primitives, dict) else [],
            extracted.get("segments", []),
        )
        merged_primitives["polygons"] = self._merge_relation_like_list(
            primitives.get("polygons", []) if isinstance(primitives, dict) else [],
            extracted.get("polygons", []),
        )
        merged_primitives["angles"] = self._merge_relation_like_list(
            primitives.get("angles", []) if isinstance(primitives, dict) else [],
            extracted.get("angles", []),
        )
        merged["primitives"] = merged_primitives

        relations = extracted.get("relations", [])
        if isinstance(relations, list) and relations:
            merged["relations"] = self._merge_relation_like_list(
                geometry_ir.get("relations", []), relations
            )
        return merged

    def extract_geometry_primitives_from_image(
        self,
        diagram_image_path: str,
        ocr_result: str,
        modeling_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        对裁剪题图执行坐标提取，返回 points/segments/polygons/angles/relations。

        points 输入格式：{"name":"A","x":123,"y":456}，其中 x,y 为 [0,1000] 归一化图像坐标。
        输出时会映射到 manim_2d 坐标系。
        """
        if not diagram_image_path or not os.path.exists(diagram_image_path):
            return None

        try:
            with open(diagram_image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()
        except OSError:
            return None

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"},
                },
                {
                    "type": "text",
                    "text": (
                        "你是几何图结构化提取器。请只输出 JSON，不要解释。"
                        "返回格式："
                        "{"
                        "\"points\":[{\"name\":\"A\",\"x\":100,\"y\":200}],"
                        "\"segments\":[{\"from\":\"A\",\"to\":\"B\"}],"
                        "\"polygons\":[{\"type\":\"triangle\",\"vertices\":[\"A\",\"B\",\"C\"]}],"
                        "\"angles\":[{\"vertex\":\"C\",\"from\":\"A\",\"to\":\"B\",\"is_right\":true}],"
                        "\"relations\":[{\"type\":\"parallel\",\"elements\":[\"AB\",\"CD\"]}]"
                        "}"
                        "要求："
                        "1) x,y 为 0~1000 整数（图像归一化坐标）；"
                        "2) 必须尽量覆盖题干中出现的全部命名点（包括 D、E、F、C'、A' 这类带撇点）；"
                        "3) 点名必须与题干一致，不要强行改写成只有 A,B,C；"
                        "4) 无法确认的关系可省略，但 points 尽量完整。"
                        f"\n题干OCR: {ocr_result}"
                        f"\n建模摘要: {json.dumps(modeling_result, ensure_ascii=False)}"
                    ),
                },
            ]
        )

        response = self.llm.invoke([message])
        parsed = self.parse_json_response(response.content)
        if not isinstance(parsed, dict) or "error" in parsed:
            return None

        points_raw = parsed.get("points", [])
        if not isinstance(points_raw, list):
            return None

        points = []
        for item in points_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            x_raw = item.get("x")
            y_raw = item.get("y")
            if not name:
                continue
            try:
                x_norm = float(x_raw)
                y_norm = float(y_raw)
            except (TypeError, ValueError):
                continue
            if not (0 <= x_norm <= 1000 and 0 <= y_norm <= 1000):
                continue

            mx, my = self._normalized_to_manim_xy(x_norm, y_norm, x_range=(-7, 7), y_range=(-4, 4))
            points.append({
                "name": name,
                "coord": [round(mx, 4), round(my, 4), 0.0],
                "source": "vision_keypoint",
            })

        if len(points) < 3:
            return None

        segments = parsed.get("segments", []) if isinstance(parsed.get("segments", []), list) else []
        polygons = parsed.get("polygons", []) if isinstance(parsed.get("polygons", []), list) else []
        angles = parsed.get("angles", []) if isinstance(parsed.get("angles", []), list) else []
        relations = parsed.get("relations", []) if isinstance(parsed.get("relations", []), list) else []

        point_names = {p["name"] for p in points}
        if not polygons and {"A", "B", "C"}.issubset(point_names):
            polygons = [{"type": "triangle", "vertices": ["A", "B", "C"]}]

        return {
            "points": points,
            "segments": segments,
            "polygons": polygons,
            "angles": angles,
            "relations": relations,
        }

    def _normalize_point_name(self, name: str) -> str:
        """归一化点名，用于对比覆盖率（保留字母和撇号语义）。"""
        if not isinstance(name, str):
            return ""
        normalized = name.strip().upper()
        normalized = normalized.replace("’", "'").replace("′", "'")
        normalized = re.sub(r"\s+", "", normalized)
        return normalized

    def _extract_required_point_names(self, ocr_result: str, modeling_result: Dict[str, Any]) -> set:
        """从题干和建模文本提取应覆盖的命名点集合。"""
        text = f"{ocr_result}\n{json.dumps(modeling_result, ensure_ascii=False)}"
        text = text.replace("’", "'").replace("′", "'")
        matches = re.findall(r"\b([A-Z](?:')?)\b", text)
        return {self._normalize_point_name(m) for m in matches if self._normalize_point_name(m)}

    def _collect_point_names(self, points: List[Dict[str, Any]]) -> set:
        names = set()
        for item in points:
            if not isinstance(item, dict):
                continue
            normalized = self._normalize_point_name(str(item.get("name", "")))
            if normalized:
                names.add(normalized)
        return names

    def _merge_point_lists(self, base_points: List[Dict[str, Any]], new_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按点名去重合并点列表，优先保留新提取点坐标。"""
        merged: Dict[str, Dict[str, Any]] = {}
        for item in base_points:
            if not isinstance(item, dict):
                continue
            key = self._normalize_point_name(str(item.get("name", "")))
            if key:
                merged[key] = item
        for item in new_points:
            if not isinstance(item, dict):
                continue
            key = self._normalize_point_name(str(item.get("name", "")))
            if key:
                merged[key] = item
        return list(merged.values())

    def _merge_relation_like_list(self, base_items: Any, new_items: Any) -> List[Dict[str, Any]]:
        """合并 segments/polygons/angles/relations 这类列表并去重。"""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for source in [base_items, new_items]:
            if not isinstance(source, list):
                continue
            for item in source:
                if not isinstance(item, dict):
                    continue
                key = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged

    def _text_mentions_fold(self, ocr_result: str, modeling_result: Dict[str, Any]) -> bool:
        text = f"{ocr_result}\n{json.dumps(modeling_result, ensure_ascii=False)}"
        keywords = ["折叠", "翻折", "关于", "对称", "镜像", "映射", "落在"]
        return any(k in text for k in keywords)

    def _normalized_to_manim_xy(
        self,
        x_norm: float,
        y_norm: float,
        x_range=(-7, 7),
        y_range=(-4, 4),
    ):
        """将归一化图像坐标 (0~1000) 映射到 manim_2d 坐标。"""
        x_min, x_max = float(x_range[0]), float(x_range[1])
        y_min, y_max = float(y_range[0]), float(y_range[1])
        x = x_min + (x_norm / 1000.0) * (x_max - x_min)
        y = y_max - (y_norm / 1000.0) * (y_max - y_min)
        return x, y

    def synthesize_geometry_primitives_from_text(
        self,
        ocr_result: str,
        modeling_result: str,
    ) -> Optional[Dict[str, Any]]:
        """
        当视觉点提取失败时，从题干文本中合成最小可用坐标。
        目标是保证下游有 points/segments/polygons 可计算，而不是退回贴图。
        """
        text = f"{ocr_result}\n{json.dumps(modeling_result, ensure_ascii=False)}"

        # 解析边长：如 AC=8, BC = 6
        side_matches = re.findall(r"\b([A-Z]{2})\s*=?\s*([0-9]+(?:\.[0-9]+)?)", text)
        side_len: Dict[str, float] = {}
        for side, val in side_matches:
            if len(side) != 2:
                continue
            a, b = side[0], side[1]
            if not (a.isalpha() and b.isalpha()):
                continue
            key = f"{a}{b}"
            try:
                side_len[key] = float(val)
            except ValueError:
                continue

        if not side_len:
            return None

        # 解析直角顶点：如 angle C = 90
        right_vertex = None
        m = re.search(r"angle\s*([A-Z])\s*=\s*90|\\angle\s*([A-Z]).{0,15}90", text, flags=re.IGNORECASE)
        if m:
            right_vertex = (m.group(1) or m.group(2) or "").upper() or None

        # 找两条共享同一顶点的边，优先使用 right_vertex
        candidates: List[tuple] = []
        sides = list(side_len.items())
        for i in range(len(sides)):
            s1, l1 = sides[i]
            for j in range(i + 1, len(sides)):
                s2, l2 = sides[j]
                common = set(s1) & set(s2)
                if len(common) != 1:
                    continue
                v = list(common)[0]
                if right_vertex and v != right_vertex:
                    continue
                u = s1[0] if s1[1] == v else s1[1]
                w = s2[0] if s2[1] == v else s2[1]
                candidates.append((v, u, w, l1, l2))

        if not candidates:
            # 放宽：不要求与 right_vertex 匹配
            for i in range(len(sides)):
                s1, l1 = sides[i]
                for j in range(i + 1, len(sides)):
                    s2, l2 = sides[j]
                    common = set(s1) & set(s2)
                    if len(common) != 1:
                        continue
                    v = list(common)[0]
                    u = s1[0] if s1[1] == v else s1[1]
                    w = s2[0] if s2[1] == v else s2[1]
                    candidates.append((v, u, w, l1, l2))

        if not candidates:
            return None

        v, u, w, l_vu, l_vw = candidates[0]
        max_leg = max(l_vu, l_vw, 1e-6)
        scale = min(1.0, 3.2 / max_leg)

        V = [0.0, 0.0, 0.0]
        U = [0.0, round(l_vu * scale, 4), 0.0]
        W = [round(l_vw * scale, 4), 0.0, 0.0]

        points = [
            {"name": v, "coord": V, "source": "text_synth"},
            {"name": u, "coord": U, "source": "text_synth"},
            {"name": w, "coord": W, "source": "text_synth"},
        ]

        segments = [
            {"from": v, "to": u},
            {"from": v, "to": w},
            {"from": u, "to": w},
        ]
        polygons = [{"type": "triangle", "vertices": [u, v, w]}]
        angles = [{"vertex": v, "from": u, "to": w, "is_right": True}]

        return {
            "points": points,
            "segments": segments,
            "polygons": polygons,
            "angles": angles,
            "relations": [],
        }

    def _prepare_local_image(self, image_path: str) -> str:
        """
        准备本地图片路径。若输入是 URL，则下载到本地缓存目录。

        Args:
            image_path: 图片 URL 或本地路径

        Returns:
            本地可访问的图片路径
        """
        if not image_path.startswith(("http://", "https://")):
            return image_path

        cache_dir = os.path.join("output", "image_cache")
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, "input_image.png")
        urlretrieve(image_path, local_path) # 从网上下载文件，并保存到本地置顶路径
        return local_path

    def extract_diagram_region(self, local_image_path: str) -> Dict[str, Any]:
        """
        从整张题目图片中提取“核心几何图形区域”，并保存裁剪后的图片。

        实现方式：
        1. 使用视觉模型返回归一化 bbox（x1,y1,x2,y2，范围 0~1000）
        2. 将 bbox 映射到像素坐标并裁剪

        Args:
            local_image_path: 本地图片路径

        Returns:
            包含裁剪图路径和 bbox 信息的字典
        """
        try:
            from PIL import Image
        except ImportError:
            # Pillow 不可用时，回退到原图
            return {
                "diagram_image_path": local_image_path,
                "diagram_bbox": {},
                "warning": "Pillow 未安装，已回退使用原图",
            }

        with open(local_image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode() # decode是将bytes 转为 Python 字符串

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"},
                },
                {
                    "type": "text",
                    "text": (
                        "请找出这张题目图片中“几何图形主体”的最小外接矩形，"
                        "尽量排除题干文字、选项文字和空白区域。"
                        "\n只输出 JSON，不要解释："
                        "\n{\"x1\":整数,\"y1\":整数,\"x2\":整数,\"y2\":整数}"
                        "\n坐标为归一化坐标，范围 0~1000（相对整图左上角原点）。"
                    ),
                },
            ]
        )

        response = self.llm.invoke([message])
        bbox = self._parse_bbox_json(response.content)
        if not bbox:
            return {
                "diagram_image_path": local_image_path,
                "diagram_bbox": {},
                "warning": "未能解析图形 bbox，已回退使用原图",
            }

        with Image.open(local_image_path) as img:
            width, height = img.size

            x1 = max(0, min(width - 1, int(bbox["x1"] / 1000 * width)))
            y1 = max(0, min(height - 1, int(bbox["y1"] / 1000 * height)))
            x2 = max(x1 + 1, min(width, int(bbox["x2"] / 1000 * width)))
            y2 = max(y1 + 1, min(height, int(bbox["y2"] / 1000 * height)))

            # 给裁剪区域增加少量边距，避免贴边
            pad_x = int(0.1 * width)
            pad_y = int(0.1 * height)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)

            cropped = img.crop((x1, y1, x2, y2))

            out_dir = os.path.join("output", "image_cache")
            os.makedirs(out_dir, exist_ok=True)
            diagram_path = os.path.join(out_dir, "diagram_only.png")
            cropped.save(diagram_path)

        return {
            "diagram_image_path": diagram_path,
            "diagram_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }

    def _parse_bbox_json(self, content: str) -> Optional[Dict[str, int]]:
        """解析视觉模型输出的 bbox JSON（x1,y1,x2,y2）"""
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return None
        json_str = json_match.group()
        json_str = json_str.replace("\u201c", '"').replace("\u201d", '"')
        json_str = json_str.replace("，", ",").replace("：", ":")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        keys = ["x1", "y1", "x2", "y2"]
        if not all(k in data for k in keys):
            return None

        try:
            parsed = {k: int(data[k]) for k in keys}
        except (TypeError, ValueError):
            return None

        # 基本合法性检查（归一化坐标）
        if not (0 <= parsed["x1"] < parsed["x2"] <= 1000):
            return None
        if not (0 <= parsed["y1"] < parsed["y2"] <= 1000):
            return None

        return parsed

    def ocr_recognize(self, image_path: str) -> str:
        """
        使用 OCR 模型识别图片中的文本

        Args:
            image_path: 题目图片的 URL 或本地文件路径

        Returns:
            图片中识别出的文本
        """
        # 判断是本地文件还是 URL
        if image_path.startswith(("http://", "https://")):
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_path, "detail": "high"}
            }
        else:
            # 本地文件转 base64
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            }

        message = HumanMessage(
            content=[
                image_content,
                {
                    "type": "text",
                    "text": """请识别这张图片中题目的所有文字内容
                    要求：
                    1. 准确提取所有文字，包括标点符号
                    2. 数学公式用 LaTeX 格式表示
                    3. 按原格式排版（换行、缩进等）
                    直接输出识别的文字，不要添加其他解释。"""
                }
            ]
        )
        response = self.llm.invoke([message])

        return response.content




    def model_problem(self, local_image_path: str) -> str:
        """
        使用视觉模型理解图片等信息，进行数学建模，主要用于构建图片代码
        Args:
            local_image_path: 裁剪后的题图路径
        Returns:
            对题目数学建模方面的理解结果（字符串）
        """
        with open(local_image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    "detail": "high",
                },
                {
                    "type": "text",
                    "text": """请根据这张题图进行数学建模分析。
                    要求：
                    1. 所有命名点（A/B/C/D/C'等）及其相对位置
                    2. 某些点是由折叠，翻转，旋转等变换得到的，要说明变换关系和变换类型，甚至是具体的相对位置信息
                    3. 所有边(说明哪两个点)及其相对长度关系
                    4. 提取题图的形状信息，哪些点形成了一个什么图形，以及多图形时要描述图形之间的相对信息
                    5. 用字符串输出，不要用 JSON 数组。"""
                }
            ]
        )
        response = self.llm.invoke([message])

        return response.content.strip()


    def _to_jsonable(self, value: Any) -> Any:
        """将 numpy 标量/数组递归转换为可 JSON 序列化的原生 Python 类型。"""
        try:
            import numpy as np
        except ImportError:
            np = None

        if isinstance(value, dict):
            return {k: self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(v) for v in value]
        if np is not None:
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
        return value

    def parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON 内容"""
        # 提取 JSON 块
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return {"error": "未找到 JSON 内容", "raw_content": content[:500]}

        json_str = json_match.group()

        # 替换中文引号
        json_str = json_str.replace('"', '"').replace('"', '"').replace('"', '"').replace('"', '"')

        # 处理 LaTeX 转义：\command 形式需要变成 \\command
        # 但要注意不要重复转义已经是 \\ 的
        def escape_latex(match):
            cmd = match.group(0)
            if cmd.startswith('\\\\'):
                return cmd  # 已经是双斜杠
            return '\\' + cmd  # 单斜杠变双斜杠

        # 匹配 \ 开头的 LaTeX 命令
        json_str = re.sub(r'\\[a-zA-Z]+', escape_latex, json_str)

        try:
            result = json.loads(json_str)
            # 后处理：把公式中的 \\\\ 清理成 \\（方便后续使用）
            self._clean_latex_backslashes(result)
            return result
        except json.JSONDecodeError as e:
            # 如果还是失败，尝试用 ast.literal_eval 处理
            import ast
            try:
                # 把 JSON 当 Python 字典处理（支持单引号等）
                return ast.literal_eval(json_str)
            except:
                pass

            return {
                "error": f"无法解析 JSON: {e}",
                "raw_content": json_str[:500]
            }

    def _clean_latex_backslashes(self, data: Any) -> None:
        """递归清理嵌套结构中的 LaTeX 反斜杠，并把列表转成字符串"""
        if isinstance(data, dict):
            for key, value in list(data.items()):
                if isinstance(value, list):
                    # 把列表拼接成字符串
                    data[key] = ', '.join(str(v) for v in value)
                elif isinstance(value, str):
                    # 把 \\\\ 变成 \
                    data[key] = value.replace('\\\\', '\\')
                else:
                    self._clean_latex_backslashes(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    data[i] = item.replace('\\\\', '\\')
                else:
                    self._clean_latex_backslashes(item)


    def _flatten_list_to_string(self, data: Any) -> None:
        """递归把嵌套结构中的列表转成字符串"""
        if isinstance(data, dict):
            for key, value in list(data.items()):
                if isinstance(value, list):
                    data[key] = ', '.join(str(v) for v in value)
                elif isinstance(value, (dict, list)):
                    self._flatten_list_to_string(value)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._flatten_list_to_string(item)

    def _indent_code(self, code: str, spaces: int) -> str:
        """给代码块添加缩进"""
        indent = " " * spaces
        lines = code.strip("\n").splitlines()
        return "\n".join(f"{indent}{line}" if line.strip() else "" for line in lines) + "\n"


