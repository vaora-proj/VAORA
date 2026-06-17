import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Sequence, Tuple


Point = Tuple[float, float]
Polygon = List[Point]

SHAPE_BALL = "BALL"
SHAPE_BAR = "BAR"
SHAPE_TRIANGLE = "TRIANGLE"
SHAPE_TRAPEZOID = "TRAPEZOID"
SHAPE_JAR = "JAR"


def _collect_json_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path] if path.lower().endswith(".json") else []
    if not os.path.isdir(path):
        return []

    files: List[str] = []
    for root, _, names in os.walk(path):
        for name in names:
            if name.lower().endswith(".json"):
                files.append(os.path.join(root, name))
    return sorted(files)


def _is_toolpicker_payload(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("world"), dict)
        and isinstance(payload.get("tools"), dict)
    )


def _to_polygon(points: Sequence[Sequence[float]]) -> Polygon:
    return [(float(x), float(y)) for x, y in points]


def _simplify_collinear(poly: Polygon, eps: float = 1e-7) -> Polygon:
    if len(poly) <= 3:
        return poly[:]

    simplified: Polygon = []
    n = len(poly)
    for i in range(n):
        p_prev = poly[(i - 1) % n]
        p_curr = poly[i]
        p_next = poly[(i + 1) % n]
        v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) > eps:
            simplified.append(p_curr)
    return simplified if len(simplified) >= 3 else poly[:]


def _is_circle_like(poly: Polygon, rel_tol: float = 0.08) -> bool:
    if len(poly) < 8:
        return False
    cx = sum(x for x, _ in poly) / len(poly)
    cy = sum(y for _, y in poly) / len(poly)
    radii = [math.hypot(x - cx, y - cy) for x, y in poly]
    mean_r = sum(radii) / len(radii)
    if mean_r <= 1e-9:
        return False
    max_dev = max(abs(r - mean_r) for r in radii)
    return (max_dev / mean_r) <= rel_tol


def _edge_vectors(poly: Polygon) -> List[Point]:
    return [
        (poly[(i + 1) % len(poly)][0] - poly[i][0], poly[(i + 1) % len(poly)][1] - poly[i][1])
        for i in range(len(poly))
    ]


def _parallel(v1: Point, v2: Point, eps: float = 1e-6) -> bool:
    return abs(v1[0] * v2[1] - v1[1] * v2[0]) < eps


def _is_concave(poly: Polygon, eps: float = 1e-7) -> bool:
    if len(poly) < 4:
        return False

    pos = False
    neg = False
    n = len(poly)
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        p2 = poly[(i + 2) % n]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross > eps:
            pos = True
        elif cross < -eps:
            neg = True
        if pos and neg:
            return True
    return False


def _classify_polygon(poly: Polygon) -> str:
    simp = _simplify_collinear(poly)

    if _is_circle_like(simp):
        return SHAPE_BALL
    if _is_concave(simp):
        return SHAPE_JAR
    if len(simp) == 3:
        return SHAPE_TRIANGLE
    if len(simp) == 4:
        edges = _edge_vectors(simp)
        # Rectangle/square (including rotated): two pairs of parallel opposite edges.
        if _parallel(edges[0], edges[2]) and _parallel(edges[1], edges[3]):
            return SHAPE_BAR
        # Fallback for non-rectangular quadrilateral.
        return SHAPE_TRAPEZOID
    # For remaining polygons (e.g., 5-point wedge), use trapezoid bucket.
    return SHAPE_TRAPEZOID


def _classify_tool(parts: Sequence[Sequence[Sequence[float]]]) -> Tuple[str, List[str]]:
    part_shapes = [_classify_polygon(_to_polygon(poly)) for poly in parts]
    counts = Counter(part_shapes)

    # Rule: multi-part tool is considered a JAR.
    if len(part_shapes) > 1:
        return SHAPE_JAR, part_shapes
    if counts[SHAPE_JAR] > 0:
        return SHAPE_JAR, part_shapes
    if counts[SHAPE_BALL] > 0:
        return SHAPE_BALL, part_shapes
    if counts[SHAPE_TRIANGLE] > 0:
        return SHAPE_TRIANGLE, part_shapes
    if counts[SHAPE_TRAPEZOID] > 0:
        return SHAPE_TRAPEZOID, part_shapes
    return SHAPE_BAR, part_shapes


def export_tool_shapes(input_path: str, output_json: str) -> None:
    json_files = _collect_json_files(input_path)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found at: {input_path}")

    output: Dict[str, Dict[str, object]] = {}
    skipped = 0

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue

        if not _is_toolpicker_payload(payload):
            skipped += 1
            continue

        task_name = os.path.splitext(os.path.basename(json_path))[0]
        tools = payload.get("tools", {})
        tool_names = payload.get("toolNames")
        if not isinstance(tool_names, list):
            tool_names = list(tools.keys())

        task_record: Dict[str, object] = {
            "task_json": json_path,
            "tools": {},
        }
        for tool_name in tool_names:
            parts = tools.get(tool_name)
            if not isinstance(parts, list):
                continue
            shape, part_shapes = _classify_tool(parts)
            task_record["tools"][tool_name] = {
                "shape": shape,
                "part_shapes": part_shapes,
            }

        output[task_name] = task_record

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(
        f"Done. tasks={len(output)}, skipped={skipped}, output_json={output_json}"
    )


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.abspath(os.path.join(script_dir, ".."))
    if env_dir not in sys.path:
        sys.path.insert(0, env_dir)
    from paths import artifact_path  # noqa: E402
    parser = argparse.ArgumentParser(
        description="Detect tool shape category for each task/tool and export JSON."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.path.join(env_dir, "Trials"),
        help="Path to ToolPicker JSON file or directory.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=artifact_path("tool_shape_labels.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_tool_shapes(args.input_path, args.output_json)
