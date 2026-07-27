from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


DAI_ALIGNMENT_KIND = "drop_warmup_and_sample_exact_source_timestamps"
SPIDER_ALIGNMENT_KIND = "sample_exact_source_timestamps"
SPIDER_SOURCE_KIND = "original_spider_mjwp"


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def resolve_manifest_path(value: str | None, root: Path, repo_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [root / path, repo_root / path]
    for parent in (root, *root.parents):
        candidates.append(parent / path)
        if parent.name == "experiments":
            candidates.append(parent.parent / path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def valid_dai_alignment(robot_root: Path, repo_root: Path) -> bool:
    data = load_json_object(robot_root / "mjwp_alignment_manifest.json")
    if not data or data.get("applied") is not True:
        return False
    if data.get("alignment_kind") != DAI_ALIGNMENT_KIND:
        return False
    try:
        start_index = int(data.get("start_index", -1))
        warmup_steps = int(data.get("warmup_steps", -2))
        target_count = int(data.get("target_count") or data.get("target_frames") or 0)
        source_frame_count = int(data.get("source_frame_count") or 0)
    except (TypeError, ValueError):
        return False
    # Upstream DAI saves post-step states and omits the initial state.  The
    # source t=0 state is therefore raw sample warmup_steps - 1.
    if (
        warmup_steps <= 0
        or start_index != warmup_steps - 1
        or start_index < 0
        or target_count <= 0
    ):
        return False
    if source_frame_count != target_count:
        return False
    if data.get("complete_source_coverage") is not True:
        return False
    if data.get("exact_source_timestamps") is not True:
        return False
    if data.get("sampling_semantics") not in {
        "exact_source_timestamp_indices",
        "exact_source_timestamp_interpolation",
    }:
        return False
    # A filename existing in the cell is not sufficient evidence: a failed
    # rerender can otherwise leave an older video next to a new trajectory and
    # manifest.  Production admission binds every render input/output by exact
    # canonical path and SHA-256, and binds the measured video grid as well.
    canonical_source_trajectory = next(
        (
            candidate
            for candidate in (
                robot_root / "trajectory_mjwp_act.npz",
                robot_root / "trajectory_mjwp.npz",
            )
            if candidate.is_file()
        ),
        None,
    )
    if canonical_source_trajectory is None:
        return False
    canonical_paths = {
        "aligned_video": robot_root / "visualization_mjwp_act_aligned.mp4",
        "aligned_trajectory": robot_root / "trajectory_mjwp_act_aligned.npz",
        "render_scene": robot_root / "scene_act.xml",
        "source_trajectory": canonical_source_trajectory,
    }
    bindings = data.get("artifact_bindings")
    if not isinstance(bindings, dict):
        return False
    for label, canonical in canonical_paths.items():
        binding = bindings.get(label)
        if not isinstance(binding, dict):
            return False
        bound_path = resolve_manifest_path(
            str(binding.get("path") or ""), robot_root, repo_root
        )
        if bound_path is None or not canonical.is_file():
            return False
        try:
            if bound_path.resolve() != canonical.resolve():
                return False
        except OSError:
            return False
        recorded_sha = binding.get("sha256")
        try:
            current_sha = file_sha256(canonical)
        except OSError:
            return False
        if not isinstance(recorded_sha, str) or recorded_sha != current_sha:
            return False

    output_root = next(
        (
            candidate
            for candidate in (robot_root, *robot_root.parents)
            if candidate.name == "retargeting_outputs"
        ),
        None,
    )
    if output_root is None:
        return False
    canonical_source_video = output_root.parent / "raw_dir" / "raw.mp4"
    source_video_binding = bindings.get("source_video")
    if not isinstance(source_video_binding, dict) or not canonical_source_video.is_file():
        return False
    bound_source_video = resolve_manifest_path(
        str(source_video_binding.get("path") or ""), robot_root, repo_root
    )
    if bound_source_video is None or not bound_source_video.is_file():
        return False
    try:
        if bound_source_video.resolve() != canonical_source_video.resolve():
            return False
        if source_video_binding.get("sha256") != file_sha256(canonical_source_video):
            return False
    except OSError:
        return False

    # The top-level paths are consumed by downstream audit/index code and must
    # agree with the authoritative bindings rather than merely share a name.
    for field in (
        "aligned_video",
        "aligned_trajectory",
        "render_scene",
        "source_trajectory",
    ):
        recorded_path = resolve_manifest_path(
            str(data.get(field) or ""), robot_root, repo_root
        )
        if recorded_path is None:
            return False
        try:
            if recorded_path.resolve() != canonical_paths[field].resolve():
                return False
        except OSError:
            return False

    video_binding = bindings["aligned_video"]
    try:
        video_frames = int(video_binding.get("frame_count"))
        video_fps = float(video_binding.get("fps"))
        source_video_frames = int(source_video_binding.get("frame_count"))
        source_video_fps = float(source_video_binding.get("fps"))
        render_fps = float(data.get("render_fps"))
        target_fps = float(data.get("target_fps"))
        robot_render_fps = float(data.get("robot_render_fps"))
        render_dt = float(data.get("render_dt"))
        trace_dt = float(data.get("trace_dt"))
    except (TypeError, ValueError):
        return False
    if video_frames != target_count or source_video_frames != source_frame_count:
        return False
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (
            video_fps,
            source_video_fps,
            render_fps,
            target_fps,
            robot_render_fps,
        )
    ):
        return False
    if not all(
        math.isclose(value, target_fps, rel_tol=1.0e-7, abs_tol=1.0e-9)
        for value in (video_fps, source_video_fps, render_fps, robot_render_fps)
    ):
        return False
    expected_dt = 1.0 / target_fps
    if not all(
        math.isfinite(value)
        and value > 0.0
        and math.isclose(value, expected_dt, rel_tol=1.0e-7, abs_tol=1.0e-9)
        for value in (render_dt, trace_dt)
    ):
        return False
    return True


def spider_alignment_manifest(
    root: Path,
    output_manifest: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    manifest = output_manifest or load_json_object(root / "spider_output_manifest.json")
    if manifest:
        alignment = manifest.get("alignment")
        if isinstance(alignment, dict):
            return alignment
    return load_json_object(root / "spider_mjwp_alignment_manifest.json")


def valid_spider_tracking(output_manifest: dict[str, Any] | None) -> bool:
    manifest = output_manifest or {}
    fallback = manifest.get("mjwp_fallback")
    if isinstance(fallback, dict) and fallback.get("applied"):
        return False
    tracking = manifest.get("object_tracking_quality")
    if not isinstance(tracking, dict) and isinstance(fallback, dict):
        tracking = fallback.get("tracking_error")
    if not isinstance(tracking, dict):
        return False
    status = tracking.get("status")
    if status is not None:
        return status == "ok"

    # Compatibility for pre-quality-schema manifests. Fresh runs always carry
    # an explicit status plus p95/lost-frame checks from the mesh-body evaluator.
    if not tracking.get("available") or not bool(tracking.get("finite", True)):
        return False
    try:
        return (
            float(tracking.get("pos_median", float("inf")))
            <= float((fallback or {}).get("pos_threshold", 0.12))
            and float(tracking.get("rot_median", float("inf")))
            <= float((fallback or {}).get("rot_threshold", 0.45))
        )
    except (TypeError, ValueError):
        return False


def valid_spider_alignment(
    root: Path,
    repo_root: Path,
    output_manifest: dict[str, Any] | None = None,
    *,
    require_tracking: bool = True,
) -> bool:
    manifest = output_manifest or load_json_object(root / "spider_output_manifest.json") or {}
    if require_tracking and not valid_spider_tracking(manifest):
        return False
    alignment = spider_alignment_manifest(root, manifest)
    if not alignment or alignment.get("applied") is not True:
        return False
    if alignment.get("alignment_kind") != SPIDER_ALIGNMENT_KIND:
        return False
    if alignment.get("source_kind") != SPIDER_SOURCE_KIND:
        return False
    if alignment.get("exact_source_timestamps") is not True:
        return False
    if alignment.get("sampling_semantics") != "exact_source_timestamp_indices":
        return False
    try:
        if int(alignment.get("start_index", -1)) != 0:
            return False
        if int(alignment.get("warmup_steps", -1)) != 0:
            return False
        target_count = int(
            alignment.get("target_count") or alignment.get("target_frames") or 0
        )
        source_frame_count = int(alignment.get("source_frame_count") or 0)
    except (TypeError, ValueError):
        return False
    if (
        target_count <= 0
        or source_frame_count != target_count
        or alignment.get("complete_source_coverage") is not True
    ):
        return False

    # SPIDER used to admit any file with an aligned-looking name.  That lets a
    # failed rerender or a later copy step pair an old video/trajectory with new
    # numerical QC.  Fresh production runs install one canonical trio directly
    # under the exact cell root and bind every byte plus the measured video grid.
    flavor = alignment.get("artifact_flavor")
    if flavor not in {"act", "plain"}:
        return False
    suffix = "_act" if flavor == "act" else ""
    canonical_paths = {
        "aligned_video": root / f"visualization_mjwp{suffix}_aligned.mp4",
        "aligned_trajectory": root / f"trajectory_mjwp{suffix}_aligned.npz",
        "timing_manifest": root / "spider_timing.json",
    }
    bindings = alignment.get("artifact_bindings")
    if not isinstance(bindings, dict):
        return False
    for label, canonical in canonical_paths.items():
        binding = bindings.get(label)
        if not isinstance(binding, dict) or not canonical.is_file():
            return False
        bound_path = resolve_manifest_path(
            str(binding.get("path") or ""), root, repo_root
        )
        if bound_path is None:
            return False
        try:
            if bound_path.resolve() != canonical.resolve():
                return False
            if binding.get("sha256") != file_sha256(canonical):
                return False
        except OSError:
            return False

    # The fields consumed by the renderer, QC, and index must all point at the
    # same canonical artifacts as the authoritative bindings.
    for field in ("aligned_video", "aligned_trajectory"):
        recorded = resolve_manifest_path(
            str(alignment.get(field) or ""), root, repo_root
        )
        if recorded is None:
            return False
        try:
            if recorded.resolve() != canonical_paths[field].resolve():
                return False
        except OSError:
            return False

    # The scene must be the freshly rebuilt scene inside this cell's dataset.
    # It intentionally stays at its native task directory because SPIDER scene
    # asset paths are relative to that location.
    scene_binding = bindings.get("render_scene")
    scene_recorded = resolve_manifest_path(
        str(alignment.get("render_scene") or ""), root, repo_root
    )
    if not isinstance(scene_binding, dict) or scene_recorded is None:
        return False
    scene_bound = resolve_manifest_path(
        str(scene_binding.get("path") or ""), root, repo_root
    )
    expected_scene_name = "scene_act.xml" if flavor == "act" else "scene.xml"
    if scene_bound is None or not scene_bound.is_file() or scene_bound.name != expected_scene_name:
        return False
    try:
        if scene_bound.resolve() != scene_recorded.resolve():
            return False
        scene_bound.resolve().relative_to((root / "dataset").resolve())
        if scene_binding.get("sha256") != file_sha256(scene_bound):
            return False
    except (OSError, ValueError):
        return False

    source_binding = bindings.get("source_trajectory")
    source_recorded = resolve_manifest_path(
        str(alignment.get("source_trajectory") or ""), root, repo_root
    )
    if not isinstance(source_binding, dict) or source_recorded is None:
        return False
    source_bound = resolve_manifest_path(
        str(source_binding.get("path") or ""), root, repo_root
    )
    if source_bound is None or not source_bound.is_file():
        return False
    try:
        if source_bound.resolve() != source_recorded.resolve():
            return False
        source_bound.resolve().relative_to(root.resolve())
        if source_binding.get("sha256") != file_sha256(source_bound):
            return False
    except (OSError, ValueError):
        return False

    timing = load_json_object(canonical_paths["timing_manifest"])
    if not isinstance(timing, dict):
        return False
    video_binding = bindings["aligned_video"]
    try:
        video_frames = int(video_binding.get("frame_count"))
        video_fps = float(video_binding.get("fps"))
        render_fps = float(alignment.get("render_fps"))
        target_fps = float(alignment.get("target_fps"))
        source_fps = float(alignment.get("source_fps"))
        ref_dt = float(alignment.get("ref_dt"))
        render_dt = float(alignment.get("render_dt"))
        trace_dt = float(alignment.get("trace_dt"))
        sim_dt = float(alignment.get("sim_dt"))
        timing_frames = int(timing.get("source_frame_count"))
        timing_fps = float(timing.get("source_fps"))
        timing_ref_dt = float(timing.get("ref_dt"))
        timing_sim_dt = float(timing.get("sim_dt"))
    except (TypeError, ValueError):
        return False
    if video_frames != target_count or timing_frames != source_frame_count:
        return False
    grid_values = (video_fps, render_fps, target_fps, source_fps, timing_fps)
    if not all(math.isfinite(value) and value > 0.0 for value in grid_values):
        return False
    if not all(
        math.isclose(value, timing_fps, rel_tol=1.0e-7, abs_tol=1.0e-9)
        for value in grid_values
    ):
        return False
    expected_dt = 1.0 / timing_fps
    if not all(
        math.isfinite(value)
        and value > 0.0
        and math.isclose(value, expected_dt, rel_tol=1.0e-7, abs_tol=1.0e-9)
        for value in (ref_dt, render_dt, trace_dt, timing_ref_dt)
    ):
        return False
    if not (
        math.isfinite(sim_dt)
        and sim_dt > 0.0
        and math.isclose(sim_dt, timing_sim_dt, rel_tol=1.0e-9, abs_tol=1.0e-12)
    ):
        return False
    return True
