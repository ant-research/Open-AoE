from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from aoe_retarget_lab.alignment import valid_dai_alignment, valid_spider_alignment


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_dai_alignment(robot: Path) -> tuple[dict, Path]:
    video = robot / "visualization_mjwp_act_aligned.mp4"
    trajectory = robot / "trajectory_mjwp_act_aligned.npz"
    source_trajectory = robot / "trajectory_mjwp_act.npz"
    scene = robot / "scene_act.xml"
    output_root = next(parent for parent in robot.parents if parent.name == "retargeting_outputs")
    source_video = output_root.parent / "raw_dir" / "raw.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"aligned-video")
    trajectory.write_bytes(b"aligned-trajectory")
    source_trajectory.write_bytes(b"raw-mjwp-source-trajectory")
    scene.write_bytes(b"<mujoco/>")
    source_video.parent.mkdir(parents=True, exist_ok=True)
    source_video.write_bytes(b"exact-route-source-video")
    fps = 30.0
    manifest = {
        "applied": True,
        "alignment_kind": "drop_warmup_and_sample_exact_source_timestamps",
        "start_index": 89,
        "warmup_steps": 90,
        "target_count": 45,
        "source_frame_count": 45,
        "complete_source_coverage": True,
        "exact_source_timestamps": True,
        "sampling_semantics": "exact_source_timestamp_indices",
        "aligned_video": str(video),
        "aligned_trajectory": str(trajectory),
        "source_trajectory": str(source_trajectory),
        "render_scene": str(scene),
        "render_fps": fps,
        "target_fps": fps,
        "robot_render_fps": fps,
        "render_dt": 1.0 / fps,
        "trace_dt": 1.0 / fps,
        "artifact_bindings": {
            "aligned_video": {
                "path": str(video),
                "sha256": sha256(video),
                "frame_count": 45,
                "fps": fps,
            },
            "aligned_trajectory": {
                "path": str(trajectory),
                "sha256": sha256(trajectory),
            },
            "source_trajectory": {
                "path": str(source_trajectory),
                "sha256": sha256(source_trajectory),
            },
            "render_scene": {
                "path": str(scene),
                "sha256": sha256(scene),
            },
            "source_video": {
                "path": str(source_video),
                "sha256": sha256(source_video),
                "frame_count": 45,
                "fps": fps,
            },
        },
    }
    return manifest, video


def make_spider_alignment(root: Path) -> tuple[dict, Path, Path]:
    video = root / "visualization_mjwp_act_aligned.mp4"
    trajectory = root / "trajectory_mjwp_act_aligned.npz"
    timing = root / "spider_timing.json"
    source = root / "dataset" / "processed" / "demo" / "robot" / "trajectory_mjwp_act.npz"
    scene = root / "dataset" / "processed" / "demo" / "robot" / "scene_act.xml"
    for path, payload in (
        (video, b"aligned-video"),
        (trajectory, b"aligned-trajectory"),
        (source, b"fresh-source-trajectory"),
        (scene, b"<mujoco/>"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    fps = 30.0
    write_json(
        timing,
        {
            "source_frame_count": 45,
            "source_fps": fps,
            "ref_dt": 1.0 / fps,
            "sim_dt": 1.0 / 180.0,
        },
    )
    alignment = {
        "applied": True,
        "alignment_kind": "sample_exact_source_timestamps",
        "source_kind": "original_spider_mjwp",
        "exact_source_timestamps": True,
        "sampling_semantics": "exact_source_timestamp_indices",
        "complete_source_coverage": True,
        "start_index": 0,
        "warmup_steps": 0,
        "source_frame_count": 45,
        "target_frames": 45,
        "target_count": 45,
        "artifact_flavor": "act",
        "aligned_video": str(video),
        "aligned_trajectory": str(trajectory),
        "render_scene": str(scene),
        "source_trajectory": str(source),
        "source_fps": fps,
        "render_fps": fps,
        "target_fps": fps,
        "ref_dt": 1.0 / fps,
        "render_dt": 1.0 / fps,
        "trace_dt": 1.0 / fps,
        "sim_dt": 1.0 / 180.0,
        "artifact_bindings": {
            "aligned_video": {
                "path": str(video),
                "sha256": sha256(video),
                "frame_count": 45,
                "fps": fps,
            },
            "aligned_trajectory": {
                "path": str(trajectory),
                "sha256": sha256(trajectory),
            },
            "render_scene": {"path": str(scene), "sha256": sha256(scene)},
            "source_trajectory": {
                "path": str(source),
                "sha256": sha256(source),
            },
            "timing_manifest": {
                "path": str(timing),
                "sha256": sha256(timing),
            },
        },
    }
    return alignment, video, trajectory


def test_dai_requires_warmup_drop_and_aligned_video(tmp_path: Path) -> None:
    robot = tmp_path / "cell" / "retargeting_outputs" / "sharpa" / "left" / "task" / "0"
    manifest, _ = make_dai_alignment(robot)
    write_json(robot / "mjwp_alignment_manifest.json", manifest)
    assert valid_dai_alignment(robot, tmp_path)
    manifest["start_index"] = 0
    write_json(robot / "mjwp_alignment_manifest.json", manifest)
    assert not valid_dai_alignment(robot, tmp_path)
    manifest["start_index"] = manifest["warmup_steps"] - 1
    manifest["complete_source_coverage"] = False
    write_json(robot / "mjwp_alignment_manifest.json", manifest)
    assert not valid_dai_alignment(robot, tmp_path)


def test_dai_rejects_unbound_or_stale_canonical_video(tmp_path: Path) -> None:
    robot = tmp_path / "cell" / "retargeting_outputs" / "sharpa" / "left" / "task" / "0"
    manifest, video = make_dai_alignment(robot)
    manifest_path = robot / "mjwp_alignment_manifest.json"

    missing_path_binding = dict(manifest)
    missing_path_binding.pop("aligned_video")
    write_json(manifest_path, missing_path_binding)
    assert not valid_dai_alignment(robot, tmp_path)

    write_json(manifest_path, manifest)
    video.write_bytes(b"stale-video-from-another-run")
    assert not valid_dai_alignment(robot, tmp_path)


def test_dai_rejects_video_grid_or_render_timing_mismatch(tmp_path: Path) -> None:
    robot = tmp_path / "cell" / "retargeting_outputs" / "sharpa" / "left" / "task" / "0"
    manifest, _ = make_dai_alignment(robot)
    manifest_path = robot / "mjwp_alignment_manifest.json"

    manifest["artifact_bindings"]["aligned_video"]["frame_count"] = 44
    write_json(manifest_path, manifest)
    assert not valid_dai_alignment(robot, tmp_path)

    manifest["artifact_bindings"]["aligned_video"]["frame_count"] = 45
    manifest["trace_dt"] = 0.04
    write_json(manifest_path, manifest)
    assert not valid_dai_alignment(robot, tmp_path)


def test_dai_rejects_stale_or_cross_route_source_video(tmp_path: Path) -> None:
    robot = tmp_path / "cell" / "retargeting_outputs" / "sharpa" / "left" / "task" / "0"
    manifest, _ = make_dai_alignment(robot)
    manifest_path = robot / "mjwp_alignment_manifest.json"
    write_json(manifest_path, manifest)
    assert valid_dai_alignment(robot, tmp_path)

    source_video = tmp_path / "cell" / "raw_dir" / "raw.mp4"
    source_video.write_bytes(b"source-video-replaced-after-retarget")
    assert not valid_dai_alignment(robot, tmp_path)

    manifest, _ = make_dai_alignment(robot)
    manifest["artifact_bindings"]["source_video"]["frame_count"] = 44
    write_json(manifest_path, manifest)
    assert not valid_dai_alignment(robot, tmp_path)

    manifest["artifact_bindings"]["source_video"]["frame_count"] = 45
    manifest["artifact_bindings"]["source_video"]["fps"] = 60.0
    write_json(manifest_path, manifest)
    assert not valid_dai_alignment(robot, tmp_path)

    other = tmp_path / "other_cell" / "raw_dir" / "raw.mp4"
    other.parent.mkdir(parents=True)
    other.write_bytes(source_video.read_bytes())
    manifest["artifact_bindings"]["source_video"].update(
        {"path": str(other), "sha256": sha256(other), "fps": 30.0}
    )
    write_json(manifest_path, manifest)
    assert not valid_dai_alignment(robot, tmp_path)


def test_dai_rejects_stale_or_cross_route_source_trajectory(tmp_path: Path) -> None:
    robot = tmp_path / "cell" / "retargeting_outputs" / "sharpa" / "left" / "task" / "0"
    manifest, _ = make_dai_alignment(robot)
    manifest_path = robot / "mjwp_alignment_manifest.json"
    write_json(manifest_path, manifest)
    assert valid_dai_alignment(robot, tmp_path)

    source_trajectory = robot / "trajectory_mjwp_act.npz"
    source_trajectory.write_bytes(b"raw-rollout-replaced-after-alignment")
    assert not valid_dai_alignment(robot, tmp_path)

    manifest, _ = make_dai_alignment(robot)
    other = tmp_path / "other_cell" / "trajectory_mjwp_act.npz"
    other.parent.mkdir(parents=True)
    other.write_bytes(source_trajectory.read_bytes())
    manifest["source_trajectory"] = str(other)
    manifest["artifact_bindings"]["source_trajectory"] = {
        "path": str(other),
        "sha256": sha256(other),
    }
    write_json(manifest_path, manifest)
    assert not valid_dai_alignment(robot, tmp_path)


def test_dai_accepts_bound_legacy_named_source_trajectory(tmp_path: Path) -> None:
    robot = tmp_path / "cell" / "retargeting_outputs" / "sharpa" / "left" / "task" / "0"
    manifest, _ = make_dai_alignment(robot)
    source_trajectory = robot / "trajectory_mjwp_act.npz"
    legacy_source = robot / "trajectory_mjwp.npz"
    source_trajectory.rename(legacy_source)
    manifest["source_trajectory"] = str(legacy_source)
    manifest["artifact_bindings"]["source_trajectory"] = {
        "path": str(legacy_source),
        "sha256": sha256(legacy_source),
    }
    write_json(robot / "mjwp_alignment_manifest.json", manifest)
    assert valid_dai_alignment(robot, tmp_path)


def test_spider_requires_original_mjwp_and_rejects_fallback(tmp_path: Path) -> None:
    root = tmp_path / "spider"
    alignment, _, _ = make_spider_alignment(root)
    write_json(
        root / "spider_output_manifest.json",
        {
            "alignment": alignment,
            "object_tracking_quality": {"status": "ok", "available": True, "finite": True},
            "mjwp_fallback": {"applied": False},
        },
    )
    assert valid_spider_alignment(root, tmp_path)
    write_json(
        root / "spider_output_manifest.json",
        {
            "alignment": alignment,
            "object_tracking_quality": {"status": "ok", "available": True, "finite": True},
            "mjwp_fallback": {"applied": True},
        },
    )
    assert not valid_spider_alignment(root, tmp_path)


def test_spider_rejects_missing_or_invalid_mesh_body_tracking(tmp_path: Path) -> None:
    root = tmp_path / "spider"
    alignment, _, _ = make_spider_alignment(root)
    write_json(root / "spider_output_manifest.json", {"alignment": alignment})
    assert not valid_spider_alignment(root, tmp_path)
    write_json(
        root / "spider_output_manifest.json",
        {
            "alignment": alignment,
            "object_tracking_quality": {
                "status": "invalid",
                "metrics": {"pos_p95_m": 0.8, "lost_frame_fraction": 0.2},
            },
            "mjwp_fallback": {"applied": False},
        },
    )
    assert not valid_spider_alignment(root, tmp_path)


def test_spider_rejects_noncanonical_source(tmp_path: Path) -> None:
    root = tmp_path / "spider"
    alignment, _, _ = make_spider_alignment(root)
    alignment["source_kind"] = "hybrid_ik_fallback"
    write_json(
        root / "spider_output_manifest.json",
        {
            "alignment": alignment,
            "object_tracking_quality": {"status": "ok"},
            "mjwp_fallback": {"applied": False},
        },
    )
    assert not valid_spider_alignment(root, tmp_path)


def test_spider_rejects_stale_artifact_and_source_grid_override(tmp_path: Path) -> None:
    root = tmp_path / "spider"
    alignment, video, trajectory = make_spider_alignment(root)
    manifest_path = root / "spider_output_manifest.json"
    payload = {
        "alignment": alignment,
        "object_tracking_quality": {"status": "ok"},
        "mjwp_fallback": {"applied": False},
    }
    write_json(manifest_path, payload)
    assert valid_spider_alignment(root, tmp_path)

    trajectory.write_bytes(b"stale-trajectory")
    assert not valid_spider_alignment(root, tmp_path)

    alignment, video, _ = make_spider_alignment(root)
    alignment["render_fps"] = 60.0
    payload["alignment"] = alignment
    write_json(manifest_path, payload)
    assert not valid_spider_alignment(root, tmp_path)

    alignment["render_fps"] = 30.0
    payload["alignment"] = alignment
    write_json(manifest_path, payload)
    assert valid_spider_alignment(root, tmp_path)
    video.write_bytes(b"stale-video")
    assert not valid_spider_alignment(root, tmp_path)


def test_spider_review_uses_plain_canonical_binding_not_act_alias(tmp_path: Path) -> None:
    root = tmp_path / "spider"
    alignment, act_video, act_trajectory = make_spider_alignment(root)
    plain_video = root / "visualization_mjwp_aligned.mp4"
    plain_trajectory = root / "trajectory_mjwp_aligned.npz"
    act_video.rename(plain_video)
    act_trajectory.rename(plain_trajectory)
    act_scene = Path(alignment["render_scene"])
    plain_scene = act_scene.with_name("scene.xml")
    act_scene.rename(plain_scene)
    alignment.update(
        {
            "artifact_flavor": "plain",
            "aligned_video": str(plain_video),
            "aligned_trajectory": str(plain_trajectory),
            "render_scene": str(plain_scene),
        }
    )
    for label, path in (
        ("aligned_video", plain_video),
        ("aligned_trajectory", plain_trajectory),
        ("render_scene", plain_scene),
    ):
        alignment["artifact_bindings"][label].update(
            {"path": str(path), "sha256": sha256(path)}
        )
    # These compatibility aliases are deliberately not authoritative.
    act_video.write_bytes(b"unbound-act-video-alias")
    act_trajectory.write_bytes(b"unbound-act-trajectory-alias")
    payload = {
        "alignment": alignment,
        "object_tracking_quality": {"status": "invalid"},
        "mjwp_fallback": {"applied": False},
    }
    write_json(root / "spider_output_manifest.json", payload)

    assert not valid_spider_alignment(root, tmp_path)
    assert valid_spider_alignment(root, tmp_path, require_tracking=False)
    act_video.write_bytes(b"mutated-unbound-alias")
    assert valid_spider_alignment(root, tmp_path, require_tracking=False)
    plain_video.write_bytes(b"mutated-bound-canonical-video")
    assert not valid_spider_alignment(root, tmp_path, require_tracking=False)


def test_dai_spider_does_not_fuse_ego_by_internal_default() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "reuse_v4_for_12_demos.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    target = next(node for node in functions if node.name == "prepare_trajectory_assets")
    kw_defaults = dict(zip((arg.arg for arg in target.args.kwonlyargs), target.args.kw_defaults))
    assert isinstance(kw_defaults["spider_fuse_ego_object_for_dai"], ast.Constant)
    assert kw_defaults["spider_fuse_ego_object_for_dai"].value is False


def test_spider_wrapper_uses_mesh_body_quality_not_qpos_tail_slices() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_spider_retarget.sh"
    source = script.read_text(encoding="utf-8")
    assert "run_backend_stage run_mjwp" in source
    assert "def object_tracking_error" not in source
    assert "def object_slice_for_hand" not in source
    assert "trajectory_hybrid_object_ref" not in source
    assert '--ref-dt "$ref_dt"' in source
    assert '"ref_dt=$ref_dt"' in source
    assert "GUIDANCE_DECAY" not in source
    assert "SPIDER_IK_SEED" not in source


def test_matrix_manifest_separates_demo_admission_from_completeness() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "reuse_v4_for_12_demos.py").read_text()
    assert '"policy": "retarget_only_plain_mujoco_numerical_and_visual_v3"' in source
    assert '"matrix_completeness_required": False' in source
    assert '"numerical_qc_is_diagnostic": False' in source
    assert '"numerical_invalid_can_be_visual_success": False' in source
    assert '"failed_routes_must_remain_explicit": True' in source


def test_dai_review_materializer_pairs_pristine_legacy_outputs() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "materialize_pristine_dai_review.py"
    source = script.read_text(encoding="utf-8")
    assert 'source_trajectory = run_dir / "trajectory_mjwp.npz"' in source
    assert 'scene = run_dir / "scene_act.xml"' in source
    assert 'legacy_scene = run_dir / "scene.xml"' in source
    assert "shutil.copy2(legacy_scene, scene)" in source


def test_dai_pristine_wrapper_forbids_guidance_tuning_and_materializer_records_grid() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "scripts" / "run_do_as_i_do_official_retarget.sh").read_text()
    materializer = (root / "scripts" / "materialize_pristine_dai_review.py").read_text()
    assert "DAI_*|MJWP_*|DO_AS_I_DO_*|ROBOT_RENDER_*" in wrapper
    assert "deprecated backend-tuning environment variable is forbidden" in wrapper
    assert "warmup_steps = int(config.get" in materializer
    assert "sim_dt = float(config.get" in materializer
    assert '"target_fps": target_fps' in materializer
    assert '"render_dt": 1.0 / target_fps' in materializer
    assert '"trace_dt": 1.0 / target_fps' in materializer
    assert '"sim_dt": sim_dt' in materializer


def test_cell_index_embeds_dai_alignment_evidence() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "index_cell_assets.py"
    source = script.read_text(encoding="utf-8")
    assert 'evidence["mjwp_alignment"] = load_json_optional(alignment_path)' in source


def test_dai_review_applies_fixed_hoi_quality_gate_after_pristine_backend() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_v4_two_full_pipelines.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        "--hoi-reference-contact-threshold 0.05",
        "--hoi-executed-contact-threshold 0.08",
        "--hoi-distance-median-threshold 0.03",
        "--hoi-distance-p95-threshold 0.08",
        "--hoi-min-reference-contact-frames 3",
        "--hoi-min-contact-retention 0.50",
    ):
        assert literal in source
    native_route = source.split("run_do_as_i_do_full() {", 1)[1].split(
        "\n}\n\nrun_adapted_do_as_i_do_retarget()", 1
    )[0]
    assert native_route.index(
        "scripts/run_do_as_i_do_official_retarget.sh"
    ) < native_route.index("materialize_and_validate_pristine_dai \\")


def test_dai_and_spider_wrappers_fail_closed_on_numerical_invalid() -> None:
    root = Path(__file__).resolve().parents[1]
    evaluator = (root / "scripts" / "diagnostics" / "evaluate_mjwp_object_tracking.py").read_text()
    dai = (root / "scripts" / "run_v4_two_full_pipelines.sh").read_text()
    spider = (root / "scripts" / "run_spider_retarget.sh").read_text()
    assert 'return 0 if result.get("status") == "ok" else 4' in evaluator
    assert "materialize_and_validate_pristine_dai()" in dai
    assert 'run_review_stage "${cell}_materialize_and_qc"' in dai
    assert "native DAI completed successfully; review/index/triptych remains unavailable" in dai
    assert '"native_returncode_authoritative": True' in (
        root / "scripts" / "reuse_v4_for_12_demos.py"
    ).read_text()
    assert "evaluate_mjwp_object_tracking" not in spider
    for literal in (
        "--pos-median-threshold 0.05",
        "--pos-p95-threshold 0.10",
        "--rot-median-threshold 0.75",
        "--rot-p95-threshold 2.0",
        "--lost-pos-threshold 0.10",
        "--lost-fraction-threshold 0.10",
        "--hoi-distance-median-threshold 0.03",
        "--hoi-distance-p95-threshold 0.08",
        "--hoi-min-contact-retention 0.50",
    ):
        assert literal in dai
    assert '"external_qc": {"role": "diagnostic_only"}' in (
        root / "scripts" / "reuse_v4_for_12_demos.py"
    ).read_text()


def test_spider_infers_adapter_from_exact_dai_cell() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "reuse_v4_for_12_demos.py").read_text()
    wrapper = (root / "scripts" / "run_spider_retarget.sh").read_text()
    assert "find_exact_spider_input_bundle(" in source
    assert 'adapter_manifest=Path(str(input_bundle["adapter_manifest"]))' in source
    assert '"--input-root",' in source
    assert "expected one exact task_info" in wrapper
