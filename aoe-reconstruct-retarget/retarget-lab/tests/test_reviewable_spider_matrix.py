from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from reuse_v4_for_12_demos import (  # noqa: E402
    assess_dai_native_spider_input,
    find_spider_robot,
)


def test_invalid_dai_object_pose_disables_both_spider_hand_sources() -> None:
    manifest = {
        "object_pose_quality": {
            "status": "invalid",
            "errors": ["object_pose_rotation_step_max_deg:176"],
        },
        "retarget_input_qc": {"status": "invalid", "errors": ["weak_visual_overlap"]},
    }
    for hand_source in ("aoe", "estimated"):
        result = assess_dai_native_spider_input(manifest, hand_source)
        assert result["status"] == "invalid"
        assert any("rotation_step_max" in error for error in result["errors"])


def test_aoe_hand_failure_does_not_disable_estimated_spider_hand() -> None:
    manifest = {
        "object_pose_quality": {"status": "ok", "errors": []},
        "retarget_input_qc": {"status": "invalid", "errors": ["weak_visual_overlap"]},
    }
    assert assess_dai_native_spider_input(manifest, "aoe")["status"] == "invalid"
    assert assess_dai_native_spider_input(manifest, "estimated")["status"] == "ok"


def test_invalid_tracking_keeps_original_aligned_video_for_review(tmp_path: Path) -> None:
    key = "traj_egoinfinity__hand_aoe__retarget_spider"
    root = tmp_path / "intermediates" / "retargeting" / "spider" / key
    root.mkdir(parents=True)
    video = root / "visualization_mjwp_act_aligned.mp4"
    trajectory = root / "trajectory_mjwp_act_aligned.npz"
    timing = root / "spider_timing.json"
    source = root / "dataset/processed/demo/robot/trajectory_mjwp_act.npz"
    scene = root / "dataset/processed/demo/robot/scene_act.xml"
    for path, payload in (
        (video, b"bound-aligned-video"),
        (trajectory, b"bound-aligned-trajectory"),
        (source, b"bound-source-trajectory"),
        (scene, b"<mujoco/>"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    fps = 30.0
    timing.write_text(
        json.dumps(
            {
                "source_frame_count": 2,
                "source_fps": fps,
                "ref_dt": 1.0 / fps,
                "sim_dt": 1.0 / 180.0,
            }
        ),
        encoding="utf-8",
    )

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (root / "spider_input_manifest.json").write_text(
        json.dumps({"status": "ok", "quality_valid": True}), encoding="utf-8"
    )
    (root / "spider_output_manifest.json").write_text(
        json.dumps(
            {
                "alignment": {
                    "applied": True,
                    "alignment_kind": "sample_exact_source_timestamps",
                    "source_kind": "original_spider_mjwp",
                    "exact_source_timestamps": True,
                    "sampling_semantics": "exact_source_timestamp_indices",
                    "complete_source_coverage": True,
                    "start_index": 0,
                    "warmup_steps": 0,
                    "source_frame_count": 2,
                    "target_count": 2,
                    "target_frames": 2,
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
                            "frame_count": 2,
                            "fps": fps,
                        },
                        "aligned_trajectory": {
                            "path": str(trajectory),
                            "sha256": sha256(trajectory),
                        },
                        "render_scene": {
                            "path": str(scene),
                            "sha256": sha256(scene),
                        },
                        "source_trajectory": {
                            "path": str(source),
                            "sha256": sha256(source),
                        },
                        "timing_manifest": {
                            "path": str(timing),
                            "sha256": sha256(timing),
                        },
                    },
                },
                "object_tracking_quality": {
                    "status": "invalid",
                    "metrics": {"lost_frame_fraction": 0.2},
                },
                "mjwp_fallback": {"applied": False},
            }
        ),
        encoding="utf-8",
    )
    assert find_spider_robot(tmp_path, key) is None
    assert find_spider_robot(
        tmp_path, key, allow_invalid_tracking_for_review=True
    ) == video


def test_spider_weak_input_is_recorded_instead_of_silently_promoted() -> None:
    source = (SCRIPTS / "run_spider_retarget.sh").read_text(encoding="utf-8")
    assert "expected one exact task_info" in source
    assert '"input_timebase_binding": "exact_dai_keypoint_frame_grid"' in source
    assert '"modified": False' in source
    assert "SPIDER_ALLOW_WEAK_INPUT" not in source
    assert "SPIDER_MJWP_FALLBACK" not in source
