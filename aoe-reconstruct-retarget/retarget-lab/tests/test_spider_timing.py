from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import resolve_spider_timing as timing_module  # noqa: E402
from resolve_spider_timing import build_timing  # noqa: E402


class SpiderTimingRegressionTests(unittest.TestCase):
    def test_reused_probe_reads_exact_route_raw_video_fps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            experiment = repo / "experiments" / "probe"
            cell = (
                experiment
                / "intermediates/retargeting/do_as_i_do"
                / "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
            )
            keypoints = (
                cell
                / "retargeting_outputs/mano/left/task/0/trajectory_keypoints.npz"
            )
            keypoints.parent.mkdir(parents=True)
            np.savez(
                keypoints,
                qpos_wrist_left=np.zeros((68, 7), dtype=np.float32),
            )
            raw_video = cell / "raw_dir/raw.mp4"
            raw_video.parent.mkdir(parents=True)
            raw_video.write_bytes(b"ffprobe fixture")
            completed = subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "streams": [
                            {
                                "avg_frame_rate": "30/1",
                                "r_frame_rate": "30/1",
                            }
                        ]
                    }
                ),
                stderr="",
            )
            with mock.patch.object(
                timing_module.subprocess, "run", return_value=completed
            ) as run:
                timing = build_timing(
                    repo_root=repo,
                    experiment_root=experiment,
                    keypoints=keypoints,
                    sim_dt=0.0,
                    requested_max_sim_steps=-1,
                )

            self.assertEqual(timing["source_fps"], 30.0)
            self.assertEqual(timing["required_sim_steps"], 402)
            self.assertTrue(
                timing["source_fps_provenance"].startswith(
                    "ffprobe_exact_route_raw_video:"
                )
            )
            self.assertEqual(run.call_args.args[0][-1], str(raw_video.resolve()))

    def test_wrapper_binds_render_and_trace_to_source_grid(self) -> None:
        wrapper = (SCRIPTS / "run_spider_retarget.sh").read_text(encoding="utf-8")
        self.assertIn('--ref-dt "$ref_dt"', wrapper)
        self.assertIn('"ref_dt=$ref_dt"', wrapper)
        self.assertIn('"input_timebase_binding": "exact_dai_keypoint_frame_grid"', wrapper)

    def test_required_video_fps_rejects_missing_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.mp4"
            with self.assertRaisesRegex(FileNotFoundError, "source video is missing"):
                timing_module.required_video_fps(missing)

    def test_required_video_fps_rejects_ffprobe_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "bad.mp4"
            video.write_bytes(b"not a video")
            failure = subprocess.CalledProcessError(1, ["ffprobe", str(video)])
            with mock.patch.object(
                timing_module.subprocess, "run", side_effect=failure
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    timing_module.required_video_fps(video)

    def test_pristine_dai_wrapper_contains_no_timing_or_review_mutation(self) -> None:
        wrapper = (SCRIPTS / "run_do_as_i_do_official_retarget.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("Native files are never rewritten", wrapper)
        self.assertIn("backend_rc_authoritative", wrapper)
        self.assertIn("DAI_*|MJWP_*|DO_AS_I_DO_*|ROBOT_RENDER_*", wrapper)
        self.assertNotIn("DAI_MJWP_RENDER_DT=", wrapper)
        self.assertNotIn("patch_dai_sampling_nonfinite_rewards.py", wrapper)
        self.assertNotIn("trajectory_mjwp_act_aligned.npz", wrapper)

    def test_dai_materializer_uses_exact_source_grid_without_fps_guess(self) -> None:
        source = (SCRIPTS / "materialize_pristine_dai_review.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"-count_frames"', source)
        self.assertIn('"stream=nb_read_frames,avg_frame_rate,r_frame_rate"', source)
        self.assertIn("target_count = int(probe", source)
        self.assertIn("target_fps = float(probe", source)
        self.assertIn("target_time = start_time + np.arange", source)
        self.assertIn('"sampling_semantics": "exact_source_timestamp_interpolation"', source)
        self.assertNotIn('print("30")', source)

    def test_dai_materializer_hash_binds_source_and_review_artifacts(self) -> None:
        source = (SCRIPTS / "materialize_pristine_dai_review.py").read_text(
            encoding="utf-8"
        )
        for binding in (
            '"aligned_video": {',
            '"aligned_trajectory": {',
            '"render_scene": {',
            '"source_trajectory": {',
            '"source_video": {',
        ):
            self.assertIn(binding, source)
        self.assertGreaterEqual(source.count('"sha256": sha256('), 5)
        self.assertIn("if rendered != probe:", source)

    def test_dai_materializer_installs_manifest_after_render_validation(self) -> None:
        source = (SCRIPTS / "materialize_pristine_dai_review.py").read_text(
            encoding="utf-8"
        )
        render = source.index("subprocess.run(")
        validate = source.index("if rendered != probe:", render)
        publish = source.index("alignment_manifest.write_text", validate)
        self.assertLess(render, validate)
        self.assertLess(validate, publish)

    def test_dai_review_is_route_scoped_and_gated_before_index(self) -> None:
        runner = (SCRIPTS / "run_v4_two_full_pipelines.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'retarget_output_root="$exp/intermediates/retargeting/do_as_i_do/$cell/'
            'retargeting_outputs"',
            runner,
        )
        self.assertIn('"${cell}_materialize_and_qc"', runner)
        self.assertLess(
            runner.index('"${cell}_materialize_and_qc"'),
            runner.index('"${cell}_index"'),
        )
        self.assertIn("native DAI completed successfully; adapted-route review/index", runner)

    def test_production_matrix_dai_callers_create_fresh_pristine_archives(self) -> None:
        runner = (SCRIPTS / "run_v4_two_full_pipelines.sh").read_text(
            encoding="utf-8"
        )
        marker = "scripts/run_do_as_i_do_official_retarget.sh"
        calls = runner.split(marker)[1:]
        self.assertEqual(len(calls), 2)
        for call in calls:
            launch = call.split("|| launcher_rc=$?", 1)[0]
            with self.subTest(call=launch[:160]):
                self.assertIn("--output-root-dir", launch)
                self.assertIn("--no-wait", launch)
                self.assertNotIn("--force", launch)
                self.assertNotIn("--max-sim-steps", launch)

    def test_spider_wrapper_requires_fresh_rollout_and_exact_render_grid(self) -> None:
        wrapper = (SCRIPTS / "run_spider_retarget.sh").read_text(encoding="utf-8")

        self.assertIn("output-dir must be a new path", wrapper)
        self.assertIn("run_backend_stage generate_xml", wrapper)
        self.assertIn("run_backend_stage ik_fast", wrapper)
        self.assertIn("run_backend_stage run_mjwp", wrapper)
        self.assertIn('"save_video=true"', wrapper)
        self.assertIn('"input_timebase_binding": "exact_dai_keypoint_frame_grid"', wrapper)
        self.assertIn('"scope": "pristine_official_spider_native_returncode"', wrapper)


def test_matrix_timing_uses_source_clip_fps_and_expands_simulation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    matrix = repo / "experiments" / "matrix_run"
    source = repo / "experiments" / "source_run"
    metadata = source / "intermediates/trajectory_6dof/do_as_i_do/reconstruction/clip/metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"source_fps": 30.0}), encoding="utf-8")
    matrix.mkdir(parents=True)
    (matrix / "reuse_12_demo_manifest.json").write_text(
        json.dumps({"source_run": "source_run"}), encoding="utf-8"
    )
    keypoints = source / "trajectory_keypoints.npz"
    np.savez(keypoints, qpos_wrist_left=np.zeros((98, 7), dtype=np.float32))

    timing = build_timing(
        repo_root=repo,
        experiment_root=matrix,
        keypoints=keypoints,
        sim_dt=0.0,
        requested_max_sim_steps=-1,
    )

    assert timing["source_frame_count"] == 98
    assert timing["source_fps"] == 30.0
    assert timing["ref_dt"] == 1.0 / 30.0
    assert timing["required_sim_steps"] == 582
    assert timing["reference_sample_count"] == 583
    assert timing["effective_max_sim_steps"] == 582
    assert timing["max_sim_steps_expanded"] is True
    assert timing["substeps_per_frame"] == 6
    assert timing["sim_dt"] == (1.0 / 30.0) / 6
    assert timing["simulation_duration_sec"] == timing["target_duration_sec"]


def test_exact_explicit_simulation_budget_is_preserved(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    experiment = repo / "experiments" / "run"
    experiment.mkdir(parents=True)
    keypoints = experiment / "trajectory_keypoints.npz"
    np.savez(keypoints, qpos_obj_right=np.zeros((61, 7), dtype=np.float32))

    timing = build_timing(
        repo_root=repo,
        experiment_root=experiment,
        keypoints=keypoints,
        sim_dt=(1.0 / 30.0) / 6,
        requested_max_sim_steps=360,
        explicit_source_fps=30.0,
    )

    assert timing["required_sim_steps"] == 360
    assert timing["reference_sample_count"] == 361
    assert timing["effective_max_sim_steps"] == 360
    assert timing["max_sim_steps_expanded"] is False


def test_non_integer_sim_ratio_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    experiment = repo / "experiments" / "run"
    experiment.mkdir(parents=True)
    keypoints = experiment / "trajectory_keypoints.npz"
    np.savez(keypoints, qpos_obj_right=np.zeros((61, 7), dtype=np.float32))

    try:
        build_timing(
            repo_root=repo,
            experiment_root=experiment,
            keypoints=keypoints,
            sim_dt=0.005,
            requested_max_sim_steps=-1,
            explicit_source_fps=30.0,
        )
    except ValueError as exc:
        assert "must be an integer" in str(exc)
    else:
        raise AssertionError("non-integer ref_dt/sim_dt ratio was accepted")


def test_reused_probe_falls_back_to_exact_route_raw_video(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    experiment = repo / "experiments" / "probe"
    cell = (
        experiment
        / "intermediates/retargeting/do_as_i_do"
        / "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    )
    keypoints = cell / "retargeting_outputs/mano/left/task/0/trajectory_keypoints.npz"
    keypoints.parent.mkdir(parents=True)
    np.savez(keypoints, qpos_wrist_left=np.zeros((68, 7), dtype=np.float32))
    raw_video = cell / "raw_dir/raw.mp4"
    raw_video.parent.mkdir(parents=True)
    raw_video.write_bytes(b"ffprobe fixture")

    def fake_run(command, **kwargs):
        assert command[-1] == str(raw_video)
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {"streams": [{"avg_frame_rate": "30/1", "r_frame_rate": "30/1"}]}
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    timing = build_timing(
        repo_root=repo,
        experiment_root=experiment,
        keypoints=keypoints,
        sim_dt=0.0,
        requested_max_sim_steps=-1,
    )

    assert timing["source_fps"] == 30.0
    assert timing["source_frame_count"] == 68
    assert timing["required_sim_steps"] == 402
    assert timing["source_fps_provenance"].startswith(
        "ffprobe_exact_route_raw_video:"
    )
