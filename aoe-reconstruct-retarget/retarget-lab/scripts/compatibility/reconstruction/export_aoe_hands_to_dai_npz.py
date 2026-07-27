#!/usr/bin/env python3
"""Convert OpenAoE bimanual MANO tracks into do-as-i-do all_hand_meshes.npz."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def patch_legacy_numpy_and_chumpy() -> None:
    """Allow the legacy MANO pickle/chumpy stack to load on recent Python/Numpy."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    np.bool = bool  # type: ignore[attr-defined]
    np.int = int  # type: ignore[attr-defined]
    np.float = float  # type: ignore[attr-defined]
    np.complex = complex  # type: ignore[attr-defined]
    np.object = object  # type: ignore[attr-defined]
    np.unicode = str  # type: ignore[attr-defined]
    np.str = str  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--hands-npz", type=Path, default=None)
    parser.add_argument("--hawor-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--pose-space", choices=("camera", "world"), default="camera")
    parser.add_argument(
        "--camera-space-source",
        choices=("pred-cam", "hawor-transform", "auto"),
        default="pred-cam",
        help=(
            "When --pose-space camera, choose whether to use HaWoR's explicit "
            "pred_trans_cam/pred_rot_cam tracks or to transform world MANO "
            "tracks through R_c2w/t_c2w. 'auto' preserves the legacy behavior."
        ),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, int, int]:
    clip_dir = args.clip_dir.resolve()
    metadata_path = args.metadata or clip_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    hands_npz = args.hands_npz
    if hands_npz is None:
        dataset_root = metadata.get("dataset_root")
        segment = metadata.get("segment")
        segment_dir = metadata.get("segment_dir")
        if segment_dir:
            hands_npz = Path(segment_dir) / "ego_process/ego_hands_reconstruction/hands.npz"
        elif dataset_root and segment:
            hands_npz = (
                Path(dataset_root)
                / segment
                / "ego_process/ego_hands_reconstruction/hands.npz"
            )
        else:
            raise ValueError("Cannot infer --hands-npz; pass it explicitly.")

    start_frame = args.start_frame
    if start_frame is None:
        start_frame = int(metadata.get("start_frame", 0))

    num_frames = args.num_frames
    if num_frames is None:
        frames_dir = clip_dir / "all_frames"
        num_frames = len(sorted(frames_dir.glob("*.png")))
        if num_frames == 0:
            num_frames = int(metadata.get("end_frame", start_frame) - start_frame)
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames={num_frames}")

    output = args.output
    if output is None:
        output = clip_dir / "raw" / "all_hand_meshes.npz"

    return clip_dir, hands_npz.resolve(), output.resolve(), start_frame, num_frames


def make_faces(base_faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Same wrist-cap triangles as reconstruction/modules/HaWoR/demo.py.
    faces_new = np.array(
        [
            [92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79],
        ],
        dtype=np.int32,
    )
    faces_right = np.concatenate([base_faces.astype(np.int32), faces_new], axis=0)
    faces_left = faces_right[:, [0, 2, 1]].astype(np.int32)
    return faces_left, faces_right


def hawor_camera_transform(hands: np.lib.npyio.NpzFile, frame_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the camera transform used by HaWoR's own visualization/export.

    HaWoR first flips the MANO/world convention with R_x, then folds that same
    flip into the SLAM camera pose before transforming hands into camera space.
    Skipping this step keeps the projected hand center roughly plausible but
    mirrors/rotates the mesh topology in RGB overlays.
    """
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    r_c2w = np.asarray(hands["R_c2w"][frame_ids], dtype=np.float32)
    t_c2w = np.asarray(hands["t_c2w"][frame_ids], dtype=np.float32)
    r_c2w = np.einsum("ij,njk->nik", rx, r_c2w)
    t_c2w = np.einsum("ij,nj->ni", rx, t_c2w)
    r_w2c = np.transpose(r_c2w, (0, 2, 1))
    t_w2c = -np.einsum("nij,nj->ni", r_w2c, t_c2w)
    return r_w2c.astype(np.float32), t_w2c.astype(np.float32)


def apply_hawor_camera_transform(
    points: np.ndarray,
    r_w2c: np.ndarray,
    t_w2c: np.ndarray,
) -> np.ndarray:
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    points_rx = np.einsum("ij,nvj->nvi", rx, points)
    return (np.einsum("nij,nvj->nvi", r_w2c, points_rx) + t_w2c[:, None]).astype(np.float32)


def transform_hawor_trans(
    trans: np.ndarray,
    r_w2c: np.ndarray,
    t_w2c: np.ndarray,
) -> np.ndarray:
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    trans_rx = np.einsum("ij,nj->ni", rx, trans)
    return (np.einsum("nij,nj->ni", r_w2c, trans_rx) + t_w2c).astype(np.float32)


def transform_hawor_rot(rotvec: np.ndarray, r_w2c: np.ndarray) -> np.ndarray:
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    mano_r = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    world_r = np.einsum("ij,njk->nik", rx, mano_r)
    camera_r = np.einsum("nij,njk->nik", np.asarray(r_w2c, dtype=np.float64), world_r)
    return Rotation.from_matrix(camera_r).as_rotvec().astype(np.float32)


def main() -> None:
    args = parse_args()
    patch_legacy_numpy_and_chumpy()
    _, hands_npz, output, start_frame, num_frames = resolve_inputs(args)

    hawor_dir = args.hawor_dir.resolve()
    sys.path.insert(0, str(hawor_dir))
    os.chdir(hawor_dir)

    import torch  # noqa: WPS433
    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left  # noqa: WPS433

    hands = np.load(hands_npz)
    frame_ids = np.arange(start_frame, start_frame + num_frames, dtype=np.int64)
    if frame_ids[-1] >= hands["pred_hand_pose"].shape[1]:
        raise IndexError(
            f"Requested frame {int(frame_ids[-1])}, "
            f"but hands file has {hands['pred_hand_pose'].shape[1]} frames."
        )

    have_world_camera_pose = (
        "R_c2w" in hands
        and "t_c2w" in hands
        and "pred_rot" in hands
        and "pred_trans" in hands
    )
    if args.pose_space == "camera":
        use_hawor_camera_space = args.camera_space_source in {"hawor-transform", "auto"} and have_world_camera_pose
        if args.camera_space_source == "hawor-transform" and not have_world_camera_pose:
            raise KeyError("requested --camera-space-source hawor-transform but R_c2w/t_c2w/pred_rot/pred_trans are missing")
        if not use_hawor_camera_space and ("pred_rot_cam" not in hands or "pred_trans_cam" not in hands):
            raise KeyError("camera-space hand tracks require pred_rot_cam and pred_trans_cam")
        rot_key = "pred_rot" if use_hawor_camera_space else "pred_rot_cam"
        trans_key = "pred_trans" if use_hawor_camera_space else "pred_trans_cam"
    else:
        use_hawor_camera_space = False
        rot_key = "pred_rot"
        trans_key = "pred_trans"
    use_cuda = args.device == "cuda" and torch.cuda.is_available()

    def tensor(key: str, hand_index: int) -> torch.Tensor:
        return torch.as_tensor(hands[key][hand_index : hand_index + 1, frame_ids], dtype=torch.float32)

    left_idx = 0
    right_idx = 1
    with torch.no_grad():
        left = run_mano_left(
            tensor(trans_key, left_idx),
            tensor(rot_key, left_idx),
            tensor("pred_hand_pose", left_idx),
            betas=tensor("pred_betas", left_idx),
            use_cuda=use_cuda,
        )
        right = run_mano(
            tensor(trans_key, right_idx),
            tensor(rot_key, right_idx),
            tensor("pred_hand_pose", right_idx),
            betas=tensor("pred_betas", right_idx),
            use_cuda=use_cuda,
        )

    faces_left, faces_right = make_faces(get_mano_faces())
    left_vertices = left["vertices"][0].detach().cpu().numpy().astype(np.float32)
    right_vertices = right["vertices"][0].detach().cpu().numpy().astype(np.float32)
    left_joints = left["joints"][0].detach().cpu().numpy().astype(np.float32)
    right_joints = right["joints"][0].detach().cpu().numpy().astype(np.float32)
    left_trans = hands[trans_key][left_idx, frame_ids].astype(np.float32)
    right_trans = hands[trans_key][right_idx, frame_ids].astype(np.float32)
    left_rot = hands[rot_key][left_idx, frame_ids].astype(np.float32)
    right_rot = hands[rot_key][right_idx, frame_ids].astype(np.float32)

    camera_space_source = "direct_pred_cam"
    if use_hawor_camera_space:
        r_w2c, t_w2c = hawor_camera_transform(hands, frame_ids)
        left_vertices = apply_hawor_camera_transform(left_vertices, r_w2c, t_w2c)
        right_vertices = apply_hawor_camera_transform(right_vertices, r_w2c, t_w2c)
        left_joints = apply_hawor_camera_transform(left_joints, r_w2c, t_w2c)
        right_joints = apply_hawor_camera_transform(right_joints, r_w2c, t_w2c)
        left_trans = transform_hawor_trans(left_trans, r_w2c, t_w2c)
        right_trans = transform_hawor_trans(right_trans, r_w2c, t_w2c)
        left_rot = transform_hawor_rot(left_rot, r_w2c)
        right_rot = transform_hawor_rot(right_rot, r_w2c)
        camera_space_source = "hawor_demo_Rx_camera_transform"

    payload = {
        "left_vertices": left_vertices,
        "left_faces": faces_left,
        "left_joints": left_joints,
        "right_vertices": right_vertices,
        "right_faces": faces_right,
        "right_joints": right_joints,
        "left_trans": left_trans,
        "left_rot": left_rot,
        "left_hand_pose": hands["pred_hand_pose"][left_idx, frame_ids].astype(np.float32),
        "left_betas": hands["pred_betas"][left_idx, frame_ids].astype(np.float32),
        "left_valid": hands["pred_valid"][left_idx, frame_ids].astype(np.float32),
        "right_trans": right_trans,
        "right_rot": right_rot,
        "right_hand_pose": hands["pred_hand_pose"][right_idx, frame_ids].astype(np.float32),
        "right_betas": hands["pred_betas"][right_idx, frame_ids].astype(np.float32),
        "right_valid": hands["pred_valid"][right_idx, frame_ids].astype(np.float32),
        "source_frame_ids": frame_ids.astype(np.int32),
        "source_hands_npz": str(hands_npz),
        "pose_space": args.pose_space,
        "hand_index_assumption": "hands.npz dim0=left, dim1=right",
        "camera_space_source": camera_space_source,
        "camera_space_mode": args.camera_space_source if args.pose_space == "camera" else "world",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    summary = {
        "output": str(output),
        "source": str(hands_npz),
        "frames": num_frames,
        "source_frame_range": [int(frame_ids[0]), int(frame_ids[-1])],
        "pose_space": args.pose_space,
        "left_vertices": list(left_vertices.shape),
        "right_vertices": list(right_vertices.shape),
        "left_valid_mean": float(payload["left_valid"].mean()),
        "right_valid_mean": float(payload["right_valid"].mean()),
        "device": "cuda" if use_cuda else "cpu",
        "camera_space_source": camera_space_source,
        "camera_space_mode": payload["camera_space_mode"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
