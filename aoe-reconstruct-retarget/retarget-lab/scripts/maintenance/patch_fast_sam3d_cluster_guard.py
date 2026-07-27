#!/usr/bin/env python3
"""Install a small stability guard into Fast-SAM3D pose clustering.

The upstream tracker treats ``cluster_min_size`` as advisory: when every pose
candidate is a singleton it falls back to the highest-scoring singleton and
continues chain propagation.  For heavily occluded HOI clips this can introduce
one-frame orientation flips that later become catastrophic retargeting poses.

This patch keeps normal clustered frames unchanged.  It only changes the
no-consensus case by selecting the candidate closest to the previous chained
pose target instead of the isolated best render score.
"""

from __future__ import annotations

import argparse
from pathlib import Path


CLUSTER_SENTINEL = "AOE_CLUSTER_GUARD"
CLUSTER_V2_SENTINEL = "AOE_LOW_CONF_CLUSTER_GUARD"
CLUSTER_V3_SENTINEL = "AOE_INVALID_POSE_CANDIDATE_GUARD"
CLUSTER_V4_SENTINEL = "AOE_SMALL_CLUSTER_METADATA_GUARD"
POSTOPT_SENTINEL = "AOE_POSTOPT_CONTINUITY_GUARD"
POSTOPT_V2_SENTINEL = "AOE_POSTOPT_INVALID_POSE_GUARD"

OLD = '''            best_idx, cluster_info = cluster_pose_candidates(
                all_samples,
                scoring_metric=scoring_metric,
                dist_thresh=cluster_dist_thresh,
                min_cluster_size=cluster_min_size,
                w_trans=cluster_w_trans,
                w_rot=cluster_w_rot,
            )
            best_result = all_samples[best_idx]
            best_result["cluster_info"] = cluster_info
            logger.info(
                f"  Cluster selection: {cluster_info['num_clusters']} clusters, "
                f"sizes={cluster_info['cluster_sizes']}, "
                f"selected cluster size={cluster_info['selected_cluster_size']}, "
                f"best candidate idx={best_idx} "
                f"(seed={best_result['sample_seed']}, "
                f"score={cluster_info['best_candidate_score']:.4f})"
            )
'''

CLUSTER_GUARD_V1 = '''            best_idx, cluster_info = cluster_pose_candidates(
                all_samples,
                scoring_metric=scoring_metric,
                dist_thresh=cluster_dist_thresh,
                min_cluster_size=cluster_min_size,
                w_trans=cluster_w_trans,
                w_rot=cluster_w_rot,
            )
            # AOE_CLUSTER_GUARD: if clustering has no stable consensus, do not
            # let an isolated high render-IoU sample become the next chained pose.
            if (
                cluster_info.get("selected_cluster_size", 0) < cluster_min_size
                and pose_target is not None
                and len(all_samples) > 1
            ):
                pose_target_for_dist = {
                    k_: pose_target[k_].to(all_samples[0][k_].device)
                    for k_ in ("translation", "rotation", "scale")
                    if k_ in pose_target and k_ in all_samples[0]
                }
                if {"translation", "rotation"}.issubset(pose_target_for_dist):
                    stable_scores = [
                        _pose_distance(sample, pose_target_for_dist, cluster_w_trans, cluster_w_rot)
                        for sample in all_samples
                    ]
                    stable_idx = int(np.argmin(stable_scores))
                    cluster_info["unstable_cluster_fallback"] = "closest_to_pose_target"
                    cluster_info["fallback_from_best_idx"] = int(best_idx)
                    cluster_info["fallback_pose_distance"] = float(stable_scores[stable_idx])
                    best_idx = stable_idx
            best_result = all_samples[best_idx]
            best_result["cluster_info"] = cluster_info
            logger.info(
                f"  Cluster selection: {cluster_info['num_clusters']} clusters, "
                f"sizes={cluster_info['cluster_sizes']}, "
                f"selected cluster size={cluster_info['selected_cluster_size']}, "
                f"best candidate idx={best_idx} "
                f"(seed={best_result['sample_seed']}, "
                f"score={cluster_info['best_candidate_score']:.4f})"
            )
'''

NEW = '''            best_idx, cluster_info = cluster_pose_candidates(
                all_samples,
                scoring_metric=scoring_metric,
                dist_thresh=cluster_dist_thresh,
                min_cluster_size=cluster_min_size,
                w_trans=cluster_w_trans,
                w_rot=cluster_w_rot,
            )
            # AOE_CLUSTER_GUARD / AOE_LOW_CONF_CLUSTER_GUARD: if clustering has
            # no stable consensus, or if the selected cluster has low render
            # confidence, do not let an isolated/highly drifting sample become
            # the next chained pose.
            if pose_target is not None and len(all_samples) > 1:
                pose_target_for_dist = {
                    k_: pose_target[k_].to(all_samples[0][k_].device)
                    for k_ in ("translation", "rotation", "scale")
                    if k_ in pose_target and k_ in all_samples[0]
                }
                if {"translation", "rotation"}.issubset(pose_target_for_dist):
                    sample_scores = []
                    for sample in all_samples:
                        if scoring_metric in sample:
                            sample_scores.append(float(sample[scoring_metric]))
                        elif "shape_iou" in sample:
                            sample_scores.append(float(sample["shape_iou"]))
                        else:
                            sample_scores.append(0.0)
                    stable_scores = [
                        _pose_distance(sample, pose_target_for_dist, cluster_w_trans, cluster_w_rot)
                        for sample in all_samples
                    ]
                    selected_size = cluster_info.get("selected_cluster_size", 0)
                    selected_avg = float(cluster_info.get("selected_cluster_avg_score", 0.0))
                    low_conf_score = float(os.environ.get("AOE_CLUSTER_LOW_CONF_SCORE", "0.45"))
                    score_margin = float(os.environ.get("AOE_CLUSTER_LOW_CONF_SCORE_MARGIN", "0.20"))
                    score_weight = float(os.environ.get("AOE_CLUSTER_LOW_CONF_SCORE_WEIGHT", "0.35"))
                    max_score = max(sample_scores)
                    should_fallback = selected_size < cluster_min_size
                    low_conf_fallback = selected_avg < low_conf_score
                    if should_fallback or low_conf_fallback:
                        eligible = [
                            i for i, score in enumerate(sample_scores)
                            if score >= max_score - score_margin
                        ]
                        if not eligible:
                            eligible = list(range(len(all_samples)))
                        composite = [
                            (
                                stable_scores[i]
                                + score_weight * ((max_score - sample_scores[i]) / max(max_score, 1e-6)),
                                i,
                            )
                            for i in eligible
                        ]
                        stable_idx = int(min(composite)[1])
                        if stable_idx != best_idx:
                            cluster_info["unstable_cluster_fallback"] = (
                                "closest_to_pose_target" if should_fallback
                                else "low_confidence_temporal_score"
                            )
                            cluster_info["fallback_from_best_idx"] = int(best_idx)
                            cluster_info["fallback_pose_distance"] = float(stable_scores[stable_idx])
                            cluster_info["fallback_score"] = float(sample_scores[stable_idx])
                            cluster_info["fallback_max_score"] = float(max_score)
                            cluster_info["fallback_selected_avg_score"] = float(selected_avg)
                            best_idx = stable_idx
            best_result = all_samples[best_idx]
            best_result["cluster_info"] = cluster_info
            logger.info(
                f"  Cluster selection: {cluster_info['num_clusters']} clusters, "
                f"sizes={cluster_info['cluster_sizes']}, "
                f"selected cluster size={cluster_info['selected_cluster_size']}, "
                f"best candidate idx={best_idx} "
                f"(seed={best_result['sample_seed']}, "
                f"score={cluster_info['best_candidate_score']:.4f})"
            )
            if cluster_info.get("unstable_cluster_fallback"):
                logger.info(
                    "  Cluster fallback: "
                    f"{cluster_info['unstable_cluster_fallback']} "
                    f"from idx={cluster_info.get('fallback_from_best_idx')} "
                    f"to idx={best_idx}, "
                    f"score={cluster_info.get('fallback_score')}, "
                    f"pose_dist={cluster_info.get('fallback_pose_distance')}"
                )
'''

INVALID_CANDIDATE_GUARD = '''            # AOE_INVALID_POSE_CANDIDATE_GUARD: diffusion can emit a
            # zero-norm quaternion or non-finite pose for an otherwise usable
            # frame. Reject only those samples before clustering; never repair
            # them into an identity pose because that would fabricate motion.
            valid_samples = []
            invalid_sample_seeds = []
            for sample in all_samples:
                required = [sample.get(k_) for k_ in ("translation", "rotation", "scale")]
                sample_valid = all(value is not None for value in required)
                if sample_valid:
                    sample_valid = all(bool(torch.isfinite(value).all()) for value in required)
                if sample_valid:
                    rotation = sample["rotation"].reshape(-1)
                    sample_valid = rotation.numel() == 4 and float(torch.linalg.vector_norm(rotation)) > 1e-8
                if sample_valid:
                    valid_samples.append(sample)
                else:
                    invalid_sample_seeds.append(sample.get("sample_seed"))
            invalid_pose_candidates_rejected = len(all_samples) - len(valid_samples)
            if invalid_pose_candidates_rejected:
                logger.warning(
                    "  Rejected invalid pose candidates before clustering: "
                    f"count={invalid_pose_candidates_rejected}, seeds={invalid_sample_seeds}"
                )
            if not valid_samples:
                raise RuntimeError("all pose candidates are non-finite or have invalid quaternions")
            all_samples = valid_samples
'''

SMALL_CLUSTER_METADATA_GUARD = '''            # AOE_SMALL_CLUSTER_METADATA_GUARD: upstream returns abbreviated
            # metadata for K <= 2, while the
            # common logging and temporal fallback path expects these fields.
            if "selected_cluster_size" not in cluster_info:
                valid_scores = []
                for sample in all_samples:
                    if scoring_metric in sample:
                        valid_scores.append(float(sample[scoring_metric]))
                    elif "shape_iou" in sample:
                        valid_scores.append(float(sample["shape_iou"]))
                    else:
                        valid_scores.append(0.0)
                cluster_info["selected_cluster_size"] = len(all_samples)
                cluster_info["selected_cluster_avg_score"] = float(np.mean(valid_scores))
                cluster_info["best_candidate_score"] = float(valid_scores[best_idx])
'''

NEW_V3 = NEW.replace(
    "            best_idx, cluster_info = cluster_pose_candidates(\n",
    INVALID_CANDIDATE_GUARD + "            best_idx, cluster_info = cluster_pose_candidates(\n",
    1,
).replace(
    "            best_result = all_samples[best_idx]\n",
    "            cluster_info[\"invalid_pose_candidates_rejected\"] = invalid_pose_candidates_rejected\n"
    "            cluster_info[\"invalid_pose_candidate_seeds\"] = invalid_sample_seeds\n"
    "            best_result = all_samples[best_idx]\n",
    1,
)

NEW_V4 = NEW_V3.replace(
    '''                w_rot=cluster_w_rot,
            )
            # AOE_CLUSTER_GUARD / AOE_LOW_CONF_CLUSTER_GUARD: if clustering has
''',
    '''                w_rot=cluster_w_rot,
            )
''' + SMALL_CLUSTER_METADATA_GUARD + '''            # AOE_CLUSTER_GUARD / AOE_LOW_CONF_CLUSTER_GUARD: if clustering has
''',
    1,
)

OLD_POSTOPT = '''            if "iou" in postopt:
                best_result["post_opt_iou"] = postopt["iou"]
        except Exception as e:
            logger.warning(f"Post-optimization failed: {e}")

        # Hard override: ensure fixed_scale is the final scale
'''

NEW_POSTOPT = '''            if "iou" in postopt:
                best_result["post_opt_iou"] = postopt["iou"]
        except Exception as e:
            logger.warning(f"Post-optimization failed: {e}")

        # AOE_POSTOPT_CONTINUITY_GUARD: render-compare is mask-only and can
        # accept a pose with a large 3D rotation/depth jump.  Keep the clustered
        # diffusion pose unless post-opt stays close to it or improves temporal
        # continuity relative to the chained pose target.
        if "post_opt_rotation" in best_result:
            try:
                post_pose = {
                    "rotation": best_result["post_opt_rotation"],
                    "translation": best_result["post_opt_translation"],
                    "scale": best_result["post_opt_scale"],
                }
                diffusion_pose = {
                    "rotation": best_result["rotation"],
                    "translation": best_result["translation"],
                    "scale": best_result["scale"],
                }
                post_diff_dist = _pose_distance(
                    post_pose, diffusion_pose, cluster_w_trans, cluster_w_rot
                )
                post_target_dist = None
                diffusion_target_dist = None
                if pose_target is not None:
                    pose_target_for_dist = {
                        k_: pose_target[k_].to(best_result[k_].device)
                        for k_ in ("translation", "rotation", "scale")
                        if k_ in pose_target and k_ in best_result
                    }
                    if {"translation", "rotation"}.issubset(pose_target_for_dist):
                        post_target_dist = _pose_distance(
                            post_pose, pose_target_for_dist, cluster_w_trans, cluster_w_rot
                        )
                        diffusion_target_dist = _pose_distance(
                            diffusion_pose, pose_target_for_dist, cluster_w_trans, cluster_w_rot
                        )
                max_post_diff = float(os.environ.get("AOE_POSTOPT_MAX_DIFFUSION_DIST", "0.65"))
                min_target_gain = float(os.environ.get("AOE_POSTOPT_MIN_TARGET_GAIN", "0.10"))
                improves_target = (
                    post_target_dist is not None
                    and diffusion_target_dist is not None
                    and post_target_dist <= diffusion_target_dist - min_target_gain
                )
                if post_diff_dist > max_post_diff and not improves_target:
                    for k_ in ("translation", "rotation", "scale"):
                        best_result.pop(f"post_opt_{k_}", None)
                    best_result["post_opt_rejected"] = "pose_continuity"
                    best_result["post_opt_diffusion_distance"] = float(post_diff_dist)
                    if post_target_dist is not None:
                        best_result["post_opt_target_distance"] = float(post_target_dist)
                        best_result["diffusion_target_distance"] = float(diffusion_target_dist)
                    logger.info(
                        "Post-opt rejected by continuity guard: "
                        f"post_diff={post_diff_dist:.3f}, "
                        f"target_diff={post_target_dist}, "
                        f"diffusion_target={diffusion_target_dist}"
                    )
            except Exception as e:
                logger.warning(f"Post-opt continuity guard failed: {e}")

        # Hard override: ensure fixed_scale is the final scale
'''

NEW_POSTOPT_V2 = NEW_POSTOPT.replace(
    '''            except Exception as e:
                logger.warning(f"Post-opt continuity guard failed: {e}")

        # Hard override: ensure fixed_scale is the final scale
''',
    '''            except Exception as e:
                # AOE_POSTOPT_INVALID_POSE_GUARD: an invalid post-opt pose must
                # not survive merely because the continuity check raised.
                for k_ in ("translation", "rotation", "scale"):
                    best_result.pop(f"post_opt_{k_}", None)
                best_result["post_opt_rejected"] = "invalid_pose"
                logger.warning(f"Post-opt rejected as invalid pose: {e}")

        # Hard override: ensure fixed_scale is the final scale
''',
    1,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-object", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = args.track_object
    text = path.read_text(encoding="utf-8")
    changed = False
    patched = text

    if CLUSTER_V4_SENTINEL in patched:
        print(f"[cluster-guard] already installed: {path}")
    elif CLUSTER_V3_SENTINEL in patched:
        if NEW_V3 in patched:
            patched = patched.replace(NEW_V3, NEW_V4, 1)
            changed = True
        else:
            raise SystemExit(f"[cluster-guard] v3 guard found but replacement block not found in {path}")
    elif CLUSTER_V2_SENTINEL in patched:
        if NEW in patched:
            patched = patched.replace(NEW, NEW_V4, 1)
            changed = True
        else:
            raise SystemExit(f"[cluster-guard] v2 guard found but replacement block not found in {path}")
    elif CLUSTER_SENTINEL in patched:
        if CLUSTER_GUARD_V1 in patched:
            patched = patched.replace(CLUSTER_GUARD_V1, NEW_V4, 1)
            changed = True
        else:
            raise SystemExit(f"[cluster-guard] v1 guard found but replacement block not found in {path}")
    elif OLD in patched:
        patched = patched.replace(OLD, NEW_V4, 1)
        changed = True
    else:
        raise SystemExit(f"[cluster-guard] expected selection block not found in {path}")

    if POSTOPT_V2_SENTINEL in patched:
        print(f"[postopt-guard] already installed: {path}")
    elif POSTOPT_SENTINEL in patched:
        if NEW_POSTOPT in patched:
            patched = patched.replace(NEW_POSTOPT, NEW_POSTOPT_V2, 1)
            changed = True
        else:
            raise SystemExit(f"[postopt-guard] v1 guard found but replacement block not found in {path}")
    elif OLD_POSTOPT in patched:
        patched = patched.replace(OLD_POSTOPT, NEW_POSTOPT_V2, 1)
        changed = True
    else:
        raise SystemExit(f"[postopt-guard] expected post-opt block not found in {path}")

    if args.dry_run:
        action = "would patch" if changed else "already current"
        print(f"[fast-sam3d-guards] {action}: {path}")
        return 0
    if not changed:
        print(f"[fast-sam3d-guards] already current: {path}")
        return 0

    backup = path.with_suffix(path.suffix + ".aoe_cluster_guard.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")
    print(f"[fast-sam3d-guards] installed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
