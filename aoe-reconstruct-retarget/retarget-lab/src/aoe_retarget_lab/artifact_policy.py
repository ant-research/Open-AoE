from __future__ import annotations

from pathlib import Path


PLAIN_ALIGNED_ROBOT_NAMES = {
    "visualization_mjwp_act_aligned.mp4",
}

FORBIDDEN_ROBOT_RGB_PATTERNS = (
    "visualization_mjwp_rgb_overlay*.mp4",
    "visualization_kinematic_rgb_overlay*.mp4",
    "*robot*inpaint*.mp4",
    "*robot_on_rgb*.mp4",
    "*e2fgvi*.mp4",
    "*phantom*stage3*.mp4",
)

EXCLUDED_VISUAL_EVIDENCE = (
    "robot_on_rgb",
    "robot_inpainting",
    "visualization_mjwp_rgb_overlay",
    "reconstruction_overlay",
)


def forbidden_robot_rgb_artifacts(root: Path) -> list[Path]:
    found: set[Path] = set()
    for pattern in FORBIDDEN_ROBOT_RGB_PATTERNS:
        found.update(path for path in root.rglob(pattern) if path.is_file() or path.is_symlink())
    return sorted(found)
