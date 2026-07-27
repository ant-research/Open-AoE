#!/usr/bin/env python3
"""Keep non-finite DAI rollout rewards out of ranking and optimizer telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path


SENTINEL = "AOE_NONFINITE_REWARD_GUARD"

OLD_WARNING = '''        if nan_mask.any():
            loguru.logger.warning(
                f"NaNs or infs in rews: {nan_mask.sum()}/{config.num_samples}"
            )

        ctrls_mean = (weights[:, None, None] * ctrls_samples).sum(dim=0)
'''

NEW_WARNING = '''        if nan_mask.any():
            loguru.logger.warning(
                f"NaNs or infs in rews: {nan_mask.sum()}/{config.num_samples}"
            )

        # AOE_NONFINITE_REWARD_GUARD: _compute_weights_impl already gives
        # invalid rollouts the worst finite reward for the control update.
        # Use the same policy for ranking, telemetry, and early stopping so a
        # partial batch of invalid simulations cannot produce NaN progress or
        # select a non-finite trace as a top candidate.
        finite_or_inf = torch.where(
            nan_mask, torch.tensor(torch.inf, device=rews.device), rews
        )
        worst_finite_rew = finite_or_inf.min()
        worst_finite_rew = torch.where(
            torch.isinf(worst_finite_rew),
            torch.tensor(-1000.0, device=rews.device),
            worst_finite_rew,
        )
        safe_rews = torch.where(nan_mask, worst_finite_rew, rews)

        ctrls_mean = (weights[:, None, None] * ctrls_samples).sum(dim=0)
'''

REPLACEMENTS = (
    ("torch.topk(rews, k=n_topk, largest=True).indices", "torch.topk(safe_rews, k=n_topk, largest=True).indices"),
    ("        rews_np = rews.cpu().numpy()\n", "        rews_np = safe_rews.cpu().numpy()\n"),
    (
        '        info["rew_mean"] = rews_np.mean()\n',
        '        info["rew_mean"] = rews_np.mean()\n'
        '        info["invalid_reward_count"] = int(nan_mask.sum().item())\n'
        '        info["invalid_reward_fraction"] = float(nan_mask.float().mean().item())\n',
    ),
    ("            info[\"trace_cost\"] = -rews[sel_idx].cpu().numpy()", "            info[\"trace_cost\"] = -safe_rews[sel_idx].cpu().numpy()"),
)


def patch_text(text: str) -> tuple[str, bool]:
    if SENTINEL in text:
        return text, False
    if OLD_WARNING not in text:
        raise ValueError("expected DAI optimize_once warning block not found")
    patched = text.replace(OLD_WARNING, NEW_WARNING, 1)
    for old, new in REPLACEMENTS:
        if old not in patched:
            raise ValueError(f"expected DAI sampling block not found: {old!r}")
        patched = patched.replace(old, new, 1)
    return patched, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-py", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = args.sampling_py
    original = path.read_text(encoding="utf-8")
    patched, changed = patch_text(original)
    if args.dry_run:
        print(f"[dai-reward-guard] {'would patch' if changed else 'already current'}: {path}")
        return 0
    if not changed:
        print(f"[dai-reward-guard] already current: {path}")
        return 0

    backup = path.with_suffix(path.suffix + ".aoe_nonfinite_reward_guard.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")
    print(f"[dai-reward-guard] installed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
