from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "maintenance" / "patch_dai_sampling_nonfinite_rewards.py"
SPEC = importlib.util.spec_from_file_location("patch_dai_sampling_nonfinite_rewards", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_patch_sanitizes_ranking_statistics_and_trace_cost() -> None:
    source = f'''        if nan_mask.any():
            loguru.logger.warning(
                f"NaNs or infs in rews: {{nan_mask.sum()}}/{{config.num_samples}}"
            )

        ctrls_mean = (weights[:, None, None] * ctrls_samples).sum(dim=0)
        idx_top = torch.topk(rews, k=n_topk, largest=True).indices
        rews_np = rews.cpu().numpy()
        info["rew_mean"] = rews_np.mean()
            info["trace_cost"] = -rews[sel_idx].cpu().numpy()
'''

    patched, changed = MODULE.patch_text(source)

    assert changed
    assert MODULE.SENTINEL in patched
    assert "torch.topk(safe_rews" in patched
    assert "rews_np = safe_rews.cpu().numpy()" in patched
    assert 'info["invalid_reward_fraction"]' in patched
    assert 'info["trace_cost"] = -safe_rews' in patched

    patched_again, changed_again = MODULE.patch_text(patched)
    assert not changed_again
    assert patched_again == patched
