from scripts.compatibility.run_egoinfinity_sam3d_worker import _InferenceCompatibilityProxy


class _Pipeline:
    def __init__(self):
        self.call = None

    def run(self, *args, **kwargs):
        self.call = (args, kwargs)
        return {"gs_4": object()}


class _CurrentInference:
    def __init__(self):
        self._pipeline = _Pipeline()

    def merge_mask_to_rgba(self, image, mask):
        return ("rgba", image, mask)

    def __call__(self, image, mask, seed=None):
        raise AssertionError("the current notebook wrapper must be bypassed")


def test_routes_legacy_sampling_controls_to_current_pipeline():
    inference = _CurrentInference()
    proxy = _InferenceCompatibilityProxy(inference)
    result = proxy(
        "image",
        "mask",
        seed=42,
        stage1_inference_steps=2,
        stage2_inference_steps=12,
        use_stage1_distillation=True,
        use_stage2_distillation=True,
        decode_formats=["gaussian_4"],
    )

    args, kwargs = inference._pipeline.call
    assert args[:3] == (("rgba", "image", "mask"), None, 42)
    assert kwargs["stage1_inference_steps"] == 2
    assert kwargs["stage2_inference_steps"] == 12
    assert kwargs["decode_formats"] == ["gaussian_4"]
    assert result.keys() == {"gs_4"}


def test_rejects_unknown_arguments_instead_of_silently_dropping_them():
    inference = _CurrentInference()
    proxy = _InferenceCompatibilityProxy(inference)
    try:
        proxy("image", "mask", unknown_control=True)
    except TypeError as exc:
        assert "unknown_control" in str(exc)
    else:
        raise AssertionError("unknown controls must fail closed")
