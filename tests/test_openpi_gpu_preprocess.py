"""Unit tests for OpenPI GPU preprocess steps (TDD red phase).

Each class covers one step of the planned migration. CPU oracles encode the
contract that the GPU implementations in
``joyrlinf.models.openpi_gpu_preprocess`` must match.

Run (after impl lands)::

    pytest tests/test_openpi_gpu_preprocess.py -v

Until implementations exist, step tests fail with NotImplementedError.
API-surface / oracle self-tests still pass.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PREPROCESS_PATH = REPO_ROOT / "joyrlinf" / "models" / "openpi_gpu_preprocess.py"
ACTION_MODEL_PATH = REPO_ROOT / "joyrlinf" / "models" / "openpi_action_model.py"


# ---------------------------------------------------------------------------
# CPU oracles (contract for GPU implementations)
# ---------------------------------------------------------------------------

def _oracle_resize_with_pad(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Torch resize_with_pad used as GPU-preprocess oracle.

    Input/output layout: ``[H,W,C]`` or ``[B,H,W,C]``.
    """
    if images.ndim not in (3, 4):
        raise ValueError(f"expected 3D/4D HWC, got shape={tuple(images.shape)}")

    is_3d = images.ndim == 3
    img = images.unsqueeze(0) if is_3d else images

    # Fast path: already at target size
    if img.shape[1] == height and img.shape[2] == width:
        return images

    img_t = img.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW
    cur_h, cur_w = img_t.shape[2], img_t.shape[3]
    ratio = max(cur_w / width, cur_h / height)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)

    interp_mode = "nearest" if "nearest" in mode.lower() else "bilinear"
    # Cast uint8 -> float for interpolate (deterministic across devices).
    resized = F.interpolate(
        img_t.float() if img_t.dtype == torch.uint8 else img_t,
        size=(resized_h, resized_w),
        mode=interp_mode,
        align_corners=False if interp_mode == "bilinear" else None,
    )
    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
        pad_value = 0
    elif images.dtype == torch.float32:
        resized = resized.clamp(-1.0, 1.0)
        pad_value = -1.0
    else:
        raise ValueError(f"unsupported dtype: {images.dtype}")

    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_w, 2)
    pad_w1 = pad_w0 + rem_w
    resized = F.pad(
        resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=pad_value
    )
    out = resized.permute(0, 2, 3, 1).contiguous()
    return out.squeeze(0) if is_3d else out


def _oracle_images_uint8_to_float(images: torch.Tensor) -> torch.Tensor:
    """Match ``Observation.from_dict`` numpy uint8 branch (layout unchanged)."""
    assert images.dtype == torch.uint8
    return images.to(torch.float32) / 255.0 * 2.0 - 1.0


def _oracle_masked_image_fill(like: torch.Tensor) -> torch.Tensor:
    if like.dtype == torch.uint8:
        return torch.zeros_like(like)
    return torch.full_like(like, -1.0)


def _oracle_pack_maniskill(
    base_rgb: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    fill = _oracle_masked_image_fill(base_rgb)
    bsz = base_rgb.shape[0]
    image = {
        "base_0_rgb": base_rgb,
        "left_wrist_0_rgb": fill,
        "right_wrist_0_rgb": fill.clone(),
    }
    device = base_rgb.device
    image_mask = {
        "base_0_rgb": torch.ones(bsz, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.zeros(bsz, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.zeros(bsz, dtype=torch.bool, device=device),
    }
    return image, image_mask


def _oracle_pack_libero(
    base_rgb: torch.Tensor,
    wrist_rgb: torch.Tensor,
    *,
    mask_right_wrist: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    zeros = _oracle_masked_image_fill(base_rgb)
    bsz = base_rgb.shape[0]
    device = base_rgb.device
    image = {
        "base_0_rgb": base_rgb,
        "left_wrist_0_rgb": wrist_rgb,
        "right_wrist_0_rgb": zeros,
    }
    image_mask = {
        "base_0_rgb": torch.ones(bsz, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(bsz, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": (
            torch.ones(bsz, dtype=torch.bool, device=device)
            if mask_right_wrist
            else torch.zeros(bsz, dtype=torch.bool, device=device)
        ),
    }
    return image, image_mask


def _oracle_normalize_state(
    state: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor | None = None,
    *,
    q01: torch.Tensor | None = None,
    q99: torch.Tensor | None = None,
    use_quantiles: bool = False,
) -> torch.Tensor:
    dim = state.shape[-1]
    if use_quantiles:
        assert q01 is not None and q99 is not None
        q01_ = q01[..., :dim]
        q99_ = q99[..., :dim]
        return (state - q01_) / (q99_ - q01_ + 1e-6) * 2.0 - 1.0
    assert std is not None
    mean_ = mean[..., :dim]
    std_ = std[..., :dim]
    return (state - mean_) / (std_ + 1e-6)


def _oracle_pad_state(state: torch.Tensor, target_dim: int) -> torch.Tensor:
    cur = state.shape[-1]
    if cur >= target_dim:
        return state
    pad = target_dim - cur
    return F.pad(state, (0, pad), mode="constant", value=0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_gpu():
    from joyrlinf.models import openpi_gpu_preprocess as m

    return m


def _rand_uint8_hwc(
    h: int, w: int, *, batch: int | None = None, seed: int = 0
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if batch is None:
        return torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, generator=g)
    return torch.randint(0, 256, (batch, h, w, 3), dtype=torch.uint8, generator=g)


def _assert_close_uint8(actual: torch.Tensor, expected: torch.Tensor, atol: int = 1):
    """Allow +/-1 grey level for bilinear round differences across devices."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.uint8
    diff = (actual.to(torch.int16) - expected.to(torch.int16)).abs()
    assert int(diff.max()) <= atol, f"max abs diff={int(diff.max())} > {atol}"


# ---------------------------------------------------------------------------
# API surface (always runnable)
# ---------------------------------------------------------------------------

class TestGpuPreprocessApiSurface:
    """Module must export the step functions used by the migration plan."""

    EXPECTED = (
        "resize_with_pad",
        "images_uint8_to_float",
        "pack_maniskill_images",
        "pack_libero_images",
        "normalize_state",
        "pad_state",
        "PromptTokenCache",
        "preprocess_obs_batch",
    )

    def test_module_exists(self):
        assert GPU_PREPROCESS_PATH.is_file()

    def test_exports(self):
        mod = _import_gpu()
        for name in self.EXPECTED:
            assert hasattr(mod, name), f"missing export: {name}"

    def test_ast_defines_expected_functions(self):
        tree = ast.parse(GPU_PREPROCESS_PATH.read_text(encoding="utf-8"))
        func_names = {
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        class_names = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        for name in self.EXPECTED:
            assert name in func_names or name in class_names, name


# ---------------------------------------------------------------------------
# Oracle self-checks (document the contract; do not need GPU impl)
# ---------------------------------------------------------------------------

class TestOracleResizeWithPad:
    def test_fast_path_noop(self):
        img = _rand_uint8_hwc(224, 224, seed=1)
        out = _oracle_resize_with_pad(img, 224, 224)
        assert torch.equal(out, img)

    def test_output_shape_single_and_batch(self):
        single = _oracle_resize_with_pad(_rand_uint8_hwc(128, 160, seed=2), 224, 224)
        assert single.shape == (224, 224, 3)
        batch = _oracle_resize_with_pad(
            _rand_uint8_hwc(128, 160, batch=4, seed=3), 224, 224
        )
        assert batch.shape == (4, 224, 224, 3)

    def test_uint8_pad_is_zero_on_letterbox(self):
        # Tall image -> horizontal letterbox pads.
        img = torch.full((200, 100, 3), 200, dtype=torch.uint8)
        out = _oracle_resize_with_pad(img, 224, 224)
        assert out[:, 0, 0].eq(0).all() or out[:, -1, 0].eq(0).all()

    def test_float_pad_is_minus_one(self):
        img = torch.ones(100, 200, 3, dtype=torch.float32)
        out = _oracle_resize_with_pad(img, 224, 224)
        assert torch.isclose(out[0, 0, 0], torch.tensor(-1.0)) or torch.isclose(
            out[-1, 0, 0], torch.tensor(-1.0)
        )

    def test_openpi_jax_resize_smoke_when_importable(self):
        """Stock openpi JAX resize_with_pad remains usable (CPU fallback path)."""
        try:
            from openpi.shared import image_tools as _image_tools
        except Exception as exc:  # noqa: BLE001 — openpi may be absent
            pytest.skip(f"openpi image_tools unavailable: {exc}")

        img = _rand_uint8_hwc(180, 240, seed=7).numpy()
        ref = np.asarray(_image_tools.resize_with_pad(img, 224, 224))
        assert ref.shape == (224, 224, 3)
        assert ref.dtype == img.dtype


class TestOracleImagesUint8ToFloat:
    def test_black_white_range(self):
        img = torch.tensor([[[0, 0, 0], [255, 255, 255]]], dtype=torch.uint8)
        out = _oracle_images_uint8_to_float(img)
        assert out.dtype == torch.float32
        assert torch.isclose(out[0, 0, 0], torch.tensor(-1.0))
        assert torch.isclose(out[0, 1, 0], torch.tensor(1.0))

    def test_mid_grey(self):
        img = torch.full((2, 2, 3), 128, dtype=torch.uint8)
        out = _oracle_images_uint8_to_float(img)
        expected = img.to(torch.float32) / 255.0 * 2.0 - 1.0
        assert torch.allclose(out, expected)


class TestOraclePackImages:
    def test_maniskill_masks(self):
        base = torch.randn(3, 8, 8, 3)
        image, mask = _oracle_pack_maniskill(base)
        assert set(image) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
        assert torch.equal(image["base_0_rgb"], base)
        assert torch.allclose(image["left_wrist_0_rgb"], torch.full_like(base, -1.0))
        assert mask["base_0_rgb"].tolist() == [True, True, True]
        assert mask["left_wrist_0_rgb"].tolist() == [False, False, False]
        assert mask["right_wrist_0_rgb"].tolist() == [False, False, False]

    def test_libero_masks_default(self):
        base = torch.randn(2, 4, 4, 3)
        wrist = torch.randn(2, 4, 4, 3)
        image, mask = _oracle_pack_libero(base, wrist)
        assert torch.equal(image["left_wrist_0_rgb"], wrist)
        assert mask["left_wrist_0_rgb"].all()
        assert not mask["right_wrist_0_rgb"].any()


class TestOracleNormalizeAndPad:
    def test_zscore(self):
        state = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        mean = torch.tensor([1.0, 2.0])
        std = torch.tensor([1.0, 2.0])
        out = _oracle_normalize_state(state, mean, std)
        expected = torch.tensor([[0.0, 0.5], [1.0, 1.0]])
        assert torch.allclose(out, expected)

    def test_quantile(self):
        state = torch.tensor([[0.0, 1.0]])
        q01 = torch.tensor([0.0, 0.0])
        q99 = torch.tensor([1.0, 2.0])
        out = _oracle_normalize_state(
            state, mean=torch.zeros(2), use_quantiles=True, q01=q01, q99=q99
        )
        # Denominator includes +1e-6 (OpenPI), so second dim is ~0 not exact 0.
        expected = (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        assert torch.allclose(out, expected)

    def test_pad_and_noop(self):
        state = torch.randn(4, 7)
        padded = _oracle_pad_state(state, 32)
        assert padded.shape == (4, 32)
        assert torch.equal(padded[:, :7], state)
        assert torch.count_nonzero(padded[:, 7:]) == 0
        assert torch.equal(_oracle_pad_state(padded, 32), padded)


# ---------------------------------------------------------------------------
# Step 1: resize_with_pad
# ---------------------------------------------------------------------------

class TestStepResizeWithPad:
    def test_matches_oracle_uint8_batch(self):
        gpu = _import_gpu()
        src = _rand_uint8_hwc(120, 200, batch=5, seed=11)
        expected = _oracle_resize_with_pad(src, 224, 224)
        actual = gpu.resize_with_pad(src, 224, 224)
        _assert_close_uint8(actual, expected, atol=1)

    def test_matches_oracle_non_square_and_already_224(self):
        gpu = _import_gpu()
        tall = _rand_uint8_hwc(300, 100, batch=2, seed=12)
        _assert_close_uint8(
            gpu.resize_with_pad(tall, 224, 224),
            _oracle_resize_with_pad(tall, 224, 224),
            atol=1,
        )
        already = _rand_uint8_hwc(224, 224, batch=2, seed=13)
        out = gpu.resize_with_pad(already, 224, 224)
        assert out.shape == (2, 224, 224, 3)
        assert torch.equal(out, already)

    def test_float_path(self):
        gpu = _import_gpu()
        img = torch.rand(2, 64, 96, 3, dtype=torch.float32) * 2 - 1
        expected = _oracle_resize_with_pad(img, 224, 224)
        actual = gpu.resize_with_pad(img, 224, 224)
        assert actual.dtype == torch.float32
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_stays_on_device(self):
        gpu = _import_gpu()
        src = _rand_uint8_hwc(128, 128, batch=3, seed=14).cuda()
        out = gpu.resize_with_pad(src, 224, 224)
        assert out.is_cuda
        assert out.shape == (3, 224, 224, 3)

    def test_cpu_exact_match_oracle(self):
        """Same algorithm on CPU must be bitwise identical to the oracle."""
        gpu = _import_gpu()
        src = _rand_uint8_hwc(120, 200, batch=8, seed=15)
        assert torch.equal(
            gpu.resize_with_pad(src, 224, 224),
            _oracle_resize_with_pad(src, 224, 224),
        )
        fsrc = torch.rand(4, 90, 140, 3) * 2 - 1
        assert torch.equal(
            gpu.resize_with_pad(fsrc, 224, 224),
            _oracle_resize_with_pad(fsrc, 224, 224),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_matches_cpu_precision(self):
        """CUDA vs CPU: uint8 within 1 grey level; float32 tightly close."""
        gpu = _import_gpu()
        src_cpu = _rand_uint8_hwc(160, 240, batch=8, seed=16)
        out_cpu = gpu.resize_with_pad(src_cpu, 224, 224)
        out_cuda = gpu.resize_with_pad(src_cpu.cuda(), 224, 224).cpu()
        _assert_close_uint8(out_cuda, out_cpu, atol=1)

        f_cpu = (torch.rand(4, 96, 128, 3) * 2 - 1).to(torch.float32)
        f_out_cpu = gpu.resize_with_pad(f_cpu, 224, 224)
        f_out_cuda = gpu.resize_with_pad(f_cpu.cuda(), 224, 224).cpu()
        assert torch.allclose(f_out_cuda, f_out_cpu, atol=1e-4, rtol=1e-4)


class TestStepResizeWithPadBenchmark:
    """Timing: per-sample CPU loop vs batched CPU vs batched CUDA.

    Run with::

        pytest tests/test_openpi_gpu_preprocess.py -k TestStepResizeWithPadBenchmark -v -s
    """

    BENCHMARK_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    SRC_H, SRC_W = 128, 160
    WARMUP = 3
    TIMED_ITERS = 20

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @classmethod
    def _bench(cls, fn, device: torch.device) -> tuple[float, float, float]:
        for _ in range(cls.WARMUP):
            fn()
        cls._sync(device)
        times: list[float] = []
        for _ in range(cls.TIMED_ITERS):
            cls._sync(device)
            t0 = time.perf_counter()
            fn()
            cls._sync(device)
            times.append(time.perf_counter() - t0)
        return min(times), sum(times) / len(times), max(times)

    @staticmethod
    def _per_sample_cpu(src: torch.Tensor, resize_fn) -> torch.Tensor:
        """Simulate current input_transform: one image at a time on CPU."""
        return torch.stack(
            [resize_fn(src[i], 224, 224) for i in range(src.shape[0])], dim=0
        )

    def test_benchmark_cpu_vs_gpu(self):
        gpu = _import_gpu()
        resize_fn = gpu.resize_with_pad
        results = []

        for bs in self.BENCHMARK_BATCH_SIZES:
            src_cpu = _rand_uint8_hwc(
                self.SRC_H, self.SRC_W, batch=bs, seed=100 + bs
            )
            # Correctness snapshot at this batch size (CPU exact, CUDA atol=1).
            batched_cpu = resize_fn(src_cpu, 224, 224)
            assert torch.equal(batched_cpu, _oracle_resize_with_pad(src_cpu, 224, 224))
            per_sample = self._per_sample_cpu(src_cpu, resize_fn)
            assert torch.equal(batched_cpu, per_sample)

            per_min, per_avg, _ = self._bench(
                lambda s=src_cpu: self._per_sample_cpu(s, resize_fn),
                torch.device("cpu"),
            )
            batch_min, batch_avg, _ = self._bench(
                lambda s=src_cpu: resize_fn(s, 224, 224),
                torch.device("cpu"),
            )

            row = {
                "bs": bs,
                "per_sample_avg_ms": per_avg * 1000,
                "per_sample_min_ms": per_min * 1000,
                "batch_cpu_avg_ms": batch_avg * 1000,
                "batch_cpu_min_ms": batch_min * 1000,
                "batch_cuda_avg_ms": None,
                "batch_cuda_min_ms": None,
                "speedup_batch_vs_persample": per_avg / batch_avg if batch_avg else float("inf"),
                "speedup_cuda_vs_persample": None,
            }

            if torch.cuda.is_available():
                src_cuda = src_cpu.cuda()
                out_cuda = resize_fn(src_cuda, 224, 224).cpu()
                _assert_close_uint8(out_cuda, batched_cpu, atol=1)
                cuda_min, cuda_avg, _ = self._bench(
                    lambda s=src_cuda: resize_fn(s, 224, 224),
                    torch.device("cuda"),
                )
                row["batch_cuda_avg_ms"] = cuda_avg * 1000
                row["batch_cuda_min_ms"] = cuda_min * 1000
                row["speedup_cuda_vs_persample"] = (
                    per_avg / cuda_avg if cuda_avg else float("inf")
                )

            results.append(row)

        has_cuda = results[0]["batch_cuda_avg_ms"] is not None
        print(f"\n{'=' * 88}")
        print(
            "resize_with_pad benchmark: "
            f"per-sample CPU vs batched CPU"
            + (" vs batched CUDA" if has_cuda else " (no CUDA)")
        )
        print(f"src={self.SRC_H}x{self.SRC_W} uint8 -> 224x224 | iters={self.TIMED_ITERS}")
        print(f"{'=' * 88}")
        if has_cuda:
            header = (
                f"{'bs':>4} | {'per_sample(ms)':>14} | {'batch_cpu(ms)':>13} | "
                f"{'batch_cuda(ms)':>14} | {'cpu_spdup':>9} | {'cuda_spdup':>10}"
            )
        else:
            header = (
                f"{'bs':>4} | {'per_sample(ms)':>14} | {'batch_cpu(ms)':>13} | "
                f"{'cpu_spdup':>9}"
            )
        print(header)
        print("-" * len(header))
        for r in results:
            if has_cuda:
                print(
                    f"{r['bs']:>4} | {r['per_sample_avg_ms']:>14.3f} | "
                    f"{r['batch_cpu_avg_ms']:>13.3f} | "
                    f"{r['batch_cuda_avg_ms']:>14.3f} | "
                    f"{r['speedup_batch_vs_persample']:>8.2f}x | "
                    f"{r['speedup_cuda_vs_persample']:>9.2f}x"
                )
            else:
                print(
                    f"{r['bs']:>4} | {r['per_sample_avg_ms']:>14.3f} | "
                    f"{r['batch_cpu_avg_ms']:>13.3f} | "
                    f"{r['speedup_batch_vs_persample']:>8.2f}x"
                )
        print(f"{'=' * 88}")

        # Soft performance expectations: batched CPU should beat per-sample for bs>=8.
        large = [r for r in results if r["bs"] >= 8]
        assert large, "expected at least one bs>=8 row"
        avg_cpu_speedup = sum(r["speedup_batch_vs_persample"] for r in large) / len(large)
        print(f"avg batched-CPU speedup vs per-sample (bs>=8): {avg_cpu_speedup:.2f}x")
        assert avg_cpu_speedup >= 1.0, (
            f"batched CPU should not be slower than per-sample on avg; got {avg_cpu_speedup:.2f}x"
        )
        if has_cuda:
            avg_cuda_speedup = sum(
                r["speedup_cuda_vs_persample"] for r in large
            ) / len(large)
            print(f"avg batched-CUDA speedup vs per-sample (bs>=8): {avg_cuda_speedup:.2f}x")


# ---------------------------------------------------------------------------
# Step 2: images_uint8_to_float
# ---------------------------------------------------------------------------

class TestStepImagesUint8ToFloat:
    def test_matches_oracle(self):
        gpu = _import_gpu()
        src = _rand_uint8_hwc(16, 16, batch=4, seed=21)
        actual = gpu.images_uint8_to_float(src)
        expected = _oracle_images_uint8_to_float(src)
        assert actual.dtype == torch.float32
        assert torch.allclose(actual, expected)
        assert actual.shape == src.shape  # no CHW permute

    def test_endpoints(self):
        gpu = _import_gpu()
        src = torch.tensor([[[[0, 128, 255]]]], dtype=torch.uint8)  # [1,1,1,3]
        out = gpu.images_uint8_to_float(src)
        assert torch.isclose(out[0, 0, 0, 0], torch.tensor(-1.0))
        assert torch.isclose(out[0, 0, 0, 2], torch.tensor(1.0))

    def test_cpu_exact_match_oracle(self):
        gpu = _import_gpu()
        src = _rand_uint8_hwc(64, 64, batch=8, seed=22)
        assert torch.equal(
            gpu.images_uint8_to_float(src),
            _oracle_images_uint8_to_float(src),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_matches_cpu_precision(self):
        gpu = _import_gpu()
        src = _rand_uint8_hwc(64, 64, batch=8, seed=23)
        out_cpu = gpu.images_uint8_to_float(src)
        out_cuda = gpu.images_uint8_to_float(src.cuda()).cpu()
        # GPU float div can differ by ~1 ULP from CPU; require tight closeness.
        assert torch.allclose(out_cuda, out_cpu, atol=1e-6, rtol=0.0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_stays_on_device(self):
        gpu = _import_gpu()
        src = _rand_uint8_hwc(32, 32, batch=2, seed=24).cuda()
        out = gpu.images_uint8_to_float(src)
        assert out.is_cuda
        assert out.dtype == torch.float32


class TestStepImagesUint8ToFloatBenchmark:
    """Timing: per-sample CPU vs batched CPU vs batched CUDA.

    Run with::

        pytest tests/test_openpi_gpu_preprocess.py -k TestStepImagesUint8ToFloatBenchmark -v -s
    """

    BENCHMARK_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    SRC_H, SRC_W = 224, 224
    WARMUP = 3
    TIMED_ITERS = 50

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @classmethod
    def _bench(cls, fn, device: torch.device) -> tuple[float, float]:
        for _ in range(cls.WARMUP):
            fn()
        cls._sync(device)
        times: list[float] = []
        for _ in range(cls.TIMED_ITERS):
            cls._sync(device)
            t0 = time.perf_counter()
            fn()
            cls._sync(device)
            times.append(time.perf_counter() - t0)
        return min(times), sum(times) / len(times)

    @staticmethod
    def _per_sample_cpu(src: torch.Tensor, convert_fn) -> torch.Tensor:
        return torch.stack([convert_fn(src[i]) for i in range(src.shape[0])], dim=0)

    def test_benchmark_cpu_vs_gpu(self):
        gpu = _import_gpu()
        convert_fn = gpu.images_uint8_to_float
        results = []

        for bs in self.BENCHMARK_BATCH_SIZES:
            src_cpu = _rand_uint8_hwc(
                self.SRC_H, self.SRC_W, batch=bs, seed=200 + bs
            )
            batched = convert_fn(src_cpu)
            assert torch.equal(batched, _oracle_images_uint8_to_float(src_cpu))
            assert torch.equal(batched, self._per_sample_cpu(src_cpu, convert_fn))

            per_min, per_avg = self._bench(
                lambda s=src_cpu: self._per_sample_cpu(s, convert_fn),
                torch.device("cpu"),
            )
            batch_min, batch_avg = self._bench(
                lambda s=src_cpu: convert_fn(s),
                torch.device("cpu"),
            )
            row = {
                "bs": bs,
                "per_sample_avg_ms": per_avg * 1000,
                "batch_cpu_avg_ms": batch_avg * 1000,
                "batch_cuda_avg_ms": None,
                "speedup_batch_vs_persample": per_avg / batch_avg if batch_avg else float("inf"),
                "speedup_cuda_vs_persample": None,
            }
            if torch.cuda.is_available():
                src_cuda = src_cpu.cuda()
                assert torch.allclose(
                    convert_fn(src_cuda).cpu(), batched, atol=1e-6, rtol=0.0
                )
                _, cuda_avg = self._bench(
                    lambda s=src_cuda: convert_fn(s),
                    torch.device("cuda"),
                )
                row["batch_cuda_avg_ms"] = cuda_avg * 1000
                row["speedup_cuda_vs_persample"] = (
                    per_avg / cuda_avg if cuda_avg else float("inf")
                )
            results.append(row)

        has_cuda = results[0]["batch_cuda_avg_ms"] is not None
        print(f"\n{'=' * 88}")
        print(
            "images_uint8_to_float benchmark: per-sample CPU vs batched CPU"
            + (" vs batched CUDA" if has_cuda else " (no CUDA)")
        )
        print(f"src={self.SRC_H}x{self.SRC_W} uint8 | iters={self.TIMED_ITERS}")
        print(f"{'=' * 88}")
        if has_cuda:
            header = (
                f"{'bs':>4} | {'per_sample(ms)':>14} | {'batch_cpu(ms)':>13} | "
                f"{'batch_cuda(ms)':>14} | {'cpu_spdup':>9} | {'cuda_spdup':>10}"
            )
        else:
            header = (
                f"{'bs':>4} | {'per_sample(ms)':>14} | {'batch_cpu(ms)':>13} | "
                f"{'cpu_spdup':>9}"
            )
        print(header)
        print("-" * len(header))
        for r in results:
            if has_cuda:
                print(
                    f"{r['bs']:>4} | {r['per_sample_avg_ms']:>14.3f} | "
                    f"{r['batch_cpu_avg_ms']:>13.3f} | "
                    f"{r['batch_cuda_avg_ms']:>14.3f} | "
                    f"{r['speedup_batch_vs_persample']:>8.2f}x | "
                    f"{r['speedup_cuda_vs_persample']:>9.2f}x"
                )
            else:
                print(
                    f"{r['bs']:>4} | {r['per_sample_avg_ms']:>14.3f} | "
                    f"{r['batch_cpu_avg_ms']:>13.3f} | "
                    f"{r['speedup_batch_vs_persample']:>8.2f}x"
                )
        print(f"{'=' * 88}")
        large = [r for r in results if r["bs"] >= 8]
        avg_cpu = sum(r["speedup_batch_vs_persample"] for r in large) / len(large)
        print(f"avg batched-CPU speedup vs per-sample (bs>=8): {avg_cpu:.2f}x")
        # This op is memory-bound on CPU; batched float32 alloc can be slower
        # than per-sample. The meaningful win is staying on CUDA.
        if has_cuda:
            avg_cuda = sum(r["speedup_cuda_vs_persample"] for r in large) / len(large)
            print(f"avg batched-CUDA speedup vs per-sample (bs>=8): {avg_cuda:.2f}x")
            assert avg_cuda >= 1.0, (
                f"batched CUDA should beat per-sample CPU; got {avg_cuda:.2f}x"
            )


# ---------------------------------------------------------------------------
# Shared timing helper (used by step 3+)
# ---------------------------------------------------------------------------

def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, device: torch.device, *, warmup: int = 3, iters: int = 30):
    for _ in range(warmup):
        fn()
    _sync_device(device)
    times: list[float] = []
    for _ in range(iters):
        _sync_device(device)
        t0 = time.perf_counter()
        fn()
        _sync_device(device)
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def _assert_finite(*tensors: torch.Tensor) -> None:
    for t in tensors:
        assert torch.isfinite(t).all(), f"non-finite values in tensor shape={tuple(t.shape)}"


def _print_speed_table(title: str, rows: list[dict], has_cuda: bool) -> None:
    print(f"\n{'=' * 88}")
    print(title)
    print(f"{'=' * 88}")
    if has_cuda:
        header = (
            f"{'bs':>4} | {'baseline(ms)':>12} | {'batch_cpu(ms)':>13} | "
            f"{'batch_cuda(ms)':>14} | {'cpu_spdup':>9} | {'cuda_spdup':>10}"
        )
    else:
        header = (
            f"{'bs':>4} | {'baseline(ms)':>12} | {'batch_cpu(ms)':>13} | "
            f"{'cpu_spdup':>9}"
        )
    print(header)
    print("-" * len(header))
    for r in rows:
        if has_cuda:
            print(
                f"{r['bs']:>4} | {r['baseline_avg_ms']:>12.3f} | "
                f"{r['batch_cpu_avg_ms']:>13.3f} | "
                f"{r['batch_cuda_avg_ms']:>14.3f} | "
                f"{r['speedup_cpu']:>8.2f}x | "
                f"{r['speedup_cuda']:>9.2f}x"
            )
        else:
            print(
                f"{r['bs']:>4} | {r['baseline_avg_ms']:>12.3f} | "
                f"{r['batch_cpu_avg_ms']:>13.3f} | "
                f"{r['speedup_cpu']:>8.2f}x"
            )
    print(f"{'=' * 88}")


# ---------------------------------------------------------------------------
# Step 3: pack_maniskill_images
# ---------------------------------------------------------------------------

class TestStepPackManiskillImages:
    def test_matches_oracle(self):
        gpu = _import_gpu()
        base = torch.randn(4, 224, 224, 3)
        image, mask = gpu.pack_maniskill_images(base)
        ref_image, ref_mask = _oracle_pack_maniskill(base)
        for k in ref_image:
            assert torch.equal(image[k], ref_image[k])
            assert torch.equal(mask[k], ref_mask[k])

    def test_wrists_are_zeros_same_shape(self):
        gpu = _import_gpu()
        base = torch.ones(2, 8, 8, 3)
        image, _ = gpu.pack_maniskill_images(base)
        assert image["left_wrist_0_rgb"].shape == base.shape
        assert torch.allclose(image["left_wrist_0_rgb"], torch.full_like(base, -1.0))
        assert torch.allclose(image["right_wrist_0_rgb"], torch.full_like(base, -1.0))

    def test_cpu_exact_match_oracle(self):
        gpu = _import_gpu()
        base = torch.randn(8, 32, 32, 3)
        image, mask = gpu.pack_maniskill_images(base)
        ref_i, ref_m = _oracle_pack_maniskill(base)
        for k in ref_i:
            assert torch.equal(image[k], ref_i[k])
            assert torch.equal(mask[k], ref_m[k])
        _assert_finite(image["base_0_rgb"], image["left_wrist_0_rgb"])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_matches_cpu_precision(self):
        gpu = _import_gpu()
        base = torch.randn(4, 16, 16, 3)
        img_cpu, mask_cpu = gpu.pack_maniskill_images(base)
        img_cuda, mask_cuda = gpu.pack_maniskill_images(base.cuda())
        for k in img_cpu:
            assert img_cuda[k].is_cuda and mask_cuda[k].is_cuda
            assert torch.equal(img_cuda[k].cpu(), img_cpu[k])
            assert torch.equal(mask_cuda[k].cpu(), mask_cpu[k])


class TestStepPackManiskillImagesBenchmark:
    """Pack is mostly allocation; still time CPU vs CUDA for bs sweep."""

    BENCHMARK_BATCH_SIZES = [1, 4, 8, 16, 32]

    def test_benchmark_cpu_vs_gpu(self):
        gpu = _import_gpu()
        rows = []
        for bs in self.BENCHMARK_BATCH_SIZES:
            base = torch.randn(bs, 224, 224, 3)
            # Correctness
            out, _ = gpu.pack_maniskill_images(base)
            ref, _ = _oracle_pack_maniskill(base)
            assert torch.equal(out["base_0_rgb"], ref["base_0_rgb"])

            def baseline():
                # Simulate per-sample packing
                for i in range(bs):
                    gpu.pack_maniskill_images(base[i : i + 1])

            _, base_avg = _time_call(baseline, torch.device("cpu"), iters=20)
            _, cpu_avg = _time_call(
                lambda b=base: gpu.pack_maniskill_images(b),
                torch.device("cpu"),
                iters=20,
            )
            row = {
                "bs": bs,
                "baseline_avg_ms": base_avg * 1000,
                "batch_cpu_avg_ms": cpu_avg * 1000,
                "batch_cuda_avg_ms": None,
                "speedup_cpu": base_avg / cpu_avg if cpu_avg else float("inf"),
                "speedup_cuda": None,
            }
            if torch.cuda.is_available():
                bcuda = base.cuda()
                _, cuda_avg = _time_call(
                    lambda b=bcuda: gpu.pack_maniskill_images(b),
                    torch.device("cuda"),
                    iters=20,
                )
                row["batch_cuda_avg_ms"] = cuda_avg * 1000
                row["speedup_cuda"] = base_avg / cuda_avg if cuda_avg else float("inf")
            rows.append(row)

        has_cuda = rows[0]["batch_cuda_avg_ms"] is not None
        _print_speed_table(
            "pack_maniskill_images: per-sample vs batched CPU"
            + (" vs CUDA" if has_cuda else ""),
            rows,
            has_cuda,
        )


# ---------------------------------------------------------------------------
# Step 4: pack_libero_images
# ---------------------------------------------------------------------------

class TestStepPackLiberoImages:
    def test_matches_oracle_default_mask(self):
        gpu = _import_gpu()
        base = torch.randn(3, 16, 16, 3)
        wrist = torch.randn(3, 16, 16, 3)
        image, mask = gpu.pack_libero_images(base, wrist)
        ref_image, ref_mask = _oracle_pack_libero(base, wrist)
        for k in ref_image:
            assert torch.equal(image[k], ref_image[k])
            assert torch.equal(mask[k], ref_mask[k])

    def test_mask_right_wrist_true(self):
        gpu = _import_gpu()
        base = torch.randn(1, 8, 8, 3)
        wrist = torch.randn(1, 8, 8, 3)
        _, mask = gpu.pack_libero_images(base, wrist, mask_right_wrist=True)
        assert mask["right_wrist_0_rgb"].all()

    def test_cpu_exact_match_oracle(self):
        gpu = _import_gpu()
        base = torch.randn(5, 20, 20, 3)
        wrist = torch.randn(5, 20, 20, 3)
        image, mask = gpu.pack_libero_images(base, wrist)
        ref_i, ref_m = _oracle_pack_libero(base, wrist)
        for k in ref_i:
            assert torch.equal(image[k], ref_i[k])
            assert torch.equal(mask[k], ref_m[k])
        _assert_finite(image["base_0_rgb"], image["left_wrist_0_rgb"])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_matches_cpu_precision(self):
        gpu = _import_gpu()
        base = torch.randn(3, 12, 12, 3)
        wrist = torch.randn(3, 12, 12, 3)
        img_cpu, mask_cpu = gpu.pack_libero_images(base, wrist)
        img_cuda, mask_cuda = gpu.pack_libero_images(base.cuda(), wrist.cuda())
        for k in img_cpu:
            assert torch.equal(img_cuda[k].cpu(), img_cpu[k])
            assert torch.equal(mask_cuda[k].cpu(), mask_cpu[k])


class TestStepPackLiberoImagesBenchmark:
    BENCHMARK_BATCH_SIZES = [1, 4, 8, 16, 32]

    def test_benchmark_cpu_vs_gpu(self):
        gpu = _import_gpu()
        rows = []
        for bs in self.BENCHMARK_BATCH_SIZES:
            base = torch.randn(bs, 224, 224, 3)
            wrist = torch.randn(bs, 224, 224, 3)
            out, _ = gpu.pack_libero_images(base, wrist)
            ref, _ = _oracle_pack_libero(base, wrist)
            assert torch.equal(out["left_wrist_0_rgb"], ref["left_wrist_0_rgb"])

            def baseline():
                for i in range(bs):
                    gpu.pack_libero_images(base[i : i + 1], wrist[i : i + 1])

            _, base_avg = _time_call(baseline, torch.device("cpu"), iters=20)
            _, cpu_avg = _time_call(
                lambda: gpu.pack_libero_images(base, wrist),
                torch.device("cpu"),
                iters=20,
            )
            row = {
                "bs": bs,
                "baseline_avg_ms": base_avg * 1000,
                "batch_cpu_avg_ms": cpu_avg * 1000,
                "batch_cuda_avg_ms": None,
                "speedup_cpu": base_avg / cpu_avg if cpu_avg else float("inf"),
                "speedup_cuda": None,
            }
            if torch.cuda.is_available():
                bcuda, wcuda = base.cuda(), wrist.cuda()
                _, cuda_avg = _time_call(
                    lambda: gpu.pack_libero_images(bcuda, wcuda),
                    torch.device("cuda"),
                    iters=20,
                )
                row["batch_cuda_avg_ms"] = cuda_avg * 1000
                row["speedup_cuda"] = base_avg / cuda_avg if cuda_avg else float("inf")
            rows.append(row)

        has_cuda = rows[0]["batch_cuda_avg_ms"] is not None
        _print_speed_table(
            "pack_libero_images: per-sample vs batched CPU"
            + (" vs CUDA" if has_cuda else ""),
            rows,
            has_cuda,
        )


# ---------------------------------------------------------------------------
# Step 5: normalize_state
# ---------------------------------------------------------------------------

class TestStepNormalizeState:
    def test_zscore_matches_oracle(self):
        gpu = _import_gpu()
        state = torch.randn(8, 7)
        mean = torch.randn(7)
        std = torch.rand(7) + 0.1
        actual = gpu.normalize_state(state, mean, std)
        expected = _oracle_normalize_state(state, mean, std)
        assert torch.allclose(actual, expected)
        _assert_finite(actual)

    def test_quantile_matches_oracle(self):
        gpu = _import_gpu()
        state = torch.randn(5, 4)
        q01 = torch.full((4,), -0.5)
        q99 = torch.full((4,), 0.5)
        actual = gpu.normalize_state(
            state, mean=torch.zeros(4), use_quantiles=True, q01=q01, q99=q99
        )
        expected = _oracle_normalize_state(
            state, mean=torch.zeros(4), use_quantiles=True, q01=q01, q99=q99
        )
        assert torch.allclose(actual, expected)
        _assert_finite(actual)

    def test_stats_wider_than_state_are_sliced(self):
        gpu = _import_gpu()
        state = torch.randn(2, 3)
        mean = torch.arange(8, dtype=torch.float32)
        std = torch.ones(8)
        actual = gpu.normalize_state(state, mean, std)
        expected = _oracle_normalize_state(state, mean, std)
        assert actual.shape == (2, 3)
        assert torch.allclose(actual, expected)

    def test_cpu_exact_match_oracle(self):
        gpu = _import_gpu()
        state = torch.randn(16, 7)
        mean = torch.randn(7)
        std = torch.rand(7) + 0.05
        assert torch.equal(
            gpu.normalize_state(state, mean, std),
            _oracle_normalize_state(state, mean, std),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_matches_cpu_precision(self):
        gpu = _import_gpu()
        state = torch.randn(16, 7)
        mean = torch.randn(7)
        std = torch.rand(7) + 0.05
        out_cpu = gpu.normalize_state(state, mean, std)
        out_cuda = gpu.normalize_state(
            state.cuda(), mean.cuda(), std.cuda()
        ).cpu()
        assert torch.allclose(out_cuda, out_cpu, atol=1e-5, rtol=1e-5)
        _assert_finite(out_cuda)

    def test_extreme_values_remain_finite(self):
        gpu = _import_gpu()
        state = torch.tensor([[1e6, -1e6, 0.0, 1e-8]])
        mean = torch.zeros(4)
        std = torch.tensor([1e-3, 1e-3, 1.0, 1e-3])
        out = gpu.normalize_state(state, mean, std)
        _assert_finite(out)


class TestStepNormalizeStateBenchmark:
    BENCHMARK_BATCH_SIZES = [1, 4, 8, 16, 32, 64]
    STATE_DIM = 7

    def test_benchmark_cpu_vs_gpu(self):
        gpu = _import_gpu()
        mean = torch.randn(self.STATE_DIM)
        std = torch.rand(self.STATE_DIM) + 0.1
        rows = []
        for bs in self.BENCHMARK_BATCH_SIZES:
            state = torch.randn(bs, self.STATE_DIM)
            assert torch.allclose(
                gpu.normalize_state(state, mean, std),
                _oracle_normalize_state(state, mean, std),
            )

            def baseline():
                for i in range(bs):
                    gpu.normalize_state(state[i : i + 1], mean, std)

            _, base_avg = _time_call(baseline, torch.device("cpu"), iters=50)
            _, cpu_avg = _time_call(
                lambda: gpu.normalize_state(state, mean, std),
                torch.device("cpu"),
                iters=50,
            )
            row = {
                "bs": bs,
                "baseline_avg_ms": base_avg * 1000,
                "batch_cpu_avg_ms": cpu_avg * 1000,
                "batch_cuda_avg_ms": None,
                "speedup_cpu": base_avg / cpu_avg if cpu_avg else float("inf"),
                "speedup_cuda": None,
            }
            if torch.cuda.is_available():
                s_cuda, m_cuda, std_cuda = state.cuda(), mean.cuda(), std.cuda()
                _, cuda_avg = _time_call(
                    lambda: gpu.normalize_state(s_cuda, m_cuda, std_cuda),
                    torch.device("cuda"),
                    iters=50,
                )
                row["batch_cuda_avg_ms"] = cuda_avg * 1000
                row["speedup_cuda"] = base_avg / cuda_avg if cuda_avg else float("inf")
            rows.append(row)

        has_cuda = rows[0]["batch_cuda_avg_ms"] is not None
        _print_speed_table(
            "normalize_state: per-sample vs batched CPU"
            + (" vs CUDA" if has_cuda else ""),
            rows,
            has_cuda,
        )


# ---------------------------------------------------------------------------
# Step 6: pad_state
# ---------------------------------------------------------------------------

class TestStepPadState:
    def test_pad_to_action_dim(self):
        gpu = _import_gpu()
        state = torch.randn(6, 7)
        out = gpu.pad_state(state, 32)
        assert out.shape == (6, 32)
        assert torch.equal(out[:, :7], state)
        assert torch.count_nonzero(out[:, 7:]) == 0

    def test_noop_when_already_wide_enough(self):
        gpu = _import_gpu()
        state = torch.randn(2, 32)
        assert torch.equal(gpu.pad_state(state, 32), state)

    def test_cpu_exact_match_oracle(self):
        gpu = _import_gpu()
        state = torch.randn(8, 7)
        assert torch.equal(gpu.pad_state(state, 32), _oracle_pad_state(state, 32))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_matches_cpu_precision(self):
        gpu = _import_gpu()
        state = torch.randn(8, 7)
        out_cpu = gpu.pad_state(state, 32)
        out_cuda = gpu.pad_state(state.cuda(), 32)
        assert out_cuda.is_cuda
        assert torch.equal(out_cuda.cpu(), out_cpu)


class TestStepPadStateBenchmark:
    BENCHMARK_BATCH_SIZES = [1, 8, 16, 32, 64]

    def test_benchmark_cpu_vs_gpu(self):
        gpu = _import_gpu()
        rows = []
        for bs in self.BENCHMARK_BATCH_SIZES:
            state = torch.randn(bs, 7)
            assert torch.equal(gpu.pad_state(state, 32), _oracle_pad_state(state, 32))

            def baseline():
                for i in range(bs):
                    gpu.pad_state(state[i : i + 1], 32)

            _, base_avg = _time_call(baseline, torch.device("cpu"), iters=50)
            _, cpu_avg = _time_call(
                lambda: gpu.pad_state(state, 32),
                torch.device("cpu"),
                iters=50,
            )
            row = {
                "bs": bs,
                "baseline_avg_ms": base_avg * 1000,
                "batch_cpu_avg_ms": cpu_avg * 1000,
                "batch_cuda_avg_ms": None,
                "speedup_cpu": base_avg / cpu_avg if cpu_avg else float("inf"),
                "speedup_cuda": None,
            }
            if torch.cuda.is_available():
                s_cuda = state.cuda()
                _, cuda_avg = _time_call(
                    lambda: gpu.pad_state(s_cuda, 32),
                    torch.device("cuda"),
                    iters=50,
                )
                row["batch_cuda_avg_ms"] = cuda_avg * 1000
                row["speedup_cuda"] = base_avg / cuda_avg if cuda_avg else float("inf")
            rows.append(row)

        has_cuda = rows[0]["batch_cuda_avg_ms"] is not None
        _print_speed_table(
            "pad_state: per-sample vs batched CPU"
            + (" vs CUDA" if has_cuda else ""),
            rows,
            has_cuda,
        )


# ---------------------------------------------------------------------------
# Step 7: PromptTokenCache
# ---------------------------------------------------------------------------

class TestStepPromptTokenCache:
    @staticmethod
    def _fake_tokenize(prompt: str):
        tokens = torch.tensor(
            [len(prompt) % 97 + 1, sum(map(ord, prompt)) % 97 + 1, 0, 0],
            dtype=torch.int32,
        )
        mask = torch.tensor([True, True, False, False])
        return tokens, mask

    def test_cache_hit_skips_tokenize(self):
        gpu = _import_gpu()
        calls: list[str] = []

        def counting_tokenize(prompt: str):
            calls.append(prompt)
            return self._fake_tokenize(prompt)

        cache = gpu.PromptTokenCache(counting_tokenize)
        prompts = ["pick cube", "pick cube", "place cube"]
        env_ids = [0, 1, 2]

        t1, m1 = cache.get(env_ids, prompts)
        assert len(calls) == 3
        assert t1.shape[0] == 3

        t2, m2 = cache.get(env_ids, prompts)
        assert len(calls) == 3  # no extra tokenize
        assert torch.equal(t1, t2)
        assert torch.equal(m1, m2)

    def test_prompt_change_refreshes_entry(self):
        gpu = _import_gpu()
        calls: list[str] = []

        def counting_tokenize(prompt: str):
            calls.append(prompt)
            return self._fake_tokenize(prompt)

        cache = gpu.PromptTokenCache(counting_tokenize)
        cache.get([0], ["task a"])
        cache.get([0], ["task a"])
        assert calls == ["task a"]
        cache.get([0], ["task b"])
        assert calls == ["task a", "task b"]

    def test_invalidate(self):
        gpu = _import_gpu()
        calls: list[str] = []

        def counting_tokenize(prompt: str):
            calls.append(prompt)
            return self._fake_tokenize(prompt)

        cache = gpu.PromptTokenCache(counting_tokenize)
        cache.get([0, 1], ["x", "y"])
        cache.invalidate([0])
        cache.get([0, 1], ["x", "y"])
        # env0 retokenized; env1 cache hit
        assert calls == ["x", "y", "x"]

    def test_values_match_tokenizer_exactly(self):
        gpu = _import_gpu()
        cache = gpu.PromptTokenCache(self._fake_tokenize)
        prompts = [f"task-{i}" for i in range(8)]
        tokens, masks = cache.get(list(range(8)), prompts)
        for i, p in enumerate(prompts):
            tok, m = self._fake_tokenize(p)
            assert torch.equal(tokens[i], tok)
            assert torch.equal(masks[i], m)

    def test_invalidate_all(self):
        gpu = _import_gpu()
        calls: list[str] = []

        def counting_tokenize(prompt: str):
            calls.append(prompt)
            return self._fake_tokenize(prompt)

        cache = gpu.PromptTokenCache(counting_tokenize)
        cache.get([0, 1], ["a", "b"])
        cache.invalidate(None)
        cache.get([0, 1], ["a", "b"])
        assert calls == ["a", "b", "a", "b"]


class TestStepPromptTokenCacheBenchmark:
    """Cold (miss) vs warm (hit) latency; hit must be clearly faster."""

    N_ENVS = 32
    TIMED_ITERS = 100

    def test_benchmark_cold_vs_warm(self):
        gpu = _import_gpu()

        def slow_tokenize(prompt: str):
            # Cheap but non-trivial stand-in for SentencePiece.
            acc = 0
            for ch in prompt:
                acc = (acc * 131 + ord(ch)) % 10_007
            tokens = torch.tensor([acc % 256, (acc // 256) % 256] + [0] * 14, dtype=torch.int32)
            mask = torch.zeros(16, dtype=torch.bool)
            mask[:2] = True
            return tokens, mask

        prompts = [f"manipulate object number {i} carefully" for i in range(self.N_ENVS)]
        env_ids = list(range(self.N_ENVS))

        # Cold: new cache each iter
        def cold():
            cache = gpu.PromptTokenCache(slow_tokenize)
            cache.get(env_ids, prompts)

        # Warm: populate once then hit
        warm_cache = gpu.PromptTokenCache(slow_tokenize)
        warm_cache.get(env_ids, prompts)

        def warm():
            warm_cache.get(env_ids, prompts)

        # Correctness on warm path
        t, m = warm_cache.get(env_ids, prompts)
        for i, p in enumerate(prompts):
            tok, mk = slow_tokenize(p)
            assert torch.equal(t[i], tok)
            assert torch.equal(m[i], mk)

        _, cold_avg = _time_call(cold, torch.device("cpu"), warmup=2, iters=20)
        _, warm_avg = _time_call(warm, torch.device("cpu"), warmup=5, iters=self.TIMED_ITERS)
        speedup = cold_avg / warm_avg if warm_avg else float("inf")

        print(f"\n{'=' * 60}")
        print("PromptTokenCache benchmark: cold miss vs warm hit")
        print(f"n_envs={self.N_ENVS}")
        print(f"{'=' * 60}")
        print(f"cold_avg_ms={cold_avg * 1000:.3f}  warm_avg_ms={warm_avg * 1000:.3f}  speedup={speedup:.2f}x")
        print(f"{'=' * 60}")
        assert speedup >= 1.5, (
            f"warm cache should be clearly faster than cold; got {speedup:.2f}x"
        )


# ---------------------------------------------------------------------------
# Composed path
# ---------------------------------------------------------------------------

class TestStepPreprocessObsBatch:
    @staticmethod
    def _tokenize_fn(prompt: str):
        tokens = torch.zeros(16, dtype=torch.int32)
        mask = torch.zeros(16, dtype=torch.bool)
        tokens[0] = len(prompt) % 256
        mask[0] = True
        return tokens, mask

    def test_composed_shapes_maniskill(self):
        gpu = _import_gpu()
        bsz = 4
        base = _rand_uint8_hwc(128, 160, batch=bsz, seed=40)
        state = torch.randn(bsz, 7)
        mean = torch.zeros(7)
        std = torch.ones(7)
        prompts = [f"task-{i}" for i in range(bsz)]

        out = gpu.preprocess_obs_batch(
            base_images=base,
            state=state,
            prompts=prompts,
            state_mean=mean,
            state_std=std,
            action_dim=32,
            pack_mode="maniskill",
            tokenize_fn=self._tokenize_fn,
        )
        assert out["image"]["base_0_rgb"].shape == (bsz, 224, 224, 3)
        assert out["image"]["base_0_rgb"].dtype == torch.float32
        assert out["state"].shape == (bsz, 32)
        assert out["tokenized_prompt"].shape[0] == bsz
        assert out["tokenized_prompt_mask"].shape[0] == bsz
        assert out["image_mask"]["base_0_rgb"].all()
        assert not out["image_mask"]["left_wrist_0_rgb"].any()
        _assert_finite(out["image"]["base_0_rgb"], out["state"])

    def test_composed_matches_stepwise_oracle(self):
        """Full path must equal composing the per-step oracles (+ tokenize)."""
        gpu = _import_gpu()
        bsz = 3
        base = _rand_uint8_hwc(100, 140, batch=bsz, seed=41)
        state = torch.randn(bsz, 7)
        mean = torch.randn(7)
        std = torch.rand(7) + 0.05
        prompts = ["alpha", "beta", "gamma"]

        out = gpu.preprocess_obs_batch(
            base_images=base,
            state=state,
            prompts=prompts,
            state_mean=mean,
            state_std=std,
            action_dim=16,
            pack_mode="maniskill",
            tokenize_fn=self._tokenize_fn,
        )

        resized = _oracle_resize_with_pad(base, 224, 224)
        floated = _oracle_images_uint8_to_float(resized)
        ref_image, ref_mask = _oracle_pack_maniskill(floated)
        ref_state = _oracle_pad_state(
            _oracle_normalize_state(state, mean, std), 16
        )
        for k in ref_image:
            assert torch.allclose(out["image"][k], ref_image[k], atol=1e-5)
            assert torch.equal(out["image_mask"][k], ref_mask[k])
        assert torch.allclose(out["state"], ref_state)
        for i, p in enumerate(prompts):
            tok, m = self._tokenize_fn(p)
            assert torch.equal(out["tokenized_prompt"][i], tok)
            assert torch.equal(out["tokenized_prompt_mask"][i], m)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_matches_cpu_precision(self):
        gpu = _import_gpu()
        bsz = 4
        base = _rand_uint8_hwc(128, 160, batch=bsz, seed=42)
        state = torch.randn(bsz, 7)
        mean = torch.randn(7)
        std = torch.rand(7) + 0.1
        prompts = [f"p{i}" for i in range(bsz)]
        kwargs = dict(
            prompts=prompts,
            state_mean=mean,
            state_std=std,
            action_dim=32,
            pack_mode="maniskill",
            tokenize_fn=self._tokenize_fn,
        )
        out_cpu = gpu.preprocess_obs_batch(
            base_images=base, state=state, **kwargs
        )
        out_cuda = gpu.preprocess_obs_batch(
            base_images=base.cuda(),
            state=state.cuda(),
            state_mean=mean.cuda(),
            state_std=std.cuda(),
            prompts=prompts,
            action_dim=32,
            pack_mode="maniskill",
            tokenize_fn=self._tokenize_fn,
        )
        for k in out_cpu["image"]:
            assert out_cuda["image"][k].is_cuda
            assert torch.allclose(
                out_cuda["image"][k].cpu(), out_cpu["image"][k], atol=1e-4, rtol=1e-4
            )
            assert torch.equal(
                out_cuda["image_mask"][k].cpu(), out_cpu["image_mask"][k]
            )
        assert torch.allclose(
            out_cuda["state"].cpu(), out_cpu["state"], atol=1e-5, rtol=1e-5
        )
        assert torch.equal(
            out_cuda["tokenized_prompt"].cpu(), out_cpu["tokenized_prompt"]
        )


class TestStepPreprocessObsBatchBenchmark:
    """End-to-end: stepwise CPU oracle pipeline vs preprocess_obs_batch CPU/CUDA."""

    BENCHMARK_BATCH_SIZES = [1, 4, 8, 16, 32]
    SRC_H, SRC_W = 128, 160

    @staticmethod
    def _tokenize_fn(prompt: str):
        tokens = torch.zeros(16, dtype=torch.int32)
        mask = torch.zeros(16, dtype=torch.bool)
        tokens[0] = len(prompt) % 256
        mask[0] = True
        return tokens, mask

    @classmethod
    def _oracle_pipeline(cls, base, state, mean, std, prompts, action_dim):
        resized = _oracle_resize_with_pad(base, 224, 224)
        floated = _oracle_images_uint8_to_float(resized)
        image, mask = _oracle_pack_maniskill(floated)
        st = _oracle_pad_state(_oracle_normalize_state(state, mean, std), action_dim)
        toks = torch.stack([cls._tokenize_fn(p)[0] for p in prompts])
        return image, mask, st, toks

    def test_benchmark_oracle_pipeline_vs_composed(self):
        gpu = _import_gpu()
        mean = torch.randn(7)
        std = torch.rand(7) + 0.1
        rows = []
        for bs in self.BENCHMARK_BATCH_SIZES:
            base = _rand_uint8_hwc(self.SRC_H, self.SRC_W, batch=bs, seed=300 + bs)
            state = torch.randn(bs, 7)
            prompts = [f"task-{i}" for i in range(bs)]

            # Precision snapshot
            out = gpu.preprocess_obs_batch(
                base_images=base,
                state=state,
                prompts=prompts,
                state_mean=mean,
                state_std=std,
                action_dim=32,
                pack_mode="maniskill",
                tokenize_fn=self._tokenize_fn,
            )
            ref_img, ref_mask, ref_state, ref_tok = self._oracle_pipeline(
                base, state, mean, std, prompts, 32
            )
            assert torch.allclose(out["image"]["base_0_rgb"], ref_img["base_0_rgb"], atol=1e-5)
            assert torch.allclose(out["state"], ref_state, atol=1e-5)
            assert torch.equal(out["tokenized_prompt"], ref_tok)

            _, base_avg = _time_call(
                lambda: self._oracle_pipeline(base, state, mean, std, prompts, 32),
                torch.device("cpu"),
                iters=15,
            )
            _, cpu_avg = _time_call(
                lambda: gpu.preprocess_obs_batch(
                    base_images=base,
                    state=state,
                    prompts=prompts,
                    state_mean=mean,
                    state_std=std,
                    action_dim=32,
                    pack_mode="maniskill",
                    tokenize_fn=self._tokenize_fn,
                ),
                torch.device("cpu"),
                iters=15,
            )
            row = {
                "bs": bs,
                "baseline_avg_ms": base_avg * 1000,
                "batch_cpu_avg_ms": cpu_avg * 1000,
                "batch_cuda_avg_ms": None,
                "speedup_cpu": base_avg / cpu_avg if cpu_avg else float("inf"),
                "speedup_cuda": None,
            }
            if torch.cuda.is_available():
                b_cuda, s_cuda = base.cuda(), state.cuda()
                m_cuda, std_cuda = mean.cuda(), std.cuda()
                _, cuda_avg = _time_call(
                    lambda: gpu.preprocess_obs_batch(
                        base_images=b_cuda,
                        state=s_cuda,
                        prompts=prompts,
                        state_mean=m_cuda,
                        state_std=std_cuda,
                        action_dim=32,
                        pack_mode="maniskill",
                        tokenize_fn=self._tokenize_fn,
                    ),
                    torch.device("cuda"),
                    iters=15,
                )
                row["batch_cuda_avg_ms"] = cuda_avg * 1000
                row["speedup_cuda"] = base_avg / cuda_avg if cuda_avg else float("inf")
            rows.append(row)

        has_cuda = rows[0]["batch_cuda_avg_ms"] is not None
        _print_speed_table(
            "preprocess_obs_batch: oracle CPU pipeline vs composed CPU"
            + (" vs CUDA" if has_cuda else ""),
            rows,
            has_cuda,
        )
        if has_cuda:
            large = [r for r in rows if r["bs"] >= 8]
            avg_cuda = sum(r["speedup_cuda"] for r in large) / len(large)
            print(f"avg composed-CUDA speedup vs oracle pipeline (bs>=8): {avg_cuda:.2f}x")
            # Soft note only: end-to-end is often tokenize-bound on CPU, and
            # full-suite GPU contention makes absolute speedup flaky.
            if avg_cuda < 1.0:
                print(
                    "note: CUDA e2e < 1x vs oracle pipeline "
                    "(likely CPU tokenize / sync dominated; not a hard failure)"
                )


# ---------------------------------------------------------------------------
# Config flag surface
# ---------------------------------------------------------------------------

class TestConfigFlagSurface:
    """Config knobs the GPU path will read."""

    def test_use_gpu_input_transform_in_action_model(self):
        source = ACTION_MODEL_PATH.read_text(encoding="utf-8")
        assert "use_gpu_input_transform" in source
        assert "cache_tokenized_prompt" in source
        assert "_gpu_input_transform" in source
        assert "_init_gpu_preprocess" in source


@pytest.mark.openpi
class TestGpuInputTransformWiredConsistency:
    """CPU input_transform vs GPU path on a real OpenPI checkpoint."""

    @staticmethod
    def _build_model(use_gpu: bool):
        pytest.importorskip("openpi")
        from omegaconf import OmegaConf

        from joyrlinf.models.openpi_model import OpenPiModel

        model_path = (
            "/ufs_500T/model/colm_paper_model_test/"
            "RLinf-Pi05-ManiSkill-25Main-SFT/"
        )
        if not Path(model_path).exists():
            pytest.skip(f"OpenPI checkpoint not found: {model_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = OmegaConf.create(
            {
                "model_type": "openpi",
                "model_path": model_path,
                "openpi": {
                    "config_name": "pi05_maniskill",
                    "use_gpu_input_transform": use_gpu,
                    "cache_tokenized_prompt": True,
                },
            }
        )
        return OpenPiModel(cfg, device=device, mode="inference")

    def test_gpu_matches_cpu_input_transform(self):
        cpu_model = self._build_model(use_gpu=False).model
        gpu_model = self._build_model(use_gpu=True).model
        assert gpu_model.config.use_gpu_input_transform is True
        assert gpu_model._gpu_tokenize_fn is not None

        bsz = 4
        device = next(gpu_model.parameters()).device
        env_obs = {
            "main_images": torch.randint(
                0, 256, (bsz, 128, 160, 3), dtype=torch.uint8, device=device
            ),
            "wrist_images": None,
            "states": torch.randn(bsz, 8, dtype=torch.float32, device=device),
            "task_descriptions": [f"pick cube {i}" for i in range(bsz)],
        }
        obs = gpu_model.obs_processor(env_obs)

        cpu_out = cpu_model.precision_processor(
            cpu_model.input_transform(
                {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in obs.items()},
                transpose=False,
            )
        )
        # Move cpu_out to device for compare
        for k, v in list(cpu_out.items()):
            if torch.is_tensor(v):
                cpu_out[k] = v.to(device)
            elif isinstance(v, dict):
                cpu_out[k] = {
                    sk: sv.to(device) if torch.is_tensor(sv) else sv
                    for sk, sv in v.items()
                }

        assert gpu_model._can_use_gpu_input_transform(obs, need_prompt=True)
        gpu_out = gpu_model._gpu_input_transform(obs)

        assert set(gpu_out["image"]) == set(cpu_out["image"])
        for k in gpu_out["image"]:
            g = gpu_out["image"][k].float()
            c_raw = cpu_out["image"][k]
            # CPU path may still be uint8 HWC until from_dict; GPU path returns
            # NCHW float in [-1, 1]. Check dtype BEFORE .float(): all-zero
            # uint8 masked cameras must map to -1.0 like from_dict does.
            if c_raw.dtype == torch.uint8:
                c = c_raw.float() / 255.0 * 2.0 - 1.0
            else:
                c = c_raw.float()
                if c.max() > 1.5:
                    c = c / 255.0 * 2.0 - 1.0
            if c.ndim == 4 and c.shape[-1] == 3 and g.shape[1] == 3:
                c = c.permute(0, 3, 1, 2)
            assert torch.allclose(g, c, atol=2e-2, rtol=2e-2), k
            assert torch.equal(
                gpu_out["image_mask"][k].cpu(),
                cpu_out["image_mask"][k].cpu().bool()
                if cpu_out["image_mask"][k].dtype != torch.bool
                else cpu_out["image_mask"][k].cpu(),
            )
        assert torch.allclose(
            gpu_out["state"].float(), cpu_out["state"].float(), atol=1e-4, rtol=1e-4
        )
        assert torch.equal(
            gpu_out["tokenized_prompt"].cpu(),
            cpu_out["tokenized_prompt"].cpu(),
        )


class TestUnnormalizeActions:
    """GPU unnormalize vs openpi.transforms.Unnormalize numpy reference."""

    @staticmethod
    def _np_unnormalize_quantile(x, q01, q99):
        d = q01.shape[-1]
        if d < x.shape[-1]:
            return np.concatenate(
                [(x[..., :d] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., d:]],
                axis=-1,
            )
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    @staticmethod
    def _np_unnormalize(x, mean, std):
        def pad(v, dim, value):
            if v.shape[-1] >= dim:
                return v
            return np.concatenate(
                [v, np.full(dim - v.shape[-1], value, dtype=v.dtype)], axis=-1
            )

        mean_p = pad(mean, x.shape[-1], 0.0)
        std_p = pad(std, x.shape[-1], 1.0)
        return x * (std_p + 1e-6) + mean_p

    def test_quantile_stats_narrower_than_actions(self):
        from joyrlinf.models import openpi_gpu_preprocess as gpu

        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 8, 32)).astype(np.float32)
        q01 = rng.standard_normal(7).astype(np.float32) - 2.0
        q99 = q01 + np.abs(rng.standard_normal(7).astype(np.float32)) + 0.5

        ref = self._np_unnormalize_quantile(x, q01, q99)
        got = gpu.unnormalize_actions(
            torch.from_numpy(x),
            q01=torch.from_numpy(q01),
            q99=torch.from_numpy(q99),
            use_quantiles=True,
        )
        np.testing.assert_array_equal(got.numpy(), ref.astype(np.float32))

    def test_quantile_stats_full_dim(self):
        from joyrlinf.models import openpi_gpu_preprocess as gpu

        rng = np.random.default_rng(1)
        x = rng.standard_normal((3, 5, 7)).astype(np.float32)
        q01 = rng.standard_normal(7).astype(np.float32) - 1.0
        q99 = q01 + 1.0

        ref = self._np_unnormalize_quantile(x, q01, q99)
        got = gpu.unnormalize_actions(
            torch.from_numpy(x),
            q01=torch.from_numpy(q01),
            q99=torch.from_numpy(q99),
            use_quantiles=True,
        )
        np.testing.assert_array_equal(got.numpy(), ref.astype(np.float32))

    def test_zscore_with_pad(self):
        from joyrlinf.models import openpi_gpu_preprocess as gpu

        rng = np.random.default_rng(2)
        x = rng.standard_normal((4, 8, 32)).astype(np.float32)
        mean = rng.standard_normal(7).astype(np.float32)
        std = np.abs(rng.standard_normal(7).astype(np.float32)) + 0.1

        ref = self._np_unnormalize(x, mean, std)
        got = gpu.unnormalize_actions(
            torch.from_numpy(x),
            mean=torch.from_numpy(mean),
            std=torch.from_numpy(std),
            use_quantiles=False,
        )
        np.testing.assert_allclose(got.numpy(), ref.astype(np.float32), rtol=1e-6)


class TestPrepareActionsManiskillTorch:
    """Torch elementwise path vs the previous numpy round-trip semantics."""

    @staticmethod
    def _np_oracle(raw: np.ndarray, num_chunks: int, action_dim: int, policy: str):
        reshaped = raw.reshape(-1, action_dim)
        world = reshaped[:, :3] * 1.0
        rot = reshaped[:, 3:6] * 1.0
        if policy in ("widowx_bridge", "panda_wristcam"):
            gripper = 2.0 * (reshaped[:, 6:7] > 0.5) - 1.0
        else:
            gripper = reshaped[:, 6:7]
        out = np.concatenate([world, rot, gripper], axis=1).astype(np.float32)
        return out.reshape(-1, num_chunks, action_dim)

    @pytest.mark.parametrize("policy", ["widowx_bridge", "other_policy"])
    def test_matches_numpy_oracle(self, policy):
        if not torch.cuda.is_available():
            pytest.skip("prepare_actions_for_maniskill returns CUDA tensors")
        from joyrlinf.envs.action_utils import prepare_actions_for_maniskill

        rng = np.random.default_rng(3)
        raw = rng.uniform(-1.5, 1.5, size=(8, 5, 7)).astype(np.float32)
        ref = self._np_oracle(raw, 5, 7, policy)

        # numpy input
        got_np = prepare_actions_for_maniskill(raw, 5, 7, 1.0, policy)
        np.testing.assert_array_equal(got_np.cpu().numpy(), ref)

        # CUDA tensor input (GPU output transform path)
        got_gpu = prepare_actions_for_maniskill(
            torch.from_numpy(raw).cuda(), 5, 7, 1.0, policy
        )
        np.testing.assert_array_equal(got_gpu.cpu().numpy(), ref)

    def test_gripper_threshold_boundary(self):
        if not torch.cuda.is_available():
            pytest.skip("prepare_actions_for_maniskill returns CUDA tensors")
        from joyrlinf.envs.action_utils import prepare_actions_for_maniskill

        raw = np.zeros((1, 1, 7), dtype=np.float32)
        raw[..., 6] = 0.5  # strictly > 0.5 required for open
        got = prepare_actions_for_maniskill(raw, 1, 7, 1.0, "widowx_bridge")
        assert got[0, 0, 6].item() == -1.0
        raw[..., 6] = 0.5000001
        got = prepare_actions_for_maniskill(raw, 1, 7, 1.0, "widowx_bridge")
        assert got[0, 0, 6].item() == 1.0


@pytest.mark.openpi


class TestGpuOutputTransformWiredConsistency:
    """CPU output_transform vs GPU path on a real OpenPI checkpoint."""

    @staticmethod
    def _build_model(use_gpu_output: bool):
        pytest.importorskip("openpi")
        from omegaconf import OmegaConf

        from joyrlinf.models.openpi_model import OpenPiModel

        model_path = (
            "/ufs_500T/model/colm_paper_model_test/"
            "RLinf-Pi05-ManiSkill-25Main-SFT/"
        )
        if not Path(model_path).exists():
            pytest.skip(f"OpenPI checkpoint not found: {model_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = OmegaConf.create(
            {
                "model_type": "openpi",
                "model_path": model_path,
                "openpi": {
                    "config_name": "pi05_maniskill",
                    "use_gpu_input_transform": True,
                    "use_gpu_output_transform": use_gpu_output,
                    "cache_tokenized_prompt": True,
                },
            }
        )
        return OpenPiModel(cfg, device=device, mode="inference")

    def test_gpu_matches_cpu_output_transform(self):
        model = self._build_model(use_gpu_output=True).model
        assert model._gpu_output_ok, "GPU output transform should be enabled"

        device = next(model.parameters()).device
        gen = torch.Generator().manual_seed(7)
        # sample_actions output range: normalized quantile actions live in
        # roughly [-1, 1]; use a wider range to exercise the transform.
        actions = torch.empty(6, model.config.action_horizon, 32).uniform_(
            -1.5, 1.5, generator=gen
        )
        state = torch.empty(6, 32).uniform_(-1.0, 1.0, generator=gen)

        cpu_out = model.output_transform(
            {"actions": actions.clone(), "state": state.clone()}
        )["actions"].numpy()
        gpu_out = model._gpu_output_transform(actions.to(device)).cpu().numpy()

        assert gpu_out.shape == cpu_out.shape
        np.testing.assert_allclose(gpu_out, cpu_out, rtol=1e-5, atol=1e-6)

    def test_flag_off_keeps_cpu_path(self):
        model = self._build_model(use_gpu_output=False).model
        assert not model._gpu_output_ok
