# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato

"""
Compute parameter counts and forward-pass FLOPs
"""

from __future__ import annotations
import torch
from factory import build_model


# -----------------------------------------------------------------------------
# Paths and measurement settings
# -----------------------------------------------------------------------------

FS = 48000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_FOR_FLOPS = 1


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def format_count(value: int | float) -> str:
    value = float(value)
    if value >= 1e9:
        return f"{value / 1e9:.3f}G"
    if value >= 1e6:
        return f"{value / 1e6:.3f}M"
    if value >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.0f}"


def _try_call(model: torch.nn.Module, x: torch.Tensor, c: torch.Tensor):
    attempts = [
        lambda: model(x, c[:, None, :]),
        lambda: model(x, c),
        lambda: model(x, c[:, None, :], y_true=x),
        lambda: model(x),
    ]
    errors = []
    for attempt in attempts:
        try:
            return attempt()
        except (TypeError, RuntimeError, ValueError) as error:
            errors.append(error)
    message = "\n".join(f"  - {type(error).__name__}: {error}" for error in errors)
    raise RuntimeError(f"All forward-call variants failed:\n{message}")


def estimate_flops(model: torch.nn.Module, x: torch.Tensor, c: torch.Tensor) -> int:
    model.eval()
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_flops=True,
            record_shapes=False,
        ) as profiler:
            _try_call(model, x, c)
    return int(sum(event.flops or 0 for event in profiler.key_averages()))


def infer_input_channels(model_type: str, model_kwargs: dict) -> int:
    if model_type == "lstm":
        return int(model_kwargs.get("ninputs", 1))
    if model_type == "sptmod":
        return int(model_kwargs.get("input_channels", 1))
    return 1


def make_inputs(model_type: str, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = int(cfg["seq_len"])
    buffer = int(cfg["buffer"])
    channels = infer_input_channels(model_type, cfg.get("model_kwargs", {}))
    cond_dim = int(cfg.get("cond_dim", 1))


    x = torch.randn(
        BATCH_SIZE_FOR_FLOPS,
        channels,
        buffer + seq_len,
        device=DEVICE,
    )

    if model_type in {
        "lstm",
        "mamba",
        "ssmdrc_ff",
        "ssmdrc_fb",
        "ssmdrc_tf",
        "greycomp",
        }:
        x = x.transpose(1, 2)

    c = torch.randn(BATCH_SIZE_FOR_FLOPS, cond_dim, device=DEVICE)
    return x, c


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}\n")
    from configs import PAPER_CONFIGS_CL1B, PAPER_CONFIGS_LA2A

    PAPER_CONFIGS = PAPER_CONFIGS_CL1B #or PAPER_CONFIGS_LA2A
    results = []

    for model_type, cfg in PAPER_CONFIGS.items():
        print(f"[{model_type}] building model ...")
        try:
            model_kwargs = dict(cfg.get("model_kwargs", {}))
            cond_dim = int(cfg.get("cond_dim", 1))
            seq_len = int(cfg["seq_len"])
            buffer = int(cfg["buffer"])

            model = build_model(
                model_type,
                cond_dim=cond_dim,
                buffer=buffer,
                seq_len=seq_len,
                **model_kwargs,
            ).to(DEVICE)

            if hasattr(model, "reset_hidden_states"):
                model.reset_hidden_states()

            parameter_count = count_parameters(model)
            x, c = make_inputs(model_type, cfg)
            flop_count = estimate_flops(model, x, c)

            results.append((model_type, parameter_count, flop_count, "ok"))
            print(
                f"  params: {parameter_count:,} ({format_count(parameter_count)}) | "
                f"FLOPs/forward: {flop_count:,} ({format_count(flop_count)})"
            )

        except Exception as error:
            results.append((model_type, None, None, f"error: {error}"))
            print(f"  ERROR: {type(error).__name__}: {error}")

        finally:
            if "model" in locals():
                del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    print("\nSummary")
    print(f"{'Model':<16} {'Parameters':>16} {'FLOPs/forward':>18} Status")
    print("-" * 70)
    for model_type, parameter_count, flop_count, status in results:
        parameter_text = "failed" if parameter_count is None else f"{parameter_count:,}"
        flop_text = "failed" if flop_count is None else f"{flop_count:,}"
        print(f"{model_type:<16} {parameter_text:>16} {flop_text:>18} {status}")


if __name__ == "__main__":
    main()