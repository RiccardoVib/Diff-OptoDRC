"""Plot predictions from multiple model directories against a shared target.

Expected directory layout:

ROOT_DIR/
├── models_dir_1/
│   ├── config_a/
│   │   ├── 0_output_audio.wav
│   │   └── 0_prediction_audio.wav
│   └── config_b/
│       ├── 0_output_audio.wav
│       └── 0_prediction_audio.wav
├── models_dir_2/
│   ├── config_a/
│   │   ├── 0_output_audio.wav
│   │   └── 0_prediction_audio.wav
│   └── ...
└── ...

"""

from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
DURATION = 24000*2

def compute_rms_envelope(signal, frame_length=2048, hop_length=512):
    """Compute RMS envelope in dB."""
    # Ensure mono
    if signal.ndim > 1:
        signal = signal.mean(axis=0)

    # Compute RMS
    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]

    # Convert to dB (with floor to avoid -inf)
    rms_db = librosa.amplitude_to_db(rms, ref=1.0, top_db=None)

    return rms_db


def load_audio(filepath):
    """Load audio file at original sample rate."""
    y, sr = librosa.load(filepath, sr=None, mono=True)
    return y, sr


def find_audio(directory: Path, stem: str) -> Path | None:
    """Find an audio file whose filename starts with ``stem``."""
    for extension in AUDIO_EXTENSIONS:
        exact_path = directory / f"{stem}{extension}"
        if exact_path.is_file():
            return exact_path

    matches = sorted(
        path for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
        and path.stem.startswith(stem)
    )
    return matches[0] if matches else None


def match_length(audio: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad audio to ``length`` samples."""
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))

def main():

    import os
    from pathlib import Path
    from src.common.utils import find_folder_upward

    current_dir = Path(os.getcwd())
    print(f"current_dir: {current_dir}")
    files_dir = find_folder_upward(folder_name="SERVER", start_path=current_dir)
    ROOT_DIR = files_dir / "deep4" / "LA2A"
    output = ROOT_DIR
    CONFIG = "1.0"

    models_dirs = ["tcn_LA2A_analog/" + CONFIG,
                   "gcntf_LA2A_analog/" + CONFIG,
                   "sptmod_LA2A_analog/" + CONFIG,
                   "mamba_LA2A_analog/" + CONFIG,
                   "greycomp_LA2A_analog/" + CONFIG,
                   "diffopto_LA2A_analog/" + CONFIG]

    model_names = ["TCN", "GCN", "SPTmod", "Mamba", "GreyComp", "DiffOpto"]

    frame_length = 2048
    hop_length = 512//4
    linewidth =  1.

    models_dirs = [ROOT_DIR / name for name in models_dirs]
    for md in models_dirs:
        if not md.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {md}")

    first_md = models_dirs[0]
    target_path = find_audio(first_md, "0_output_audio")
    if target_path is None:
        raise FileNotFoundError(
            f"Could not find 0_output_audio in {first_md}"
        )

    target, target_sr = load_audio(target_path)
    print(f"Reference target: {target_path} (sample rate: {target_sr})")

    # Collect all predictions
    predictions: list[tuple[str, Path, np.ndarray, int]] = []
    for md, model_name in zip(models_dirs, model_names):
        prediction_path = find_audio(md, "0_prediction_audio")
        if prediction_path is None:
            print(
                f"Warning: no 0_prediction_audio found in {md}; skipping."
            )
            continue

        prediction, prediction_sr = load_audio(prediction_path)
        predictions.append(
            (model_name, prediction_path, prediction[:DURATION], prediction_sr)
        )
        print(
            f"Prediction: {prediction_path} "
            f"(sample rate: {prediction_sr})"
        )

    if not predictions:
        raise FileNotFoundError("No 0_prediction_audio files were found.")

    target = target[:DURATION]
    # Compute target envelope
    target_env = compute_rms_envelope(
        target, frame_length=frame_length, hop_length=hop_length
    )
    target_time = (
            np.arange(len(target_env)) * hop_length / target_sr
    )

    fig, ax = plt.subplots(3, 1, figsize=(12, 6), constrained_layout=True)

    # Plot target envelope
    ax[0].plot(
        target_time,
        target_env,
        color="black",
        linewidth=linewidth + 0.5,
        label="Target",
        alpha=0.9,
    )

    # Plot all prediction envelopes
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    for color, (model_dir_name, _, prediction, prediction_sr) in zip(
            colors, predictions
    ):
        # Optionally trim/pad prediction to target length before envelope
        if len(prediction) != len(target):
            prediction = match_length(prediction, len(target))

        env = compute_rms_envelope(
            prediction, frame_length=frame_length, hop_length=hop_length
        )
        pred_time = np.arange(len(env)) * hop_length / prediction_sr

        label = f"{model_dir_name}"
        ax[0].plot(
            pred_time,
            env,
            color=color,
            linewidth=linewidth,
            label=label,
            alpha=0.8,
        )

    ax[0].set_ylabel("RMS Level (dB)", fontsize=15)
    ax[0].legend(loc="upper right", fontsize=9)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_xlim([0, target_time[-1]])

    if output is None:
        plt.show()
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()