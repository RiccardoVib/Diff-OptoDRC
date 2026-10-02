# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato


import math
import numpy as np
import re
from scipy.io.wavfile import write
import os
import matplotlib.pyplot as plt
import json
from pathlib import Path
from torch.utils.flop_counter import FlopCounterMode
import time
from datetime import datetime, time as dtime

# ──────────────────────────────────────────────────────────
# Training-window configuration
TRAIN_END = dtime(18, 0)   # 18:00
TRAIN_START = dtime(6, 0)    # 06:00

def _is_allowed_time(start: dtime, end: dtime) -> bool:
    """Handles both same-day and overnight windows."""
    now = datetime.now().time()
    if start <= end:       # e.g. 09:00 – 17:00
        return start <= now < end
    else:                  # overnight e.g. 22:00 – 08:00
        return now >= start or now < end

def wait_for_allowed_time(start=TRAIN_START, end=TRAIN_END,
                          poll_interval: int = 900) -> None:
    if _is_allowed_time(start, end):
        return
    print(f"[TimeGuard] Outside allowed window "
          f"({start.strftime('%H:%M')}–{end.strftime('%H:%M')}). Waiting...")
    while not _is_allowed_time(start, end):
        print(f"[TimeGuard] {datetime.now().strftime('%H:%M:%S')} – sleeping {poll_interval}s …")
        time.sleep(poll_interval)
    print(f"[TimeGuard] Resuming at {datetime.now().strftime('%H:%M:%S')}.")


def get_flops(model, *inputs, with_backward=False):
    is_train = model.training
    model.eval()

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(*inputs).sum().backward()
        else:
            model(*inputs)

    total_flops = flop_counter.get_total_flops()
    if is_train:
        model.train()
    return total_flops

def find_folder_upward(folder_name, start_path=None):
    """
    Search backward through parent directories until finding the requested folder.

    Args:
        folder_name: Name of the folder to find
        start_path: Starting directory (defaults to current working directory)

    Returns:
        Path object of the found folder, or None if not found
    """
    if start_path is None:
        current_path = Path.cwd()
    else:
        current_path = Path(start_path).resolve()

    # Check current directory and all parents
    for parent in [current_path] + list(current_path.parents):
        target = parent / folder_name
        if target.exists() and target.is_dir():
            return target

        # Stop at filesystem root
        if parent == parent.parent:
            break

    return None

def save_audio_files(input_audio, output_audio, prediction_audio, model_path, prefix='0', sample_rate=48000):
    """
    Save audio files in WAV format.

    Parameters:
        input_audio (np.ndarray): Input audio data array.
        output_audio (np.ndarray): Output audio data array (processed).
        prediction_audio: Predicted labels or values (could be additional info to save).
        model_path (str): The path where to save the audio files (should exist).
    """
    # Create the model path directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Saving input audio
    input_file_path = os.path.join(model_path, prefix + '_input_audio.wav')
    input_audio = np.array(input_audio.squeeze(), dtype=np.float32)
    write(input_file_path, sample_rate, input_audio)  # Scale to int16

    # Saving output audio
    output_file_path = os.path.join(model_path, prefix + '_output_audio.wav')
    output_audio = np.array(output_audio.squeeze(), dtype=np.float32)
    write(output_file_path, sample_rate, output_audio)  # Scale to int16

    # Saving output audio
    output_file_path = os.path.join(model_path, prefix + '_prediction_audio.wav')
    prediction_audio = np.array(prediction_audio.squeeze(), dtype=np.float32)
    write(output_file_path, sample_rate, prediction_audio)  # Scale to int16

    plot(input_audio, output_audio, prediction_audio, model_path, prefix='0')

    print(f"Audio files saved to {model_path}")

def plot(input_audio, output_audio, prediction_audio, model_path, prefix='0'):

    plt.figure(figsize=(10, 6))
    plt.plot(input_audio, 'tab:gray', label='input_audio', alpha=0.9)
    plt.plot(output_audio, 'tab:red', label='output_audio', alpha=0.7)
    plt.plot(prediction_audio, 'tab:green', label='prediction_audio', alpha=0.5)
    plt.title(' ', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.legend(fontsize=12)

    filename = prefix + '_plot.png'
    # Save plot
    plt.savefig(model_path/filename, dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()
    print(f"Plot saved to {filename}")


# Function to save losses to file
def save_losses(train_losses, val_losses, filename='losses.json'):
    losses_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(filename, 'w') as f:
        json.dump(losses_dict, f)
    print(f"Losses saved to {filename}")

# Function to plot losses
def plot_losses(train_losses, val_losses, filename='loss_plot.png'):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss Over Time', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()
    print(f"Plot saved to {filename}")


def natural_sort_key(s):
    """
    Function to use as a key for sorting strings in natural order.
    This ensures that strings with numbers are sorted in human-expected order.
    For example: ["file1", "file10", "file2"] -> ["file1", "file2", "file10"]

    Args:
        s: The string to convert to a natural sort key

    Returns:
        A list of string and integer parts that can be used for natural sorting
    """
    # Split the string into text and numeric parts
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]



def compute_lcm(x, y):
    """Compute the least common multiple of two numbers."""
    return (x * y) // math.gcd(x, y)


