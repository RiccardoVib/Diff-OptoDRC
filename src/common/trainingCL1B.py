# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato

import torch
from src.common.loaderHDF5 import DataGeneratorHDF5, highpass
from tqdm import tqdm
from src.common.CheckpointManager import CheckpointManager
from src.common.utils import save_audio_files, save_losses, plot_losses
from src.common.Sampler import SequentialWithinRecordingBatchSampler
from src.common.paper_losses import build_loss, MSELoss, ESRLoss, MultiResolutionSTFTLoss
from src.common.factory import build_model

def train_model(data_dir: str,
             filename: str,
             model_name: str,
             model_type: str = "custom",
             model_kwargs: dict = None,
             fs: int = 48000,
             epochs: int = 1,
             seq_len: int = 2048,
             batch_size: int = 22,
             cond_dim: int = 4,
             input_length: int = 5,
             buffer: int = 5,
             lr: float = 1e-4,
             return_mem_y: bool = False,
             train_the_model: bool = True,
             ):

    model_kwargs = model_kwargs or {}

    model_path = script_dir.parent.parent / "TrainedModels" / model_name

    print(f"model_name: {model_name}")
    print(f"model_path: {model_path}")
    print(f"model_type: {model_type}")

    print(f'cuda available: {torch.cuda.is_available()}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    torch.set_default_dtype(torch.float32)

    if str(device) == "cuda":
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i).name)

    loss_fn = build_loss(model_type=model_type, seq_len=seq_len).to(device)
    print(f"loss: {loss_fn}")

    model = build_model(model_type, cond_dim=cond_dim, buffer=buffer, seq_len=seq_len, **model_kwargs).to(device)
    model.set_criterion(loss_fn) if hasattr(model, "set_criterion") else None

    if model_type == "sptmod" or model_type == "gcntf":
        buffer = model.get_input_length()

    print(f"seq_len (target): {seq_len}, buffer: {buffer}, total: {input_length}")

    dataset = DataGeneratorHDF5(
        h5path=data_dir + "/" + filename + 'train.h5',
        mini_batch_size=seq_len,
        buffer=buffer,
        hop_size=None,
        mem_mode="",
        return_mem_y=return_mem_y,
    )

    dataset_val = DataGeneratorHDF5(
        h5path=data_dir + "/" + filename + 'test.h5',
        mini_batch_size=seq_len,
        buffer=buffer,
        mem_mode=""
    )

    if torch.cuda.is_available():
        num_workers = 4
    else:
        num_workers = 0


    batch_sampler_val = SequentialWithinRecordingBatchSampler(dataset_val, batch_size=batch_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(dataset_val, num_workers=num_workers, batch_sampler=batch_sampler_val, shuffle=False, pin_memory=True)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                       shuffle=True, pin_memory=True)
    lr_count = 0

    ckpt_manager = CheckpointManager(model_path / "my_checkpoints")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    print(f"Dataset length: {len(dataset)}")
    print('\n train batch_size', batch_size)
    print('\n epochs ', epochs)
    print('\n lr ', lr)
    print('\n seq_len ', seq_len)
    print('\n')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        eps=1e-07,
        betas=(0.9, 0.999),
        weight_decay=0
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=1,
        factor=0.75,
        threshold=1e-6,
    )

    if train_the_model:
        checkpoint = ckpt_manager.load_last_checkpoint(model, optimizer, scheduler, device='cpu')

        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_val_loss']
            print(f"Resuming from epoch {start_epoch}, best metric: {best_loss}")
        else:
            print("Starting training from scratch")
            best_loss = float('inf')

        avg_val_loss, avg_train_loss = 0, 0
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            model.reset_hidden_states()
            train_batches = 0
            train_loss, val_loss = 0, 0
            model.train()
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", disable=False):
                x, y, mem, c, reset = batch[:5]
                extra = {"mem_y": batch[5].to(device)} if len(batch) > 5 else {}

                x = x.to(device)
                y = y.to(device)
                mem = mem.to(device)
                c = c.to(device)

                if reset.any():
                    model.reset_hidden_states()

                loss = model.train_step(x=x, y=y, mem=mem, c=c, optimizer=optimizer, criterion=loss_fn, **extra)
                train_loss += loss
                train_batches += 1

            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)

            if (epoch + 1) % 1 == 0:
                model.reset_hidden_states()
                total_val_loss = 0
                val_batches = 0
                model.eval()
                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}", disable=True):
                        x, y, mem, c, reset = batch[:5]
                        x = x.to(device)
                        y = y.to(device)
                        mem = mem.to(device)
                        c = c.to(device)
                        if reset.any():
                            model.reset_hidden_states()
                        val_loss = model.val_step(x=x, y=y, mem=mem, c=c, criterion=loss_fn)

                        total_val_loss += val_loss
                        val_batches += 1

                avg_val_loss = total_val_loss / val_batches
                val_losses.append(avg_val_loss)

                print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
                print(f'Learning Rate {optimizer.param_groups[0]["lr"]:.2e}')
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    state_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'best_val_loss': best_loss
                    }
                    ckpt_manager.save_best_checkpoint(state_dict)
                    print(f"Epoch {epoch + 1}, Validation loss improved: ", best_loss)
                    lr_count = 0
                else:
                    lr_count += 1
                    print(f"\nlr_count {lr_count}")
                    print(f"\nEpoch {epoch + 1}, Validation loss did not improved.")
                    if lr_count == 30:
                        print("Early stopping triggered.")
                        break

            state_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_loss
            }
            ckpt_manager.save_last_checkpoint(state_dict)
            scheduler.step(avg_val_loss)

        filename_ = model_path / ('losses.json')
        save_losses(train_losses=train_losses, val_losses=val_losses, filename=filename_)
        filename_ = model_path / ('loss_plot.png')
        plot_losses(train_losses=train_losses, val_losses=val_losses, filename=filename_)

    best_checkpoint = ckpt_manager.load_best_checkpoint(model, device='cpu')
    if best_checkpoint:
        best_loss = best_checkpoint.get('best_val_loss', 0)
        print(f"Loaded best model with metric: {best_loss}")
    else:
        best_loss = 0
        print(f"Problem!!!!")

    batch_sampler_val = SequentialWithinRecordingBatchSampler(dataset_val, batch_size=108, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(dataset_val, num_workers=num_workers, batch_sampler=batch_sampler_val, shuffle=False)
    lenght_so_far = 0
    with torch.no_grad():
        model.eval()
        model.reset_hidden_states()
        inp, tar, prediction, params = [], [], [], []
        for i, (x, y, mem, c, reset) in enumerate(val_dataloader):
            x = x.to(device)
            y = y.to(device)
            mem = mem.to(device)
            c = c.to(device)

            x_ = torch.cat([mem, x], dim=1).permute(0, 2, 1)
            y = y.permute(0, 2, 1)
            if model_type == "sptmod":
                pred = []
                for b in range(x_.shape[0]):
                    pr = model(x_[b:b + 1], c[b:b + 1], y_true=y[b:b + 1])
                    pred.append(pr)
                pred = torch.cat(pred, dim=0)
            elif model_type == "mamba":
                pred = []
                for b in range(x_.shape[0]):
                    pr = model(x_[b:b + 1].permute(0, 2, 1), c[b:b + 1]).permute(0, 2, 1)
                    pred.append(pr)
                pred = torch.cat(pred, dim=0)
            elif model_type == "lstm":
                pred = []
                for b in range(x_.shape[0]):
                    pr = model(x_[b:b + 1].permute(0, 2, 1), c[b:b + 1, None, :])
                    pred.append(pr)
                pred = torch.cat(pred, dim=0)
            elif model_type == "ssmdrc_ff" or model_type == "ssmdrc_fb" or model_type == "ssmdrc_tf" or model_type == "greycomp" or model_type == "hyperssm_ff" or model_type == "hyperssm_tf" or model_type == "hyperssm_fb":
                pred = []
                for b in range(x_.shape[0]):
                    pr = model(x_[b:b + 1].permute(0, 2, 1), c[b:b + 1, None, :]).permute(0, 2, 1)
                    pred.append(pr)
                pred = torch.cat(pred, dim=0)
            else:
                pred = []
                for b in range(x_.shape[0]):
                    pr = model(x_[b:b + 1], c[b:b + 1, None, :])
                    pred.append(pr)
                pred = torch.cat(pred, dim=0)

            inp.append(x.cpu().permute(0, 2, 1))
            tar.append(y.cpu())
            prediction.append(pred.cpu())
            lenght_so_far += seq_len
            if lenght_so_far >= 4 * fs:
                break

        inp = torch.cat(inp, dim=-1)
        tar = torch.cat(tar, dim=-1)
        prediction = torch.cat(prediction, dim=-1)
        params = c
        inp = inp[:, :, :4 * fs]
        tar = tar[:, :, :4 * fs]
        prediction = prediction[:, :, :4 * fs]

    # under different objectives stay comparable
    test_loss = loss_fn(prediction.to(device), tar.to(device))
    test_terms = dict(getattr(loss_fn, "last_terms", {}))
    mse_metric = MSELoss()(prediction.to(device), tar.to(device))
    esr_metric = ESRLoss()(prediction.to(device), tar.to(device))
    mrstft_metric = MultiResolutionSTFTLoss()(prediction, tar)


    for x, y, c, prediction_audio in zip(inp, tar, params, prediction):
        z_str = '_'.join(map(str, c.tolist()))
        prediction_audio = highpass(prediction_audio.detach().flatten(), cutoff=20, sample_rate=48000)
        x = x.detach().flatten()
        y = y.detach().flatten()
        save_audio_files(x[:], y[:], prediction_audio[:], model_path / z_str, sample_rate=fs)

    with open(model_path/'test_loss.txt', 'w') as f:
        f.write(f'Model type: {model_type}\n')
        f.write(f'Best val loss: {best_loss:.8f}\n')
        f.write(f'Test loss (training objective): {test_loss:.8f}\n')
        for k, v in test_terms.items():
            f.write(f'    {k}: {v:.8f}\n')
        f.write(f'Test MSE  (common metric): {mse_metric:.8f}\n')
        f.write(f'Test ESR  (common metric): {esr_metric:.8f}\n')
        f.write(f'Test MRSTFT  (common metric): {mrstft_metric:.8f}\n')
        f.write(f'Number of parameters: {total_params}\n')
        f.write(f'Dataset length: {len(dataset)}\n')
        f.write(f'train batch_size: {batch_size}\n')
        f.write(f'lr {lr}\n')
        f.write(f'seq_len: {seq_len}\n')

    return 42


if __name__ == '__main__':

    import os
    from pathlib import Path
    from src.common.utils import find_folder_upward
    from configs import PAPER_CONFIGS_CL1B

    current_dir = Path(os.getcwd())
    print(f"current_dir: {current_dir}")
    files_dir = find_folder_upward(folder_name="Files", start_path=current_dir)
    ROOT_DIR = files_dir / "CL1B"
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent

    filename = 'CL1B_analog_'
    fs = 48000
    lr = 3e-4
    epochs = 100

    model_types = ["tcn", "gcntf", "sptmod", "mamba", "lstm", "diffopto", "greybox"]

    for model_type in model_types:
        cfg = PAPER_CONFIGS_CL1B[model_type]

        model_name = "_".join([model_type, filename, str(cfg["seq_len"])])

        train_model(filename=filename,
                    data_dir=str(ROOT_DIR),
                    model_name=model_name,
                    model_type=model_type,
                    model_kwargs=cfg["model_kwargs"],
                    fs=fs,
                    epochs=epochs,
                    seq_len=cfg["seq_len"],
                    cond_dim=4,
                    batch_size=cfg["batch_size"],
                    input_length=cfg["input_length"],
                    buffer=cfg["buffer"],
                    lr=lr,
                    train_the_model=False,
                    )
