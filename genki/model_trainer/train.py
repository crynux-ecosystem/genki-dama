import argparse
import json
import os
import random

import torch
import torch.nn
import torch.nn.functional as F
import wandb
from audiocraft.data.audio import audio_info, audio_read
from audiocraft.data.audio_dataset import AudioMeta
from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.music_dataset import MusicInfo
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import WavCondition
from audiocraft.models.loaders import load_lm_model_ckpt, load_compression_model_ckpt
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler


class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        sample_rate: int,
        channels: int,
        duration: float,
        device: torch.device,
    ):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = duration
        self.device = device

        self.data_map = []

        dir_map = os.listdir(data_dir)
        for d in dir_map:
            name, ext = os.path.splitext(d)
            if ext == ".wav":
                attr_file = os.path.join(data_dir, name + ".json")
                if os.path.exists(attr_file):
                    self.data_map.append(
                        {
                            "audio": os.path.join(data_dir, d),
                            "attr": attr_file,
                        }
                    )
                else:
                    raise ValueError(f"No label file for {name}")

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio_path = data["audio"]
        attr_path = data.get("attr", "")

        info = audio_info(audio_path)

        rng = torch.Generator()
        rng.manual_seed(idx + random.randint(0, 2**24) * len(self))

        max_seek = min(0, info.duration - self.duration)
        seek_time = torch.rand(1, generator=rng).item() * max_seek

        wav, sr = audio_read(
            audio_path, seek_time=seek_time, duration=self.duration, pad=False
        )
        wav = convert_audio(wav, sr, self.sample_rate, self.channels)

        n_frames = wav.shape[-1]
        target_frames = int(self.duration * self.sample_rate)
        channels = wav.shape[0]

        meta = AudioMeta(
            path=audio_path,
            duration=info.duration,
            sample_rate=sr,
        )

        attrs = {
            "meta": meta,
            "seek_time": seek_time,
            "n_frames": n_frames,
            "total_frames": target_frames,
            "channels": channels,
        }

        with open(attr_path, mode="r", encoding="utf-8") as f:
            text_attrs = json.load(f)

        attrs.update(text_attrs)

        music_info = MusicInfo.from_dict(attrs)
        music_info.self_wav = WavCondition(
            torch.zeros((1, 1, 1), device=self.device),
            torch.tensor([0], device=self.device),
            sample_rate=[self.sample_rate],
            path=[None],
        )
        condition = music_info.to_condition_attributes()

        return wav.to(self.device), condition


def count_nans(tensor):
    nan_mask = torch.isnan(tensor)
    num_nans = torch.sum(nan_mask).item()
    return num_nans


def fixnan(tensor: torch.Tensor):
    nan_mask = torch.isnan(tensor)
    result = torch.where(nan_mask, torch.zeros_like(tensor), tensor)

    return result


def one_hot_encode(tensor, num_classes=2048):
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], num_classes))

    for i in range(shape[0]):
        for j in range(shape[1]):
            index = tensor[i, j].item()
            one_hot[i, j, index] = 1

    return one_hot


def train(
    dataset_path: str,
    checkpoint_path: str,
    save_path: str,
    model_id: str,
    lr: float,
    epochs: int,
    use_wandb: bool = False,
    save_epoch: int = 0,
    grad_acc: int = 8,
    use_scaler: bool = False,
    weight_decay: float = 1e-5,
    warmup_steps: int = 10,
    batch_size: int = 10,
):
    if use_wandb:
        run = wandb.init(project="audiocraft")

    model = MusicGen.get_pretrained(model_id)
    model.lm = model.lm.to(torch.float32)  # important
    model.lm.train()

    device = model.device

    dataset = AudioDataset(
        dataset_path,
        model.sample_rate,
        model.audio_channels,
        model.max_duration,
        device,
    )

    def collate_fn(data):
        audios, conditions = zip(*data)
        audio_batch = torch.stack(audios, dim=0)
        return audio_batch, list(conditions)

    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    learning_rate = lr
    model.lm.train()

    scaler = torch.cuda.amp.GradScaler()

    # from paper
    optimizer = AdamW(
        model.lm.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        warmup_steps,
        int(epochs * len(train_dataloader) / grad_acc),
    )

    num_epochs = epochs

    save_models = save_epoch > 0

    os.makedirs(checkpoint_path, exist_ok=True)

    current_step = 0

    for epoch in range(num_epochs):
        for batch_idx, (audios, conditions) in enumerate(train_dataloader):
            optimizer.zero_grad()

            with torch.no_grad():
                codes, _ = model.compression_model.encode(audios)

            with torch.autocast(device_type=device.type, dtype=torch.float16):

                lm_output = model.lm.compute_predictions(
                    codes=codes,
                    conditions=conditions,
                )

                logits = lm_output.logits
                targets = codes
                mask = lm_output.mask

                B, K, T, card = logits.shape
                assert targets.shape == logits.shape[:-1]
                assert targets.shape == mask.shape
                ce = torch.zeros([], device=device)

                for k in range(K):
                    logits_k = logits[:, k, ...].contiguous().view(-1, card)
                    targets_k = targets[:, k, ...].contiguous().view(-1)
                    mask_k = mask[:, k, ...].contiguous().view(-1)

                    ce_logits = logits_k[mask_k]
                    ce_targets = targets_k[mask_k]
                    ce_k = F.cross_entropy(ce_logits, ce_targets)
                    ce += ce_k

                loss = ce / K
                loss = loss / grad_acc

            current_step += 1

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            print(
                f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}"
            )

            if current_step % grad_acc == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.lm.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.lm.parameters(), 1.0)
                    optimizer.step()

                # assert count_nans(masked_logits) == 0

                if use_wandb:
                    run.log(
                        {
                            "loss": loss.item(),
                        }
                    )

                scheduler.step()

        if save_models:
            if (epoch + 1) % save_epoch == 0:
                torch.save(
                    model.lm.state_dict(), f"{checkpoint_path}/lm_{current_step}.pt"
                )

    torch.save(model.lm.state_dict(), f"{checkpoint_path}/lm_final.pt")

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    lm_state_dict = {
        k: v.to(dtype=torch.float16, device=torch.device("cpu"))
        for k, v in model.lm.state_dict().items()
    }

    print(f"save finetuned model to {save_path}")

    # release the GPU VRAM of training model
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    lm_pkg = load_lm_model_ckpt(model_id)
    lm_pkg["best_state"] = lm_state_dict
    torch.save(lm_pkg, os.path.join(save_path, "state_dict.bin"))

    compression_pkg = load_compression_model_ckpt(model_id)
    torch.save(compression_pkg, os.path.join(save_path, "compression_state_dict.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path", type=str, required=True, help="training dataset path"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        default="checkpoints",
        help="path to save the checkpoints",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=False,
        default="models",
        help="path to save the finetuned model",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=False,
        default="facebook/musicgen-small",
        help="base audiocraft model id to finetune",
    )
    parser.add_argument(
        "--lr", type=float, required=False, default=1e-4, help="initial learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, required=False, default=100, help="total training epochs"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="whether to use wandb to log"
    )
    parser.add_argument(
        "--save_epoch",
        type=int,
        required=False,
        default=0,
        help="interval epochs to save checkpoints",
    )
    parser.add_argument(
        "--weight_decay", type=float, required=False, default=1e-4, help="weight decay"
    )
    parser.add_argument(
        "--grad_acc",
        type=int,
        required=False,
        default=2,
        help="gradiant accumulation steps",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        required=False,
        default=16,
        help="gradient scheduler warmup steps",
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=4, help="batch size"
    )
    parser.add_argument(
        "--no_scaler", action="store_true", help="whether to disable gradient scaler"
    )
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        model_id=args.model_id,
        lr=args.lr,
        epochs=args.epochs,
        use_wandb=args.use_wandb,
        batch_size=args.batch_size,
        grad_acc=args.grad_acc,
        use_scaler=(not args.no_scaler),
        weight_decay=args.weight_decay,
        save_epoch=args.save_epoch,
        warmup_steps=args.warmup_steps,
    )
