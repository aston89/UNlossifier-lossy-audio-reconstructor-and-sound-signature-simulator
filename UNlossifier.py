import os
import argparse
import subprocess
import random
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch as sf_torch

# =========================================================
# CONFIG
# =========================================================
TEMP_DIR = "./temp_audio"
CHECKPOINT_DIR = "./"
os.makedirs(TEMP_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_LEN_SEC = 4
AUDIO_CACHE = {}

# =========================================================
# AUDIO UTILS
# =========================================================
def load_audio(path, target_sr):
    audio, sr = librosa.load(path, sr=target_sr, mono=False)

    if audio.ndim == 1:
        audio = np.stack([audio, audio])

    return audio.astype(np.float32), sr


def save_audio(path, audio, sr):
    audio = audio.T if audio.shape[0] == 2 else audio
    sf.write(path, audio, sr)


# =========================================================
# MS (NUMPY-FIRST, STABLE)
# =========================================================
def to_ms(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    L = x[0]
    R = x[1]
    M = 0.5 * (L + R)
    S = 0.5 * (L - R)

    return np.stack([L, R, M, S], axis=0).astype(np.float32)


def from_ms(x):
    # x: numpy [4, T]
    L = x[2] + x[3]
    R = x[2] - x[3]
    return np.stack([L, R], axis=0)


def to_torch(x, device):
    return torch.from_numpy(x).float().to(device)


# =========================================================
# FFmpeg
# =========================================================
def get_codec_extension(codec):
    return {
        "mp3": ".mp3",
        "aac": ".m4a",
        "opus": ".opus",
        "vorbis": ".ogg"
    }.get(codec, "." + codec)


def compress_audio(input_path, output_path, bitrate, sr, codec):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-b:a", bitrate,
        "-ar", str(sr),
        "-ac", "2",
        "-codec:a", codec,
        str(output_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


# =========================================================
# MODEL
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 7, padding=3),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 5, padding=2),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class StereoUNet(nn.Module):
    def __init__(self, base=128):
        super().__init__()

        self.enc1 = ConvBlock(4, base)
        self.enc2 = ConvBlock(base, base)

        self.mid = ConvBlock(base, base)

        self.dec2 = ConvBlock(base * 2, base)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv1d(base, 4, 7, padding=3)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        x = self.mid(e2)

        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        return self.out(x)


# =========================================================
# LOSS
# =========================================================
def stft_lr_loss(pred_lr, target_lr):
    fft_sizes = [128, 1024, 2048]
    losses = []

    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred_lr.device)

        loss_scale = 0.0

        for ch in [0, 1]:
            p = torch.stft(pred_lr[:, ch, :], n_fft, hop, window=window, return_complex=True)
            t = torch.stft(target_lr[:, ch, :], n_fft, hop, window=window, return_complex=True)

            mag_p = torch.abs(p)
            mag_t = torch.abs(t)

            loss_scale += (
                F.l1_loss(mag_p, mag_t) +
                F.l1_loss(torch.log(mag_p + 1e-7), torch.log(mag_t + 1e-7))
            )

        losses.append(loss_scale / 2)

    return 0.3 * losses[0] + 0.5 * losses[1] + 0.2 * losses[2]


# =========================================================
# DATASET
# =========================================================
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, seg_len, sr):
        self.pairs = pairs
        self.seg_len = seg_len
        self.sr = sr

    def __len__(self):
        return len(self.pairs) * 10

    def __getitem__(self, idx):
        clean, noisy = self.pairs[random.randint(0, len(self.pairs) - 1)]

        c, n = load_audio(clean, self.sr)[0], load_audio(noisy, self.sr)[0]

        L = min(c.shape[1], n.shape[1])

        if L <= self.seg_len:
            start = 0
        else:
            start = random.randint(0, L - self.seg_len)

        c = c[:, start:start + self.seg_len]
        n = n[:, start:start + self.seg_len]

        c = to_ms(c)
        n = to_ms(n)

        return n, c


# =========================================================
# TRAIN
# =========================================================
def train(args):

    sr = args.sr
    seg_len = SEG_LEN_SEC * sr

    input_path = Path(args.input)
    pairs = []

    for wav in input_path.glob("*.wav"):

        comp_path = Path(TEMP_DIR) / (
            f"{wav.stem}_{args.codec}_{args.bitrate}_{sr}"
            + get_codec_extension(args.codec)
        )

        if not comp_path.exists():
            compress_audio(wav, comp_path, args.bitrate, sr, args.codec)

        pairs.append((wav, comp_path))

    dataset = AudioDataset(pairs, seg_len, sr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = StereoUNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):

        for noisy, clean in loader:

            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            pred = model(noisy)

            min_len = min(pred.size(-1), clean.size(-1))
            pred = pred[..., :min_len]
            clean = clean[..., :min_len]

            L_p, R_p, M_p, S_p = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            L_t, R_t, M_t, S_t = clean[:, 0], clean[:, 1], clean[:, 2], clean[:, 3]

            l_lr = F.l1_loss(
                torch.stack([L_p, R_p], dim=1),
                torch.stack([L_t, R_t], dim=1)
            )

            l_ms = F.l1_loss(
                torch.stack([M_p, S_p], dim=1),
                torch.stack([M_t, S_t], dim=1)
            )

            l_stft = stft_lr_loss(
                torch.stack([pred[:, 0], pred[:, 1]], dim=1),
                torch.stack([clean[:, 0], clean[:, 1]], dim=1)
            )

            loss = l_lr + l_ms + l_stft

            opt.zero_grad()
            loss.backward()
            opt.step()

        ckpt = f"model_{args.codec}_{args.bitrate}_{sr}_epoch{epoch:03d}.safetensors"
        sf_torch.save_model(model, os.path.join(CHECKPOINT_DIR, ckpt))

        print(f"Epoch {epoch} l_lr: {l_lr.item():.6f} l_ms: {l_ms.item():.6f} l_stft: {l_stft.item():.6f}")


# =========================================================
# INFERENCE (FIXED PIPELINE)
# =========================================================
def inference(args):

    model = StereoUNet().to(DEVICE)
    sf_torch.load_model(model, str(args.model))
    model.eval()

    sr = args.sr
    audio, _ = load_audio(args.input, sr)

    total = audio.shape[1]
    chunk = SEG_LEN_SEC * sr
    step = chunk - int(chunk * 0.1)

    out = np.zeros((4, total), dtype=np.float32)
    w = np.zeros((4, total), dtype=np.float32)

    window = np.hanning(chunk).astype(np.float32)

    with torch.no_grad():
        for i in range(0, total, step):

            x = audio[:, i:i+chunk]

            if x.shape[1] < chunk:
                pad = chunk - x.shape[1]
                x = np.pad(x, ((0,0),(0,pad)))

            x = to_torch(to_ms(x), DEVICE).unsqueeze(0)

            y = model(x).squeeze(0).cpu().numpy()
            y = y[:, :min(chunk, total - i)]

            win = window[:y.shape[1]]

            out[:, i:i+y.shape[1]] += y * win
            w[:, i:i+y.shape[1]] += win

    out /= np.clip(w, 1e-8, None)

    stereo = from_ms(out)
    save_audio(args.output, stereo, sr)

    print("Saved:", args.output)


# =========================================================
# MAIN
# =========================================================
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True)
    p.add_argument("--output", default="restored.wav")
    p.add_argument("--model", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--sr", type=int, required=True)
    p.add_argument("--codec", default="mp3")
    p.add_argument("--bitrate", default="96k")

    args = p.parse_args()

    if args.model:
        inference(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
import os
import argparse
import subprocess
import random
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch as sf_torch

# =========================================================
# CONFIG
# =========================================================
TEMP_DIR = "./temp_audio"
CHECKPOINT_DIR = "./"
os.makedirs(TEMP_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_LEN_SEC = 4
AUDIO_CACHE = {}

# =========================================================
# AUDIO UTILS
# =========================================================
def load_audio(path, target_sr):
    audio, sr = librosa.load(path, sr=target_sr, mono=False)

    if audio.ndim == 1:
        audio = np.stack([audio, audio])

    return audio.astype(np.float32), sr


def save_audio(path, audio, sr):
    audio = audio.T if audio.shape[0] == 2 else audio
    sf.write(path, audio, sr)


# =========================================================
# MS (NUMPY-FIRST, STABLE)
# =========================================================
def to_ms(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    L = x[0]
    R = x[1]
    M = 0.5 * (L + R)
    S = 0.5 * (L - R)

    return np.stack([L, R, M, S], axis=0).astype(np.float32)


def from_ms(x):
    # x: numpy [4, T]
    L = x[2] + x[3]
    R = x[2] - x[3]
    return np.stack([L, R], axis=0)


def to_torch(x, device):
    return torch.from_numpy(x).float().to(device)


# =========================================================
# FFmpeg
# =========================================================
def get_codec_extension(codec):
    return {
        "mp3": ".mp3",
        "aac": ".m4a",
        "opus": ".opus",
        "vorbis": ".ogg"
    }.get(codec, "." + codec)


def compress_audio(input_path, output_path, bitrate, sr, codec):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-b:a", bitrate,
        "-ar", str(sr),
        "-ac", "2",
        "-codec:a", codec,
        str(output_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


# =========================================================
# MODEL
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 7, padding=3),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 5, padding=2),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class StereoUNet(nn.Module):
    def __init__(self, base=128):
        super().__init__()

        self.enc1 = ConvBlock(4, base)
        self.enc2 = ConvBlock(base, base)

        self.mid = ConvBlock(base, base)

        self.dec2 = ConvBlock(base * 2, base)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv1d(base, 4, 7, padding=3)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        x = self.mid(e2)

        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        return self.out(x)


# =========================================================
# LOSS (unchanged logic)
# =========================================================
def stft_lr_loss(pred_lr, target_lr):
    fft_sizes = [128, 1024, 2048]
    losses = []

    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred_lr.device)

        loss_scale = 0.0

        for ch in [0, 1]:
            p = torch.stft(pred_lr[:, ch, :], n_fft, hop, window=window, return_complex=True)
            t = torch.stft(target_lr[:, ch, :], n_fft, hop, window=window, return_complex=True)

            mag_p = torch.abs(p)
            mag_t = torch.abs(t)

            loss_scale += (
                F.l1_loss(mag_p, mag_t) +
                F.l1_loss(torch.log(mag_p + 1e-7), torch.log(mag_t + 1e-7))
            )

        losses.append(loss_scale / 2)

    return 0.3 * losses[0] + 0.5 * losses[1] + 0.2 * losses[2]


# =========================================================
# DATASET
# =========================================================
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, seg_len, sr):
        self.pairs = pairs
        self.seg_len = seg_len
        self.sr = sr

    def __len__(self):
        return len(self.pairs) * 10

    def __getitem__(self, idx):
        clean, noisy = self.pairs[random.randint(0, len(self.pairs) - 1)]

        c, n = load_audio(clean, self.sr)[0], load_audio(noisy, self.sr)[0]

        L = min(c.shape[1], n.shape[1])

        if L <= self.seg_len:
            start = 0
        else:
            start = random.randint(0, L - self.seg_len)

        c = c[:, start:start + self.seg_len]
        n = n[:, start:start + self.seg_len]

        c = to_ms(c)
        n = to_ms(n)

        return n, c


# =========================================================
# TRAIN
# =========================================================
def train(args):

    sr = args.sr
    seg_len = SEG_LEN_SEC * sr

    input_path = Path(args.input)
    pairs = []

    for wav in input_path.glob("*.wav"):

        comp_path = Path(TEMP_DIR) / (
            f"{wav.stem}_{args.codec}_{args.bitrate}_{sr}"
            + get_codec_extension(args.codec)
        )

        if not comp_path.exists():
            compress_audio(wav, comp_path, args.bitrate, sr, args.codec)

        pairs.append((wav, comp_path))

    dataset = AudioDataset(pairs, seg_len, sr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = StereoUNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):

        for noisy, clean in loader:

            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            pred = model(noisy)

            min_len = min(pred.size(-1), clean.size(-1))
            pred = pred[..., :min_len]
            clean = clean[..., :min_len]

            L_p, R_p, M_p, S_p = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            L_t, R_t, M_t, S_t = clean[:, 0], clean[:, 1], clean[:, 2], clean[:, 3]

            l_lr = F.l1_loss(
                torch.stack([L_p, R_p], dim=1),
                torch.stack([L_t, R_t], dim=1)
            )

            l_ms = F.l1_loss(
                torch.stack([M_p, S_p], dim=1),
                torch.stack([M_t, S_t], dim=1)
            )

            l_stft = stft_lr_loss(
                torch.stack([pred[:, 0], pred[:, 1]], dim=1),
                torch.stack([clean[:, 0], clean[:, 1]], dim=1)
            )

            loss = l_lr + l_ms + l_stft

            opt.zero_grad()
            loss.backward()
            opt.step()

        ckpt = f"model_{args.codec}_{args.bitrate}_{sr}_epoch{epoch:03d}.safetensors"
        sf_torch.save_model(model, os.path.join(CHECKPOINT_DIR, ckpt))

        print(f"Epoch {epoch} llr: {l_lr.item():.6f} l_ms: {l_ms.item():.6f} l_stft: {l_stft.item():.6f}")


# =========================================================
# INFERENCE (FIXED PIPELINE)
# =========================================================
def inference(args):

    model = StereoUNet().to(DEVICE)
    sf_torch.load_model(model, str(args.model))
    model.eval()

    sr = args.sr
    audio, _ = load_audio(args.input, sr)

    total = audio.shape[1]
    chunk = SEG_LEN_SEC * sr
    step = chunk - int(chunk * 0.1)

    out = np.zeros((4, total), dtype=np.float32)
    w = np.zeros((4, total), dtype=np.float32)

    window = np.hanning(chunk).astype(np.float32)

    with torch.no_grad():
        for i in range(0, total, step):

            x = audio[:, i:i+chunk]

            if x.shape[1] < chunk:
                pad = chunk - x.shape[1]
                x = np.pad(x, ((0,0),(0,pad)))

            x = to_torch(to_ms(x), DEVICE).unsqueeze(0)

            y = model(x).squeeze(0).cpu().numpy()
            y = y[:, :min(chunk, total - i)]

            win = window[:y.shape[1]]

            out[:, i:i+y.shape[1]] += y * win
            w[:, i:i+y.shape[1]] += win

    out /= np.clip(w, 1e-8, None)

    stereo = from_ms(out)
    save_audio(args.output, stereo, sr)

    print("Saved:", args.output)


# =========================================================
# MAIN
# =========================================================
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True)
    p.add_argument("--output", default="restored.wav")
    p.add_argument("--model", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--sr", type=int, required=True)
    p.add_argument("--codec", default="mp3")
    p.add_argument("--bitrate", default="96k")

    args = p.parse_args()

    if args.model:
        inference(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
