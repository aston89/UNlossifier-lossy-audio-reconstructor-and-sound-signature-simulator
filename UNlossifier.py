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
def load_audio_cached(path, target_sr):
    key = str(path)

    if key in AUDIO_CACHE:
        return AUDIO_CACHE[key]

    audio, sr = librosa.load(path, sr=target_sr, mono=False)

    if audio.ndim == 1:
        audio = np.stack([audio, audio])

    audio = audio.astype(np.float32)

    AUDIO_CACHE[key] = (audio, sr)
    return audio, sr


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
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    L = x[2] + x[3]
    R = x[2] - x[3]
    return np.stack([L, R], axis=0).astype(np.float32)


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
            p = torch.stft(
                pred_lr[:, ch, :],
                n_fft,
                hop,
                window=window,
                return_complex=True
            )
            t = torch.stft(
                target_lr[:, ch, :],
                n_fft,
                hop,
                window=window,
                return_complex=True
            )

            mag_p = torch.abs(p)
            mag_t = torch.abs(t)

            mag_p = mag_p / (mag_p.mean(dim=(-2, -1), keepdim=True) + 1e-6)
            mag_t = mag_t / (mag_t.mean(dim=(-2, -1), keepdim=True) + 1e-6)

            log_p = torch.log(mag_p + 1e-6)
            log_t = torch.log(mag_t + 1e-6)

            loss_scale += (
                F.l1_loss(mag_p, mag_t) +
                F.l1_loss(log_p, log_t)
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

        self.flux_cache = {}

    def __len__(self):
        return len(self.pairs) * 10

    def compute_flux(self, audio):
        x = audio[0]

        # STFT
        S = librosa.stft(x, n_fft=512, hop_length=256)
        mag = np.abs(S)

        # spectral flux
        diff = np.diff(mag, axis=1)
        flux = np.mean(np.maximum(diff, 0.0), axis=0)

        # stereo awareness boost
        S_stereo = 0.5 * (audio[0] - audio[1])
        stereo_energy = np.mean(np.abs(S_stereo))

        flux = flux * (1 + 0.3 * stereo_energy)

        # normalize
        flux = flux + 1e-6
        flux = flux / flux.sum()

        return flux

    def sample_start(self, flux, total_len):
        frames = len(flux)
        probs = flux / flux.sum()

        idx = np.random.choice(np.arange(frames), p=probs)

        hop_audio = 256

        start = idx * hop_audio
        start = min(start, total_len - self.seg_len)

        return int(start)

    def __getitem__(self, idx):
        clean, noisy = self.pairs[random.randint(0, len(self.pairs) - 1)]

        c, _ = load_audio_cached(clean, self.sr)
        n, _ = load_audio_cached(noisy, self.sr)

        L = min(c.shape[1], n.shape[1])

        if L <= self.seg_len:
            start = 0
        else:
            key = str(clean)

            if key not in self.flux_cache:
                self.flux_cache[key] = self.compute_flux(c)

            flux = self.flux_cache[key]

            start = self.sample_start(flux, L)

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

        # preload audio
        load_audio_cached(wav, sr)
        load_audio_cached(comp_path, sr)

        pairs.append((wav, comp_path))

    dataset = AudioDataset(pairs, seg_len, sr)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    model = StereoUNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        opt.zero_grad()

        for noisy, clean in loader:

            noisy = noisy.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)

            pred = model(noisy)

            min_len = min(pred.size(-1), clean.size(-1))
            pred = pred[..., :min_len]
            clean = clean[..., :min_len]

            L_p, R_p = pred[:, 0], pred[:, 1]
            M_p, S_p = pred[:, 2], pred[:, 3]

            l_lr = F.l1_loss(
                torch.stack([L_p, R_p], dim=1),
                torch.stack([L_t, R_t], dim=1)
            )

            l_ms = F.l1_loss(
                torch.stack([M_p, S_p], dim=1),
                torch.stack([M_t, S_t], dim=1)
            )

            L_rec = M_p + S_p
            R_rec = M_p - S_p

            l_consistency = F.l1_loss(
                torch.stack([L_rec, R_rec], dim=1),
                torch.stack([L_p, R_p], dim=1)
            )

            l_stft = stft_lr_loss(
                torch.stack([L_p, R_p], dim=1),
                torch.stack([L_t, R_t], dim=1)
            )

            loss = l_lmrs + 0.20 * l_stft + 0.50 * l_consistency

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

        ckpt = f"model_{args.codec}_{args.bitrate}_{sr}_epoch{epoch:03d}.safetensors"
        sf_torch.save_model(model, os.path.join(CHECKPOINT_DIR, ckpt))

        print(f"Epoch {epoch} l_lmrs: {l_lmrs.item():.6f} l_stft: {l_stft.item():.6f} TOTAL: {(l_lmrs + 0.20 * l_stft).item():.6f}")

# =========================================================
# INFERENCE
# =========================================================
def inference(args):

    model = StereoUNet().to(DEVICE)
    sf_torch.load_model(model, str(args.model))
    model.eval()

    sr = args.sr
    audio, _ = load_audio_cached(args.input, sr)

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
                x = np.pad(x, ((0, 0), (0, pad)))

            x = to_torch(to_ms(x), DEVICE).unsqueeze(0)

            y = model(x).squeeze(0).cpu().numpy().astype(np.float32)
            y = np.clip(y, -3.0, 3.0)

            valid = min(chunk, total - i)
            y = y[:, :valid]
            win = window[:valid]

            out[:, i:i+valid] += y * win
            w[:, i:i+valid] += win

    out = out / np.clip(w, 1e-8, None)
    out = np.nan_to_num(out)
    out = np.clip(out, -3.0, 3.0)

    stereo = from_ms(out)
    stereo = np.clip(stereo, -1.0, 1.0)

    save_audio(args.output, stereo.astype(np.float32), sr)

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
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--sr", type=int, required=True)
    p.add_argument("--codec", default="mp3", choices=["mp3", "aac", "opus", "vorbis", "wav"])
    p.add_argument("--bitrate", default=None, choices=["64k", "96k", "128k", "160k", "192k", "256k", "320k"])

    args = p.parse_args()

    if args.model:
        inference(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
