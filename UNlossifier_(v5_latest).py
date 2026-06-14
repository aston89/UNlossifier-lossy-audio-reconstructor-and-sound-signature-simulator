import os
import argparse
import subprocess
import random
import hashlib
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch as sf_torch

torch.set_float32_matmul_precision("high")

# =========================================================
# CONFIG
# =========================================================
TEMP_DIR = "./cache_ffmpeg"
CACHE_DIR = "./cache_numpy"
CHECKPOINT_DIR = "./checkpoints"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_LEN_SEC = 4   # more seconds means higher vram usage
AUDIO_CACHE = {}  # in-RAM cache of loaded .npy arrays


# =========================================================
# CACHE PATHS
# =========================================================
def cache_key(path, target_sr, codec=None, bitrate=None, tag="raw"):
    p = Path(path).resolve()
    st = p.stat()

    seed = f"{p}|{st.st_size}|{st.st_mtime_ns}|{target_sr}|{codec}|{bitrate}|{tag}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def get_cache_path(path, target_sr, codec=None, bitrate=None, tag="raw"):
    p = Path(path)
    stem = p.stem

    key = cache_key(path, target_sr, codec, bitrate, tag)
    return Path(CACHE_DIR) / f"{stem}__{tag}__{codec}_{bitrate}_{target_sr}__{key}.npy"


# =========================================================
# FFmpeg DECODER -> NUMPY CACHE
# =========================================================
def decode_audio_ffmpeg(path, target_sr):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "2",
        "-ar", str(target_sr),
        "-"
    ]

    p = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    audio = np.frombuffer(p.stdout, dtype=np.float32)

    if audio.size == 0:
        raise RuntimeError(f"FFmpeg returned empty audio for: {path}")

    if audio.size % 2 != 0:
        audio = audio[:-1]

    audio = audio.reshape(-1, 2).T  # (2, time)
    return audio.astype(np.float32)


def load_audio_cached(path, target_sr, codec=None, bitrate=None):
    cache_path = get_cache_path(path, target_sr, codec, bitrate)

    if cache_path.exists():
        audio = np.load(cache_path, mmap_mode="r")
        return audio, target_sr

    audio = decode_audio_ffmpeg(path, target_sr)
    np.save(cache_path, audio)
    return audio, target_sr


def prebuild_audio_cache(paths, target_sr):
    for p in paths:
        load_audio_cached(p, target_sr)


# =========================================================
# SAVE AUDIO
# =========================================================
def save_audio(path, audio, sr):
    audio = audio.T if audio.shape[0] == 2 else audio
    sf.write(path, audio, sr, subtype="FLOAT")


# =========================================================
# MS (NUMPY-FIRST)
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
# FFmpeg ENCODE
# =========================================================
def get_codec_extension(codec):
    return {
        "mp3": ".mp3",
        "aac": ".m4a",
        "opus": ".opus",
        "vorbis": ".ogg",
        "wav": ".wav",
    }.get(codec, "." + codec)


def compress_audio(input_path, output_path, bitrate, sr, codec):
    if codec == "wav":
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-nostdin",
            "-i", str(input_path),
            "-ar", str(sr),
            "-ac", "2",
            "-c:a", "pcm_s16le",
            str(output_path),
        ]
    else:
        if bitrate is None:
            raise ValueError(f"bitrate is required for codec={codec}")

        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-nostdin",
            "-i", str(input_path),
            "-b:a", bitrate,
            "-ar", str(sr),
            "-ac", "2",
            "-codec:a", codec,
            str(output_path),
        ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )


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
def lr_loss(pred, target):
    L_p, R_p = pred[:, 0], pred[:, 1]
    L_t, R_t = target[:, 0], target[:, 1]
    return F.l1_loss(
        torch.stack([L_p, R_p], dim=1),
        torch.stack([L_t, R_t], dim=1)
    )


def ms_loss(pred, target):
    M_p, S_p = pred[:, 2], pred[:, 3]
    M_t, S_t = target[:, 2], target[:, 3]
    return F.l1_loss(
        torch.stack([M_p, S_p], dim=1),
        torch.stack([M_t, S_t], dim=1)
    )


def consistency_loss(pred):
    L = pred[:, 0]
    R = pred[:, 1]
    M = pred[:, 2]
    S = pred[:, 3]

    L_from_ms = M + S
    R_from_ms = M - S

    M_from_lr = 0.5 * (L + R)
    S_from_lr = 0.5 * (L - R)

    return (
        F.l1_loss(L, L_from_ms) +
        F.l1_loss(R, R_from_ms) +
        F.l1_loss(M, M_from_lr) +
        F.l1_loss(S, S_from_lr)
    ) * 0.25


def coherency_loss(pred, target, eps=1e-6):
    scales = (1, 2, 4, 8, 16, 32)

    energy_term = 0.0
    corr_term = 0.0
    wave_term = 0.0

    # 1) ENERGY JITTER (physical stability)
    for s in scales:
        ep = pred[..., s:] - pred[..., :-s]
        et = target[..., s:] - target[..., :-s]

        ep = (ep ** 2).mean(dim=-1)
        et = (et ** 2).mean(dim=-1)

        energy_term += F.l1_loss(ep, et)

    energy_term /= len(scales)

    # 2) SOFT CORRELATION (structural movements)
    for s in scales:
        p = pred[..., s:] * pred[..., :-s]
        t = target[..., s:] * target[..., :-s]

        corr_term += F.l1_loss(p.mean(dim=-1), t.mean(dim=-1))

    corr_term /= len(scales)

    # 3) STOCHASTIC MICRO-WAVE (anti-overfit)
    n_samples = 4
    for _ in range(n_samples):
        s = random.choice(scales)
        offset = random.randint(0, s)

        dp = pred[..., offset + s:] - pred[..., offset:-s]
        dt = target[..., offset + s:] - target[..., offset:-s]

        wave_term += F.l1_loss(dp, dt)

    wave_term /= n_samples

    # FINAL BLEND 
    return (
        energy_term +
        corr_term +
        wave_term
    ) / 3


# =========================================================
# DATASET
# =========================================================
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, seg_len, sr):
        self.pairs = [(str(c), str(n)) for c, n in pairs]
        self.seg_len = seg_len
        self.sr = sr
        self.flux_cache = {}

    def __len__(self):
        return len(self.pairs) * 10

    def compute_flux(self, audio):
        x = np.asarray(audio[0], dtype=np.float32)

        S = librosa.stft(x, n_fft=512, hop_length=256)
        mag = np.abs(S)

        diff = np.diff(mag, axis=1)
        flux = np.mean(np.maximum(diff, 0.0), axis=0)

        stereo_energy = np.mean(np.abs(0.5 * (audio[0] - audio[1])))
        flux = flux * (1 + 0.3 * stereo_energy)

        flux = flux + 1e-6
        flux = flux / flux.sum()
        return flux.astype(np.float32)

    def sample_start(self, flux, total_len):
        frames = len(flux)
        probs = flux / flux.sum()

        idx = np.random.choice(np.arange(frames), p=probs)

        hop_audio = 256
        start = idx * hop_audio
        start = min(start, max(0, total_len - self.seg_len))
        return int(start)

    def __getitem__(self, idx):
        clean_path, noisy_path = self.pairs[random.randint(0, len(self.pairs) - 1)]

        c, _ = load_audio_cached(clean_path, self.sr)
        n, _ = load_audio_cached(noisy_path, self.sr)

        L = min(c.shape[1], n.shape[1])

        if L <= self.seg_len:
            start = 0
        else:
            key = f"{Path(clean_path).resolve()}|{self.sr}|flux"
            if key not in self.flux_cache:
                self.flux_cache[key] = self.compute_flux(c)
            start = self.sample_start(self.flux_cache[key], L)

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
    ctx = sr // 2
    train_len = seg_len + ctx * 2

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

    if not pairs:
        raise RuntimeError(f"No .wav files found in {input_path}")

    # Predecode/cache once
    all_paths = []
    for clean_path, noisy_path in pairs:
        all_paths.append(clean_path)
        all_paths.append(noisy_path)
    prebuild_audio_cache(all_paths, sr)

    dataset = AudioDataset(pairs, train_len, sr)
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
        model.train()

        for noisy, clean in loader:
            noisy = noisy.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)

            pred = model(noisy)

            min_len = min(pred.size(-1), clean.size(-1))
            pred = pred[..., :min_len]
            clean = clean[..., :min_len]

            ctx = sr // 2
            pred = pred[..., ctx:-ctx]
            clean = clean[..., ctx:-ctx]

            L_p, R_p = pred[:, 0], pred[:, 1]
            M_p, S_p = pred[:, 2], pred[:, 3]

            L_t, R_t = clean[:, 0], clean[:, 1]
            M_t, S_t = clean[:, 2], clean[:, 3]

            L_rec = M_p + S_p
            R_rec = M_p - S_p
            
            l_lr = lr_loss(pred, clean)
            
            l_ms = ms_loss(pred, clean)

            l_consistency = consistency_loss(pred)

            l_coherency = coherency_loss(pred, clean)

            # Total losses weighting
            loss = 0.50 * l_lr + 0.50 * l_ms + 0.50 * l_consistency + 0.50 * l_coherency

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        ckpt = f"model_{args.codec}_{args.bitrate}_{sr}_epoch{epoch:03d}.safetensors"
        sf_torch.save_model(model, os.path.join(CHECKPOINT_DIR, ckpt))

        print(
            f"Epoch {epoch} "
            f"l_lr: {l_lr.item():.6f} "
            f"l_ms: {l_ms.item():.6f} "
            f"l_consistency: {l_consistency.item():.6f} "
            f"l_coherency: {l_coherency.item():.6f} "
            f"TOTAL: {loss.item():.6f}"
        )


# =========================================================
# INFERENCE (SAFE LR + MS ENSEMBLE)
# =========================================================
def inference(args):
    model = StereoUNet().to(DEVICE)
    sf_torch.load_model(model, str(args.model))
    model.eval()

    sr = args.sr
    audio, _ = load_audio_cached(args.input, sr)

    total = audio.shape[1]
    chunk = SEG_LEN_SEC * sr
    ctx = sr // 2
    step = chunk // 2  # 50% overlap

    # padding safety
    padded = np.pad(audio, ((0, 0), (ctx, ctx)), mode="edge")

    out = np.zeros((2, total), dtype=np.float32)
    w = np.zeros((2, total), dtype=np.float32)

    window = np.hanning(chunk).astype(np.float32)
    eps = 1e-8

    w_lr = 0.50
    w_ms = 0.50

    expected_in = chunk + 2 * ctx

    with torch.no_grad():
        for i in range(0, total, step):
            # ctx windows pre-post
            x = padded[:, i:i + expected_in]

            # padding if near end.
            if x.shape[1] < expected_in:
                pad = expected_in - x.shape[1]
                x = np.pad(x, ((0, 0), (0, pad)), mode="edge")

            x_t = to_torch(to_ms(x), DEVICE).unsqueeze(0)
            y = model(x_t).squeeze(0).cpu().numpy().astype(np.float32)

            # LR
            L1, R1 = y[0], y[1]

            # MS -> LR
            M, S = y[2], y[3]
            L2 = M + S
            R2 = M - S

            L = w_lr * L1 + w_ms * L2
            R = w_lr * R1 + w_ms * R2

            stereo = np.stack([L, R], axis=0)

            # cut ctx
            stereo = stereo[:, ctx:ctx + chunk]

            valid = min(chunk, total - i)
            win = window[:valid]

            out[:, i:i + valid] += stereo[:, :valid] * win
            w[:, i:i + valid] += win

    out = out / np.clip(w, eps, None)
    out = np.nan_to_num(out)
    out = np.clip(out, -1.0, 1.0)

    save_audio(args.output, out.astype(np.float32), sr)
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
