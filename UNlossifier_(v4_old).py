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

import sys
import json
import time
import threading

torch.set_float32_matmul_precision("high")

# =========================================================
# CONFIG
# =========================================================
TEMP_DIR = "./cache_ffmpeg"
CACHE_DIR = "./cache_numpy"
FLUX_CACHE_DIR = "./cache_flux"
CHECKPOINT_DIR = "./checkpoints"

RESUME_STATE_PATH = Path(CHECKPOINT_DIR) / "resume_state.pt"
RESUME_META_PATH = Path(CHECKPOINT_DIR) / "resume_state.json"

PAUSE_EVENT = threading.Event()
STOP_EVENT = threading.Event()

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FLUX_CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_LEN_SEC = 4   # chunk seconds, more means higher vram usage 
CTX_RATIO = 0.25   # 1.0 = same lenght of seg_len in S

AUDIO_MEM_CACHE = {}
FLUX_MEM_CACHE = {}


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
# STOP PAUSE RESUME
# =========================================================
def _make_resume_cfg(args, sr, seg_len, ctx, train_len):
    return {
        "sr": int(sr),
        "codec": str(args.codec),
        "bitrate": None if args.bitrate is None else str(args.bitrate),
        "seg_len_sec": int(SEG_LEN_SEC),
        "ctx_ratio": float(CTX_RATIO),
        "seg_len": int(seg_len),
        "ctx": int(ctx),
        "train_len": int(train_len),
    }


def save_resume_state(model, opt, epoch, batch_idx, args, sr, seg_len, ctx, train_len):
    state = {
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),   # last batch done
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "cfg": _make_resume_cfg(args, sr, seg_len, ctx, train_len),
        "saved_at": time.time(),
    }

    tmp_path = str(RESUME_STATE_PATH) + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, RESUME_STATE_PATH)

    meta = {
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "saved_at": time.time(),
        "cfg": _make_resume_cfg(args, sr, seg_len, ctx, train_len),
    }
    with open(RESUME_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_resume_state(expected_cfg):
    if not RESUME_STATE_PATH.exists():
        return None

    state = torch.load(RESUME_STATE_PATH, map_location="cpu", weights_only=False)

    cfg = state.get("cfg", {})
    for k, v in expected_cfg.items():
        if cfg.get(k) != v:
            print(f"[resume] resume found but different config on '{k}', it will be ignored.")
            return None

    return state


def restore_resume_state(state, model, opt):
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])

    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])

    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def start_keyboard_listener():
    if not sys.stdin.isatty():
        return None

    def _worker():
        try:
            if os.name == "nt":
                import msvcrt

                while not STOP_EVENT.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch.lower() == "p":
                            if PAUSE_EVENT.is_set():
                                PAUSE_EVENT.clear()
                                print("\n[resume] riparto.", flush=True)
                            else:
                                PAUSE_EVENT.set()
                                print("\n[pause] Paused. Press 'p' to resume.", flush=True)
                    time.sleep(0.05)
            else:
                import select
                import termios
                import tty

                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                try:
                    while not STOP_EVENT.is_set():
                        r, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if r:
                            ch = sys.stdin.read(1)
                            if ch.lower() == "p":
                                if PAUSE_EVENT.is_set():
                                    PAUSE_EVENT.clear()
                                    print("\n[resume] resumed.", flush=True)
                                else:
                                    PAUSE_EVENT.set()
                                    print("\n[pause] Pauseda. Press 'p' to resume.", flush=True)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            # if terminal is wonky, ignore.
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t

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

    key = str(cache_path)
    if key in AUDIO_MEM_CACHE:
        return AUDIO_MEM_CACHE[key], target_sr

    if cache_path.exists():
        audio = np.load(cache_path)
        AUDIO_MEM_CACHE[key] = audio
        return audio, target_sr

    audio = decode_audio_ffmpeg(path, target_sr)
    np.save(cache_path, audio)
    AUDIO_MEM_CACHE[key] = audio
    return audio, target_sr


def prebuild_audio_cache(paths, target_sr):
    for p in paths:
        load_audio_cached(p, target_sr)


# =========================================================
# FLUX CACHE
# =========================================================
def flux_cache_key(path, target_sr, tag="flux_v1"):
    p = Path(path).resolve()
    st = p.stat()
    seed = f"{p}|{st.st_size}|{st.st_mtime_ns}|{target_sr}|{tag}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def get_flux_cache_path(clean_path, target_sr):
    p = Path(clean_path)
    key = flux_cache_key(clean_path, target_sr)
    return Path(FLUX_CACHE_DIR) / f"{p.stem}__flux__{target_sr}__{key}.npy"


def compute_flux_from_audio(audio):
    # audio: (2, time)
    x = torch.from_numpy(np.array(audio[0], dtype=np.float32, copy=True))
    x_diff = torch.from_numpy(np.array(audio[0] - audio[1], dtype=np.float32, copy=True))

    n_fft = 512
    hop_length = 256
    window = torch.hann_window(n_fft, device=x.device)

    with torch.no_grad():
        S = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        mag = S.abs()

        diff = torch.diff(mag, dim=-1)
        flux = torch.relu(diff).mean(dim=0)

        S_diff = torch.stft(
            x_diff,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).abs()

        stereo_energy = S_diff.mean(dim=0)[: flux.shape[0]]
        flux = flux[: stereo_energy.shape[0]]

        flux = flux * (1.0 + 0.3 * stereo_energy / (stereo_energy.mean() + 1e-8))
        flux = flux + 1e-6
        flux = flux / flux.sum()

    return flux.cpu().numpy().astype(np.float32)


def load_flux_cached(clean_path, target_sr):
    cache_path = get_flux_cache_path(clean_path, target_sr)

    key = str(cache_path)
    if key in FLUX_MEM_CACHE:
        return FLUX_MEM_CACHE[key]

    if cache_path.exists():
        flux = np.load(cache_path)
        FLUX_MEM_CACHE[key] = flux
        return flux

    audio, _ = load_audio_cached(clean_path, target_sr)
    flux = compute_flux_from_audio(audio)
    np.save(cache_path, flux)
    FLUX_MEM_CACHE[key] = flux
    return flux


def prebuild_flux_cache(clean_paths, target_sr):
    for p in clean_paths:
        load_flux_cached(p, target_sr)


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


def to_torch(x, device):
    return torch.from_numpy(x).float().to(device)


# =========================================================
# FFmpeg ENCODER
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


def consistency_loss(pred, target):
    Lp, Rp, Mp, Sp = pred.unbind(1)
    Lt, Rt, Mt, St = target.unbind(1)

    return (
        F.l1_loss(Mp + Sp, Lt) +
        F.l1_loss(Mp - Sp, Rt) +
        F.l1_loss(0.5 * (Lp + Rp), Mt) +
        F.l1_loss(0.5 * (Lp - Rp), St)
    ) * 0.25


def coherence_loss(pred, target):
    pred_lr = pred[:, :2]
    target_lr = target[:, :2]

    # proxy high freq temporal difference.
    pred_diff = pred_lr[:, :, 1:] - pred_lr[:, :, :-1]
    target_diff = target_lr[:, :, 1:] - target_lr[:, :, :-1]

    return F.l1_loss(pred_diff, target_diff)


# =========================================================
# DATASET
# =========================================================
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, seg_len, sr):
        self.pairs = [(str(c), str(n)) for c, n in pairs]
        self.seg_len = seg_len
        self.sr = sr

    def __len__(self):
        return len(self.pairs) * 10  # ten times all the input dataset (audio files)

    def sample_start(self, flux, total_len):
        frames = len(flux)
        hop_audio = 256

        max_start = max(0, total_len - self.seg_len)
        if max_start <= 0:
            return 0

        idx = np.random.choice(frames, p=flux)
        start = min(int(idx) * hop_audio, max_start)

        return start

    def __getitem__(self, idx):
        idx = idx % len(self.pairs)
        clean_path, noisy_path = self.pairs[idx]

        c, _ = load_audio_cached(clean_path, self.sr)
        n, _ = load_audio_cached(noisy_path, self.sr)

        L = min(c.shape[1], n.shape[1])

        if L <= self.seg_len:
            start = 0
        else:
            flux = load_flux_cached(clean_path, self.sr)
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
    ctx = int(seg_len * CTX_RATIO)
    train_len = seg_len + ctx * 2

    input_path = Path(args.input)
    pairs = []

    for wav in sorted(input_path.glob("*.wav")):
        comp_path = Path(TEMP_DIR) / (
            f"{wav.stem}_{args.codec}_{args.bitrate}_{sr}"
            + get_codec_extension(args.codec)
        )

        if not comp_path.exists():
            compress_audio(wav, comp_path, args.bitrate, sr, args.codec)

        pairs.append((wav, comp_path))

    if not pairs:
        raise RuntimeError(f"No .wav files found in {input_path}")

    clean_paths = [clean_path for clean_path, _ in pairs]
    prebuild_flux_cache(clean_paths, sr)

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
        shuffle=False,
        num_workers=2,            # dont put zero, you have prefetch now, it will give error. try 1 or 2 not 8.
        persistent_workers=True,  # avoid ri-spawn every epoch.
        prefetch_factor=2,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = StereoUNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    expected_cfg = _make_resume_cfg(args, sr, seg_len, ctx, train_len)
    resume_state = load_resume_state(expected_cfg)

    start_epoch = 0

    if resume_state is not None:
        restore_resume_state(resume_state, model, opt)
        start_epoch = int(resume_state["epoch"]) + 1
        print(f"[resume] loaded: epoch={start_epoch} (batch resume disabled)")

    kb_thread = start_keyboard_listener()

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()

            for batch_idx, (noisy, clean) in enumerate(loader):
                while PAUSE_EVENT.is_set():
                    time.sleep(0.1)

                noisy = noisy.to(DEVICE, non_blocking=True)
                clean = clean.to(DEVICE, non_blocking=True)

                pred = model(noisy)

                pred = pred[..., ctx:-ctx]
                clean = clean[..., ctx:-ctx]

                l_lr = lr_loss(pred, clean)
                l_ms = ms_loss(pred, clean)
                l_consistency = consistency_loss(pred, clean)
                l_coherence = coherence_loss(pred, clean)

                # Total losses weighting (DONT touch the ratio !)
                loss = 0.0 * l_lr + 0.0 * l_ms + 1.0 * l_consistency + 1.0 * l_coherence

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # save resumable state each epoch
            save_resume_state(model, opt, epoch, batch_idx, args, sr, seg_len, ctx, train_len)

            if args.saveonce > 0 and ((epoch % args.saveonce) == 0 or epoch == args.epochs - 1):
                ckpt = f"model_{args.codec}_{args.bitrate}_{sr}_epoch{epoch:03d}.safetensors"
                sf_torch.save_model(model, os.path.join(CHECKPOINT_DIR, ckpt))

            print(
                f"Epoch {epoch} "
                f"l_lr: {l_lr.item():.6f} "
                f"l_ms: {l_ms.item():.6f} "
                f"l_consistency: {l_consistency.item():.6f} "
                f"l_coherence: {l_coherence.item():.6f} "
                f"TOTAL: {loss.item():.6f} "
            )

        print("Training finished.")
        # cancel resume file after finishing epochs batches:
        # if RESUME_STATE_PATH.exists(): RESUME_STATE_PATH.unlink()
        # if RESUME_META_PATH.exists(): RESUME_META_PATH.unlink()

    except KeyboardInterrupt:
        print("\n[stop] Ctrl-C : saving resume... exiting...")
        # saving last batch so at restart it will continue
        save_resume_state(model, opt, epoch, batch_idx, args, sr, seg_len, ctx, train_len)

    finally:
        STOP_EVENT.set()
        if kb_thread is not None:
            kb_thread.join(timeout=0.2)


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
    seg_len = SEG_LEN_SEC * sr
    ctx = int(seg_len * CTX_RATIO)

    chunk = seg_len
    expected_in = chunk + 2 * ctx

    # stride
    step = max(1, chunk - 2 * ctx)

    padded = np.pad(audio, ((0, 0), (ctx, ctx)), mode="edge")

    out = np.zeros((2, total), dtype=np.float32)
    w = np.zeros((2, total), dtype=np.float32)

    window = np.hanning(chunk).astype(np.float32)
    eps = 1e-8

    w_lr = 0.50
    w_ms = 0.50

    with torch.no_grad():
        for i in range(0, total, step):
            x = padded[:, i:i + expected_in]

            if x.shape[1] < expected_in:
                pad = expected_in - x.shape[1]
                x = np.pad(x, ((0, 0), (0, pad)), mode="edge")

            x_t = to_torch(to_ms(x), DEVICE).unsqueeze(0)
            y = model(x_t).squeeze(0).cpu().numpy().astype(np.float32)

            L1, R1 = y[0], y[1]
            M, S = y[2], y[3]
            L2 = M + S
            R2 = M - S

            L = w_lr * L1 + w_ms * L2
            R = w_lr * R1 + w_ms * R2

            stereo = np.stack([L, R], axis=0)
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
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--sr", type=int, required=True)
    p.add_argument("--codec", default="mp3", choices=["mp3", "aac", "opus", "vorbis", "wav"])
    p.add_argument("--bitrate", default=None, choices=["64k", "96k", "128k", "160k", "192k", "256k", "320k"])
    p.add_argument("--saveonce", type=int, default=10, help="save a checkpoint every N epoch")

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
import hashlib
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch as sf_torch

# enable this free performance boost if you have an rtx Ampere/Ada/Blackwell gpu.
# torch.set_float32_matmul_precision("high")

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
SEG_LEN_SEC = 4
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
# LOSS (PSYCHOACOUSTIC + HARMONIC AWARE + ORTOGONAL)
# =========================================================
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


def stft_loss(pred_lr, target_lr):
    fft_sizes = [128, 256, 512, 1024, 2048]
    losses = []
    
    device = pred_lr.device

    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=device)
        
        freq_bins = torch.linspace(0, 1, n_fft // 2 + 1, device=device)
        freq_weights = 1.0 / (torch.sqrt(freq_bins * 10.0) + 0.1)
        freq_weights = freq_weights / freq_weights.max()
        freq_weights = freq_weights.unsqueeze(0).unsqueeze(-1)

        loss_scale = 0.0

        for ch in [0, 1]:
            p = torch.stft(
                pred_lr[:, ch, :],
                n_fft=n_fft,
                hop_length=hop,
                window=window,
                return_complex=True,
                center=True
            )
            t = torch.stft(
                target_lr[:, ch, :],
                n_fft=n_fft,
                hop_length=hop,
                window=window,
                return_complex=True,
                center=True
            )

            mag_p = torch.abs(p)
            mag_t = torch.abs(t)
            
            # 1. MAGNITUDE LOSS
            l_mag = F.l1_loss(mag_p, mag_t)
            
            # 2. LOG-MAGNITUDE LOSS
            log_p = torch.log1p(mag_p)
            log_t = torch.log1p(mag_t)
            l_log = F.l1_loss(log_p, log_t)
            
            # 3. SPECTRAL GRADIENT LOSS
            grad_p = torch.diff(mag_p, dim=1)
            grad_t = torch.diff(mag_t, dim=1)
            l_grad = F.l1_loss(grad_p, grad_t)
            
            weighted_mag = (mag_p - mag_t).abs() * freq_weights
            weighted_log = (log_p - log_t).abs() * freq_weights
            
            l_mag_w = weighted_mag.mean()
            l_log_w = weighted_log.mean()
            
            l_grad_w = l_grad

            loss_scale += (
                l_mag_w * 1.0 +
                l_log_w * 1.0 +
                l_grad_w * 0.5
            )

        losses.append(loss_scale / 2.0)

    return sum(losses) / len(losses)


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
        model.train()

        for noisy, clean in loader:
            noisy = noisy.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)

            pred = model(noisy)

            min_len = min(pred.size(-1), clean.size(-1))
            pred = pred[..., :min_len]
            clean = clean[..., :min_len]

            L_p, R_p = pred[:, 0], pred[:, 1]
            M_p, S_p = pred[:, 2], pred[:, 3]

            L_t, R_t = clean[:, 0], clean[:, 1]
            M_t, S_t = clean[:, 2], clean[:, 3]

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

            l_consistency = consistency_loss(pred)

            l_stft_lr = stft_loss(
                torch.stack([L_p, R_p], dim=1),
                torch.stack([L_t, R_t], dim=1)
            )

            l_stft_ms = stft_loss(
                torch.stack([L_rec, R_rec], dim=1),
                torch.stack([L_t, R_t], dim=1)
            )

            l_stft = (l_stft_lr + l_stft_ms) / 2.0
            loss = l_lr + l_ms + 0.50 * l_consistency + 0.20 * l_stft

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
            f"l_stft: {l_stft.item():.6f} "
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

    step = chunk // 2

    # context buffer size (0.5s default)
    ctx = sr // 2

    out = np.zeros((2, total), dtype=np.float32)
    w = np.zeros((2, total), dtype=np.float32)

    window = np.hanning(chunk).astype(np.float32)
    eps = 1e-8

    w_lr = 0.50
    w_ms = 0.50

    prev_tail = np.zeros((2, ctx), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, total, step):

            # 1. INPUT WITH CARRY BUFFER (LOOKBACK)
            x_main = audio[:, i:i + chunk]

            if x_main.shape[1] < chunk:
                pad = chunk - x_main.shape[1]
                x_main = np.pad(x_main, ((0, 0), (0, pad)))

            x = np.concatenate([prev_tail, x_main], axis=1)

            # 2. MODEL
            x_t = to_torch(to_ms(x), DEVICE).unsqueeze(0)
            y = model(x_t).squeeze(0).cpu().numpy().astype(np.float32)

            # LR
            L1, R1 = y[0], y[1]

            # MS
            M, S = y[2], y[3]
            L2 = M + S
            R2 = M - S

            L = w_lr * L1 + w_ms * L2
            R = w_lr * R1 + w_ms * R2

            stereo = np.stack([L, R], axis=0)

            # 3. REMOVE LOOKBACK PART FROM OUTPUT
            # model output length == chunk + ctx
            stereo = stereo[:, ctx:ctx + chunk]

            # 4. OVERLAP-ADD
            valid = min(chunk, total - i)
            win = window[:valid]

            out[:, i:i + valid] += stereo[:, :valid] * win
            w[:, i:i + valid] += win

            # 5. UPDATE BUFFER
            prev_tail = audio[:, i + chunk - ctx:i + chunk]

            if prev_tail.shape[1] < ctx:
                prev_tail = np.pad(prev_tail, ((0, 0), (0, ctx - prev_tail.shape[1])))

    # FINAL NORMALIZATION
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
