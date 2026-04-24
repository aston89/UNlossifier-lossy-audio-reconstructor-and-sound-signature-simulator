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
import torch.optim as optim
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

    return audio, sr

def save_audio(path, audio, sr):
    audio = audio.T if audio.shape[0] == 2 else audio
    sf.write(path, audio, sr)

def preload_audio_pair(clean_path, noisy_path, sr):
    key = (str(clean_path), str(noisy_path))

    if key in AUDIO_CACHE:
        return AUDIO_CACHE[key]

    c, _ = load_audio(clean_path, sr)
    n, _ = load_audio(noisy_path, sr)

    c = torch.from_numpy(c).float()
    n = torch.from_numpy(n).float()

    AUDIO_CACHE[key] = (c, n)

    return c, n

# =========================================================
# M/S ENCODING
# =========================================================
def to_ms(x):
    L = x[0]
    R = x[1]
    M = 0.5 * (L + R)
    S = 0.5 * (L - R)
    return np.stack([L, R, M, S], axis=0)

def from_ms(x):
    L = x[2] + x[3]
    R = x[2] - x[3]
    return np.stack([L, R], axis=0)

# =========================================================
# FFmpeg
# =========================================================
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

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
# MODEL (4-channel MS)
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
    def __init__(self, base=64):
        super().__init__()

        self.enc1 = ConvBlock(4, base)

        self.down1 = nn.Conv1d(base, base, 4, stride=2, padding=1)

        self.enc2 = ConvBlock(base, base * 2)
        self.down2 = nn.Conv1d(base * 2, base * 2, 4, stride=2, padding=1)

        self.enc3 = ConvBlock(base * 2, base * 4)
        self.down3 = nn.Conv1d(base * 4, base * 4, 4, stride=2, padding=1)

        self.mid = ConvBlock(base * 4, base * 4)

        self.up3 = nn.ConvTranspose1d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec3 = ConvBlock(base * 4, base * 2)

        self.up2 = nn.ConvTranspose1d(base * 2, base, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(base * 2, base)

        self.out = nn.Conv1d(base, 4, 7, padding=3)

    def forward(self, x):
        e1 = self.enc1(x)
        x = self.down1(e1)

        e2 = self.enc2(x)
        x = self.down2(e2)

        e3 = self.enc3(x)
        x = self.down3(e3)

        x = self.mid(x)

        x = self.up3(x)
        x = match_size(x, e2)
        x = torch.cat([x, e2], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = match_size(x, e1)
        x = torch.cat([x, e1], dim=1)
        x = self.dec2(x)

        return self.out(x)

# =========================================================
# SIZE FIX
# =========================================================
def match_size(x, ref):
    diff = ref.size(-1) - x.size(-1)
    return F.pad(x, (0, diff)) if diff > 0 else x[..., :ref.size(-1)]

# =========================================================
# LOSS
# =========================================================
def multi_stft_loss(pred, target):

    fft_sizes = [128, 1024, 2048]
    losses = []

    px_L = pred[:, 0, :]
    px_R = pred[:, 1, :]
    tx_L = target[:, 0, :]
    tx_R = target[:, 1, :]

    for n_fft in fft_sizes:

        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device)

        loss_scale = 0.0

        for px, tx in [(px_L, tx_L), (px_R, tx_R)]:

            p = torch.stft(px, n_fft, hop, window=window, return_complex=True)
            t = torch.stft(tx, n_fft, hop, window=window, return_complex=True)

            mag_p = torch.abs(p)
            mag_t = torch.abs(t)

            loss_scale += (
                F.l1_loss(mag_p, mag_t) +
                F.l1_loss(torch.log(mag_p + 1e-7), torch.log(mag_t + 1e-7))
            )

        losses.append(loss_scale / 2)

    return losses  # [stft_128, stft_1024, stft_2048]

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

        clean, noisy = self.pairs[random.randint(0, len(self.pairs)-1)]

        c, n = preload_audio_pair(clean, noisy, self.sr)

        L = min(c.shape[1], n.shape[1])
        start = random.randint(0, max(1, L - self.seg_len))

        c = to_ms(c[:, start:start+self.seg_len])
        n = to_ms(n[:, start:start+self.seg_len])

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
    opt = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):

        for noisy, clean in loader:

            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            pred = model(noisy)

            min_len = min(pred.size(-1), clean.size(-1))
            pred = pred[..., :min_len]
            clean = clean[..., :min_len]

            # MS SPLIT LOSS (foundamental)
            L_p, R_p, M_p, S_p = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            L_t, R_t, M_t, S_t = clean[:, 0], clean[:, 1], clean[:, 2], clean[:, 3]

            # waveform each domain
            l_lr = F.l1_loss(torch.stack([L_p, R_p], dim=1),
                             torch.stack([L_t, R_t], dim=1))

            l_ms = F.l1_loss(torch.stack([M_p, S_p], dim=1),
                             torch.stack([M_t, S_t], dim=1))

            # ms consistency
            M_p, S_p = pred[:, 2], pred[:, 3]
            L_from_ms = M_p + S_p
            R_from_ms = M_p - S_p

            ms_consistency = (
                F.l1_loss(L_from_ms, pred[:, 0]) +
                F.l1_loss(R_from_ms, pred[:, 1])
            )

            # spectral coherency (all channels)
            stft_128, stft_1024, stft_2048 = multi_stft_loss(pred, clean)

            l_stft = (
                0.3 * stft_128 +
                0.5 * stft_1024 +
                0.2 * stft_2048
            )

            # FINAL LOSS
            loss = (
                l_lr +
                l_ms +
                l_stft +
                0.10 * ms_consistency
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

        ckpt = f"model_{args.codec}_{args.bitrate}_{sr}_epoch{epoch:03d}.safetensors"
        sf_torch.save_model(model, os.path.join(CHECKPOINT_DIR, ckpt))

        print(f"""Epoch {epoch} l_lr: {l_lr.item():.6f} l_ms: {l_ms.item():.6f} stft_128: {stft_128.item():.6f} stft_1024: {stft_1024.item():.6f} stft_2048: {stft_2048.item():.6f} ms_consistency: {ms_consistency.item():.6f} TOTAL: {loss.item():.6f}""")

# =========================================================
# INFERENCE
# =========================================================
def inference(args):

    model = StereoUNet().to(DEVICE)
    sf_torch.load_model(model, str(args.model))
    model.eval()

    sr = args.sr
    audio, _ = load_audio(args.input, sr)

    if audio.ndim == 1:
        audio = np.stack([audio, audio])

    total = audio.shape[1]
    chunk = SEG_LEN_SEC * sr
    step = chunk - int(chunk * 0.1)

    out = np.zeros((4, total), dtype=np.float32)
    w = np.zeros((4, total), dtype=np.float32)

    window = np.hanning(chunk)

    with torch.no_grad():
        for i in range(0, total, step):

            x = audio[:, i:i+chunk]

            if x.shape[1] < chunk:
                pad = chunk - x.shape[1]
                x = np.pad(x, ((0,0),(0,pad)))

            x = torch.tensor(to_ms(x)).float().unsqueeze(0).to(DEVICE)

            y = model(x).squeeze(0).cpu().numpy()
            y = y[:, :min(chunk, total-i)]

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
