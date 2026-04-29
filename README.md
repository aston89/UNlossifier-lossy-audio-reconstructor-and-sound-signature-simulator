# UNlossifier
**UNlossifier is an AI powered U-Net based audio system for lossy restoration and learned sound-domain transformation.**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Status](https://img.shields.io/badge/status-active-success)

---

## 1. Overview
UNlossifier is an AI-driven audio restoration tool designed primarily to reconstruct high-quality sound from heavily compressed lossy files.
Unlike traditional denoisers or enhancers, UNlossifier focuses on reversing codec-induced degradation. It not only reduces compression artifacts (e.g. smearing, ringing, bandwidth loss) but also attempts to reconstruct missing spectral content that was discarded during encoding.
Can also be used to reproduce a specific mixing/mastering style or creatively to model a specific sound signature.

Built around a Mid/Side-aware U-Net architecture, the system operates in both time and frequency domains, enabling coherent stereo restoration and perceptual audio recovery.

The project is designed to be:
- **Effective** on extremely degraded audio (e.g. mp3 64 kbps)
- **Lightweight** in training requirements
- **Highly customizable** for specific restoration tasks

---

## 2. Key Features

- **Lossy Artifact Removal**  
  Reduces typical compression artifacts such as pre-echo, high-frequency loss and temporal smearing.

- **Sound Signature Simulator**
  Coherently imitate a specific style, sound, instrumentation or effect.

- **Spectral Reconstruction**  
  Rebuilds plausible high-frequency content removed by lossy codecs.

- **Mid/Side Processing**  
  Ensures stereo coherence and spatial consistency during restoration.

- **U-Net Architecture**  
  Deep 1D convolutional network optimized for audio reconstruction tasks.

- **Minimal Training Data**  
  Achieves meaningful results even with very small datasets.

- **Custom Training Support**  
  Easily train specialized models for specific codecs, bitrates, or signal types.

- **Codec-Aware Pipeline**  
  Training process simulates real-world compression using configurable codecs and bitrates.

---

## 3. How It Works
UNlossifier approaches lossy audio restoration as a **reverse codec problem**: given a degraded signal, the model learns to reconstruct a plausible high-quality version by inverting compression artifacts.

The pipeline combines **time-domain learning** with **spectral supervision** while explicitly modeling stereo information through Mid/Side decomposition.

### Operating Modes
UNlossifier operates in two complementary regimes:

1. Restoration Mode
Recovers degraded audio from lossy compression (MP3, AAC, Opus, etc.), removing artifacts and reconstructing missing spectral content.

2. Signature Mode (Experimental)
Learns audio domain transformations such as:
- vinyl / tape coloration
- codec-style degradation
- analog console emulation
- lo-fi / vintage textures

Both modes share the same architecture and differ only in training data structure.

### Processing Flow

Clean Audio
Lossy Compression (mp3 / aac / opus / ...)
Degraded Audio
[ Mid/Side Encoding ]
[ U-Net Reconstruction ]
[ Multi-domain Loss Optimization ]
Restored Audio (LR reconstructed from MS)

### Core Components

- **Mid/Side Representation (4 channels)**  
  The model operates on L, R, Mid, and Side simultaneously.  
  This allows it to preserve stereo image while restoring shared and differential content.

- **U-Net Architecture (1D)**  
  Captures both local details (transients) and long-range dependencies (structure and texture).

- **Multi-Domain Training**  
  Combines:
  - waveform loss (time domain)
  - STFT magnitude loss (frequency domain)
  - stereo coherence constraints

- **Overlap-Add Inference**  
  Audio is processed in chunks with windowing to ensure seamless reconstruction.

The result is a system that does not simply clean audio, but **learns how compression destroys information and approximates its reversal**.

---

## 3b. Extended Capability: Sound Signature Simulation
Beyond restoration, UNlossifier can learn *audio domain transformations* when trained on structured pairs.
When the dataset is inverted or reinterpreted (e.g. vinyl-clean vs digital-clean, analog chain emulation, codec style mapping), the model shifts from reconstruction to **audio style transfer**.

This enables:
- Vinyl / tape coloration simulation  
- Console / mixer signature emulation  
- Lo-fi / vintage texture generation  
- Codec-style transformation modeling  

In this mode, UNlossifier behaves less like a repair tool and more like a **learned audio transformation engine**, capturing statistical characteristics of a target sound domain.

---

## 4. Audio Restoration Philosophy
UNlossifier is not a traditional restoration tool.  
It is based on a simple but important premise:
> Once audio is compressed with a lossy codec, the original signal is **irreversibly altered**.

**This means true reconstruction is impossible.**

### Instead, the goal becomes:

**Plausible Reconstruction**
The model does not recover the exact original waveform, but generates a version that is:
- perceptually closer to high-quality audio
- spectrally richer
- spatially coherent

**Beyond Denoising**
Unlike standard approaches that:
- remove noise
- smooth artifacts

UNlossifier actively:
- **rebuilds missing frequency content**
- **restores perceived detail**
- **reconstructs stereo structure**

**Learning the Codec Damage**
By training on clean vs compressed pairs, the model implicitly learns:
- what information codecs discard
- how artifacts manifest across bitrates
- how to approximate the inverse transformation

**Data-Efficiency by Design**
The system is intentionally designed to:
- work with **very small datasets**
- specialize quickly on specific distortions
- generalize from minimal examples

This makes it suitable not only for general restoration, but also for:
- niche audio domains
- specific codecs/bitrates
- synthetic or controlled training scenarios

In short, UNlossifier does not try to "fix" audio.  
It tries to **coherently reimagine what was lost**.

---

## 5. Model Architecture
UNlossifier is built around a **1D U-Net architecture** specifically adapted for stereo audio reconstruction.

### Input Representation
Instead of operating only on left/right channels, the model uses a **4-channel Mid/Side representation**:
- Left (L)
- Right (R)
- Mid (M = (L + R) / 2)
- Side (S = (L - R) / 2)

This hybrid representation allows the network to:
- preserve stereo coherence
- separate shared vs differential content
- reconstruct spatial information more effectively 

### Network Design
The model follows a standard encoder–decoder U-Net structure:

- **Encoder**
  - Progressive downsampling via strided convolutions
  - Increasing channel depth
  - Captures global structure and compression artifacts

- **Bottleneck**
  - High-level feature processing
  - Learns compact representations of degraded audio

- **Decoder**
  - Transposed convolutions for upsampling
  - Skip connections from encoder layers
  - Restores fine temporal details

- **Output Layer**
  - Produces 4 channels (L, R, M, S)
  - Final stereo is a continuously graduated ensembled transformation from LS and MS contemporaneously, minimizing further the loss error
  
### Design Choices

- **1D Convolutions**  
  Optimized for raw waveform processing and temporal precision.

- **Group Normalization**
  Stable training with small batch sizes.

- **GELU Activation**  
  Smooth non-linearity for better gradient flow.

- **Skip Connections**  
  Preserve micro-details lost during downsampling.

### Loss Function (Multi-Domain)
Training is guided by a composite loss that balances multiple aspects of audio quality:

- **Waveform Loss (L1)**  
  Ensures time-domain alignment.

- **L/S to Mid/Side Consistency**  
  Ensures LR and MS representations agree.

- **Multi-Scale STFT Loss**  
  Operates at multiple FFT sizes to capture:
  - transient detail (small FFT)
  - harmonic structure (large FFT)

This combination allows the model to balance **mathematical accuracy** and **perceptual quality**.

---

## 6. Training
UNlossifier is trained using **paired audio data**:
- **Clean audio** (reference)
- **Compressed audio** (degraded via codec)

The goal is to learn a mapping:
Lossy Audio -> Reconstructed High-Quality Audio

### Data Preparation
Training pairs are generated automatically:

1. Start from clean `.wav` files
2. Apply lossy compression using:
   - MP3
   - AAC
   - Opus
   - Vorbis
3. Control degradation via bitrate (e.g. 64k, 96k, 128k)

This simulates real-world codec damage in a controlled way.

### Segment-Based Training
Instead of full tracks, audio is processed in short segments:
- Typical length: ~4 seconds
- Random sampling per iteration
- Improves generalization and efficiency

### Training Strategy
- Small batch sizes (GPU-friendly)
- Adam optimizer
- Multi-loss optimization (time + frequency + stereo)

The model quickly learns:
- compression artifacts
- spectral gaps
- stereo inconsistencies

### Why It Works with Few Samples
Unlike generic audio models, UNlossifier learns a **structured degradation process**.

Lossy compression:
- follows predictable patterns
- removes specific frequency bands
- introduces characteristic artifacts

This makes the learning problem:
- **highly constrained**
- **data-efficient**

Even a handful of audio files can be sufficient to:
- learn artifact signatures
- approximate reconstruction behavior

---

## 7. Custom Training (Core Feature)

One of UNlossifier’s defining strengths is its ability to **specialize rapidly**.
Instead of requiring massive datasets, the model can be trained for:
- specific codecs (e.g. MP3 @ 64 kbps)
- specific content types (speech, music, FX)
- specific degradation patterns

### Minimal Dataset Training
UNlossifier is designed to work with:
- as few as **5–10 audio samples**
- short training cycles
- fast iteration

This enables:
- rapid experimentation
- targeted restoration models

### Specialized Models
You can train models that are:
- **Codec-specific**  
  e.g. MP3 artifacts only
- **Bitrate-specific**  
  e.g. aggressive 64 kbps recovery
- **Domain-specific**  
  e.g. voice-only or instrument-focused

### Synthetic Training (Unique Capability)
A key differentiator is support for **synthetic datasets**, such as:
- white noise
- pink noise
- sine waves
- complex waveforms

These signals allow the model to:
- learn frequency response loss explicitly
- understand codec behavior in isolation
- build reconstruction priors

This approach is rarely used in traditional tools and opens the door to:
- highly controlled experiments
- deeper codec inversion learning

### General vs Specialized Models

- **General models**
  - trained on diverse audio
  - robust across scenarios

- **Specialized models**
  - trained on narrow domains
  - higher quality in specific use cases

**UNlossifier is designed to support both approaches seamlessly.**
In essence, training is not just a requirement, it is the **core interface** through which the system adapts to the problem.

---

## 8. Usage
UNlossifier provides a simple CLI interface for both training and inference.

### Training
To train a model, provide a folder containing clean `.wav` files:
> python main.py --input ./data --sr 44100 --epochs 10 --batch 2 --codec mp3 --bitrate 96k

#### Arguments
- --input      Path to folder containing clean WAV files ("./folder")
- --sr         Sample rate (e.g. 44100) (also, pairs in ram will have this sample rate)
- --epochs     Number of training epochs 
- --batch      Batch size (affects ram/vram) 
- --codec      Compression codec (mp3, aac, opus, vorbis) - (use "wav" for creative signature style model training)
- --bitrate    Target bitrate (e.g. 64k, 96k, 128k) - (skip this for creative signature style model training)

During training:
- Clean audio is automatically compressed using ffmpeg
- Pairs (clean vs degraded) are stored in ram on the fly (degraded are cached in "./temp_audio")
- Model checkpoints are saved after each epoch

### Inference
To restore an audio file using a trained model:
> python main.py --input input.wav --output restored.wav --model model.safetensors --sr 44100

#### Arguments
--input      Input audio file  
--output     Output restored file  
--model      Path to trained model (.safetensors)  
--sr         Sample rate  

### Processing Details
- Audio is processed in overlapping chunks
- Windowing is applied to avoid artifacts
- Output is reconstructed using overlap-add (hann)
- Stereo is restored from Mid/Side representation

### Example Workflow
1. Collect clean audio samples  
2. Train model with desired codec/bitrate  
3. Run inference on degraded audio  
4. Evaluate and iterate 

UNlossifier is designed to be simple to use, while remaining flexible enough for advanced workflows.

---

## 9. Examples / Demos
UNlossifier is particularly effective on heavily degraded audio where traditional tools struggle but can be used also to simulate a specific style or instrumentation.

### Use Cases
- **Low bitrate audio (64–96 kbps)**
  Restoration of heavily compressed music or recordings.
- **Multiple compression passes**  
  Recovery from files that have been re-encoded multiple times.
- **Streaming / legacy audio**
  Enhancement of low-quality sources from web or archives.
- **Lo-Fi / Vinyl / Bitcrush / Tape Saturation / analog mixer warmness**
  Creative training from proper "dirty" wav sources can be used to simulate a specific instrumentation or mix/master behaviour or even specific music genre like old jazz recordings.

### Before / After (Conceptual)
**Input (lossy):**
- muffled high frequencies  
- smeared transients  
- stereo collapse  

**Output (UNlossifier):**
- restored brightness  
- improved transient clarity  
- reconstructed stereo field

### Demo Suggestions
To showcase the model effectively:
- Use identical audio segments before/after
- Focus on difficult material (dense mixes, cymbals, vocals)
- Include extreme cases (e.g. 64 kbps MP3)

---

## 10. Limitations
UNlossifier is powerful, but it operates under fundamental constraints.

### Not True Reconstruction
Lossy compression permanently removes information.  
The model generates a **plausible reconstruction** not the exact original signal.

### Dependency on Training Data
Performance depends on:
- codec type
- bitrate
- similarity between training and inference data

A poorly matched model may:
- underperform
- introduce artifacts
- over-smooth the signal

### Hallucinated Content
Reconstructed frequencies are:
- inferred, not recovered
- perceptually convincing, but not ground truth

### Extreme Degradation
Very low bitrates or heavily damaged audio may:
- limit reconstruction quality
- reduce stereo accuracy

---

## 11. Potential
- **Real-time inference**
- **VST / DAW plugin integration**
- **GUI interface**
- **Hybrid time-frequency architectures**
- **Perceptual loss improvements (psychoacoustic models)**

---

## 12. Installation

### Requirements
- Python 3.9+
- PyTorch
- ffmpeg

### Setup
Install dependencies:
pip install -r requirements.txt
- torch
- numpy
- librosa
- soundfile
- safetensors

### FFmpeg
Ensure ffmpeg is installed and accessible:
ffmpeg -version
(on windows install ffmpeg and ensure it's present on "path" enviroment variables)

---

### Notes
- GPU is strongly recommended for training
- CPU inference is possible but slower
- Disk space is required for temporary compressed files
- model_mp3_96k_32000_epoch393.safetensors (Epoch 393 l_lr: 0.008856 l_ms: 0.005933 l_stft: 0.677289 l_consistency: 0.001343 TOTAL: 0.150919)
  its an example of model trained on only 6 pairs of different music style and genre, it's specifically usefull to restore mp3 compressed at 96kbps 32khz.
