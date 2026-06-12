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
Learns audio domain transformations such as vinyl or tape coloration, codec-style degradation, analog console emulation, lo-fi / vintage textures.
  
Both modes share the same architecture and differ only in training data structure.

### Processing Flow

| Stage | Process |
|------|--------|
| 1 | Clean Audio |
| 2 | Lossy Compression (mp3 / aac / opus / ...) |
| 3 | Degraded Audio |
| 4 | Mid/Side Encoding |
| 5 | U-Net Reconstruction |
| 6 | Multi-domain Loss Optimization |
| 7 | Restored Audio (LR reconstructed from MS) |

### Core Components

- **Hybrid Left/Right and Mid/Side Representation (4 channels)**  
  The model operates on 2+2 channels simultaneously.  
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

## 3b. Experimental Sound Signature Simulation: Limitations 
Beyond restoration, UNlossifier can be repurposed as a **learned audio style transformation engine** by supplying custom source/target pairs instead of clean/compressed pairs even if [it's a bit tricky](https://github.com/aston89/UNlossifier-lossy-audio-reconstructor-and-sound-signature-simulator/issues/1).
For example, users can intentionally provide custom source/target pairs to teach the model a desired audio transformation by placing processed audio in the source dataset and unprocessed audio in the target dataset, allowing the model to learn a custom domain transformation.
However, this approach is considered experimental. Neural networks learn consistent relationships between inputs and targets.
Transformations that follow stable patterns (EQ curves, tonal coloration, frequency response changes, console character, tape saturation, etc.) can often be learned successfully.
Effects containing **strong random or non-deterministic elements (vinyl crackle, random clicks, stochastic noise bursts, unpredictable modulation, etc.) may be harder or impossible to reproduce faithfully**, instead, the more deterministic and consistent the transformation is, the more reliably it can be learned, as a result, Signature Mode should be viewed as a creative experimentation tool rather than a guaranteed audio effect cloning system.



This enables:
- Vinyl / tape coloration simulation  
- Console / mixer signature emulation  
- Lo-fi / vintage texture generation  
- Codec-style transformation modeling  

In this mode, UNlossifier behaves like a **learned audio style transformation engine** by capturing statistical characteristics of a target sound domain.

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
  - texture timbre (medium FFT)
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
> UNlossifier.py --input ./yourfolder --sr 44100 --epochs 10 --batch 2 --codec mp3 --bitrate 96k

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
> UNlossifier.py --input input.wav --output restored.wav --model yourtrainedmodel.safetensors --sr 44100*

*be sure that your file output sample rate matches the trained model sample rate.

#### Arguments
--input      Input audio file  
--output     Output restored file  
--model      Path to trained model (.safetensors)  
--sr         Sample rate  

### Processing Details
- Audio is processed in overlapping chunks
- Windowing is applied to avoid artifacts
- Output is reconstructed using overlap-add (hann)
- Stereo is restored from L/S+M/S representation

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
  Recovery from files that have been re-encoded multiple times (in this case, put your degraded files pre-made in the temp_audio folder, compression will be skipped)
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

---

### Safetensor weights

- **model_mp3_96k_32000_epoch393.safetensors** (Epoch 393 l_lr: 0.008856 l_ms: 0.005933 l_stft: 0.677289 l_consistency: 0.001343 TOTAL: 0.150919).
  Trained with 6 pairs of different music style and genre using the V1 version, it's specifically usefull to restore mp3 compressed at 96kbps 32khz.
  This model is optimal for mp3 encoded with libmp3lame (FFmpeg LAME encoder) at CBR.
  
- **model_mp3_128k_44100_epoch397.safetensors** (Epoch 397 l_lr: 0.003997 l_ms: 0.002769 l_stft: 0.015105 l_consistency: 0.004153 TOTAL: 0.011863).
  Trained with 6 pairs of different music style and genre using v2 version, it's specifically usefull to restore mp3 compressed at 128kbps 44.1khz.
  This model is optimal for mp3 encoded with libmp3lame (FFmpeg LAME encoder) at CBR.
  
- **model_aac_128k_44100_epoch998.safetensors** (Epoch 998 l_lr: 0.004069 l_ms: 0.002902 l_stft: 0.018460 l_consistency: 0.000108 TOTAL: 0.010717).
  Trained with 6 pairs of different music style and genre using v2 version, it's specifically usefull to restore aac compressed at 128kbps 44.1khz.
  This model is "casually" optimal for youtube AAC encoded files, very similar to ffmpeg default aac encoder.
  This model may exhibit peaks overshoots or instability when applied to AAC files generated through different encoding pipelines or third-party online converters, therefore if you are uncertain about the restored.wav, do a check for overshoot peaks using [DeltaWave](https://deltaw.org/), if peaks are present, model/coded-pipeline mismatch its the problem.

- **model_mp3_64k_44100_epoch997.safetensors** (Epoch 997 l_lr: 0.005969 l_ms: 0.004224 l_stft: 0.021610 l_consistency: 0.000126 TOTAL: 0.014579)-
  Trained with 6 pairs of different music style and genre using latest version, it's specifically usefull to restore mp3 compressed at 64kbps 44.1khz.
  This model is optimal for mp3 encoded with libmp3lame (FFmpeg LAME encoder) at CBR.


tip: A model trained at 32 kHz sampling rate can be used to infer audio at 44.1 kHz, but it will not be able to reconstruct or meaningfully restore content above its training bandwidth limit (≈16 kHz effective Nyquist region). Higher-frequency components will remain absent or be implicitly hallucinated rather than recovered.

tip: A model trained on 64 kbps compressed audio can be applied to higher-quality sources (e.g. 320 kbps “near-lossless” MP3). In this case, the model does not simply restore missing information; it may reinterpret existing spectral content, subtly altering timbre and texture. In some cases, what is actually valid signal content may be treated as compression artifacts and reshaped accordingly.

---

### Note about codec restoration difficulty !

Not all lossy codecs are equally recoverable, **older codecs such as MP3 tend to introduce relatively predictable and stationary artifacts**, making them easier for neural models to learn and compensate.
Modern codecs such as AAC, Vorbis and especially Opus rely on increasingly sophisticated psychoacoustic models, adaptive transforms, temporal masking and dynamic bitrate allocation. Their artifacts are often highly non-stationary and context-dependent.
As a consequence, restoration quality does not scale linearly with training time. **Even at very high epoch counts, localized artifacts (clicks, pops, transient instabilities) may remain** because the original codec decisions are not directly observable from the decoded waveform alone.
In practice: MP3 (easiest) / AAC (difficult) / Vorbis (very difficult) / Opus (extremely difficult).

### Note about "codec lasagna" !

Not all files are degraded equally, some audio sources are clean single-pass encodes (e.g. WAV to AAC once), others come from a far more chaotic ecosystem: repeated uploads, platform re-encodes, format conversions, streaming optimizations, and unknown intermediate processing steps.
We refer to this condition informally as **codec lasagna**, a stack of unknown lossy transformations applied over time.
In practice, a “multiple times converted” file may behave less like a single compression artifact and more like an accumulation of heterogeneous distortions, including multiple psychoacoustic re-encodings, transient misalignment across generations, resampling and normalization artifacts, unknown limiter/encoder interactions.
**What should you expect from restoration?**
Depending on the depth of the “lasagna”, results may range from:
* **Clean single-pass encodes** - “roasted potatoes”: structured, recoverable, predictable artifacts.
* **Moderately processed sources** - “seasoned stew”: recoverable but with residual instability.
* **Deep multiconversion chains** - “lentil purée”: dense, chaotic, partially irreducible structure.
In the latter case, residual artifacts (clicks, micro-glitches, spectral smearing) may not reflect model failure but rather the absence of a consistent underlying encoding process to invert.

### Note about Vorbis and Opus codecs !

When working with Vorbis and Opus, it’s important to drop the intuition that bitrate is a stable or even fully meaningful parameter. Both codecs behave more like adaptive systems than fixed-rate encoders, and what you see in tools (MediaInfo) or filenames (JDownloader2) is usually a **derived or averaged value** rather than something strictly enforced during encoding.
Opus, in particular, operates on a frame-by-frame basis, continuously reallocating its internal bitrate budget depending on the complexity of the signal at any given moment. A dense transient might temporarily “consume” far more bits, while a sustained or simple segment will drop dramatically, all while maintaining a long-term average that loosely matches the requested target. 
**For Opus, the sampling rate is effectively fixed at 48 kHz by design. The codec internally operates on a 48 kHz time base regardless of the input material, and any incoming audio is implicitly resampled to match this domain before encoding**. As a result, specifying a different sampling rate at the CLI level does not change the fundamental working rate of the codec; it only affects the preprocessing and postprocessing stages of the pipeline. In practice, **this means that using --sr 48000 is not just recommended but functionally aligned with how Opus is intended to operate**, while other values introduce unnecessary resampling without changing codec behavior. **TL:DR If you use Opus, always set --sr 48000 for both training and inference. Only as a final post-processing step, use FFmpeg to resample to 44100 if needed**. This is the cleanest and most consistent pipeline.
**Vorbis, on the other hand, is more flexible. It does not enforce a fixed internal sampling rate and can operate across a range of common rates such as 44.1 kHz or 48 kHz**. In practice, however, most modern streaming and production pipelines tend to standardize Vorbis around 48 kHz as well, largely for ecosystem consistency rather than codec limitation. Unlike Opus, Vorbis does not require resampling to a single canonical rate, meaning the choice of --sr has a more direct impact on the actual processing domain of the model and the dataset consistency.
**In conclusion : In both Vorbis and Opus, when using --bitrate 128k, you are only setting a target average bitrate. The encoder will dynamically allocate bits per frame based on audio complexity, so the actual instantaneous bitrate will vary, even if the long-term average stays around 128k.**

---

### Update v2 (08/06/2026): Model refinement & training/inference redesign
* **Reworked STFT loss (major upgrade):** replaced the previous magnitude/log-loss stack with a psychoacoustic-aware formulation, adding frequency-weighted emphasis (higher sensitivity to low frequencies), spectral gradient loss, and `log1p` stabilization for improved dynamic range handling.
* **Richer multi-resolution analysis:** expanded STFT scales and made the loss more perceptually balanced across resolutions (from ultra-low to high frequency bands).
* **Dual-path STFT supervision:** added a second STFT loss branch computed on both LR output and MS-reconstructed LR signal, improving consistency between representations.
* **Training objective redesign:** loss now jointly enforces LR fidelity, MS structure alignment, STFT perceptual quality, and reconstruction consistency in a more tightly coupled way.
* **Inference blending simplified:** removed adaptive energy-based fusion; replaced with fixed LR/MS mixing (balanced ensemble approach for stability and predictability).
* **Overlap strategy improved:** inference switched from ~10% overlap to 50% overlap, significantly reducing boundary artifacts with Hann window overlap-add.
* **Output stability tightened:** final waveform clipping range adjusted from wider dynamic range to a stricter [-1, 1] normalization for safer audio export.
* **Cleaner separation of concerns:** LR and MS branches are now treated more symmetrically during both training and inference, reducing representational drift.

### Update v3 (11/06/2026): Improvements & bug fixes
* Replaced `librosa.load` pipeline with a **FFmpeg-based raw float32 decoder**, eliminating librosa as primary audio loader during training/inference.
* Introduced a **disk-based NumPy cache system (`.npy`)** for decoded audio instead of pure in-RAM caching, enabling persistence across runs.
* Added **deterministic cache keys using SHA1 + file metadata (size, mtime, sr, codec, bitrate, tag)** to avoid stale or mismatched cached audio.
* Split caching into **separate CACHE_DIR (numpy audio) and TEMP_DIR (ffmpeg compressed audio)**, improving pipeline clarity and isolation.
* Added `prebuild_audio_cache()` step to **pre-decode all training audio upfront**, reducing runtime bottlenecks during dataset iteration.
* `load_audio_cached()` now supports **automatic cache hit/miss logic with mmap loading (`np.load(..., mmap_mode="r")`)** for lower memory pressure.
* Removed reliance on the previous global **in-memory `AUDIO_CACHE` as the main acceleration mechanism**, shifting optimization toward disk cache strategy.
* Updated dataset handling so `pairs` are **normalized to string paths early**, avoiding Path object overhead in worker processes.
* Improved FFmpeg commands with **quiet mode (`-hide_banner -loglevel error -nostdin`)** for cleaner and faster subprocess execution.
* Audio decoding now ensures **strict stereo shape validation and correction of odd-length buffers**, making pipeline more robust to corrupted/edge outputs.
* Cache keying in dataset flux logic updated to include a **dedicated `"flux"` tag namespace**, preventing collision with other cached representations.
* Minor optimization in dataset: **flux computation explicitly casts to float32 early**, reducing implicit NumPy dtype churn.
* Training pipeline now explicitly **builds full audio cache before DataLoader creation**, improving first-epoch stability and throughput.
* Overall architecture shift from **runtime decoding-heavy pipeline → preprocessing + cached dataset-driven pipeline**, improving training speed consistency at the cost of disk usage.
* Added a cuda performance boost switch ""torch.set_float32_matmul_precision("high")"", if you have an rtx gpu like Ampere, Ada or Blackwell, remove the comment.
* restored the audio output file at 32bit float instead of the default pcm16 to avoid dithering, noise shaping and eventual peak clipping.

### Update v3b (12/06/2026): consistency constraint update
* Replaced the old consistency loss  with a stronger `consistency_loss()` that checks both directions: LR to MS // MS to LR
* Renamed `stft_lr_loss()` to `stft_loss()` just for cleaner naming.
* Training now uses the new orthogonal consistency constraint as the main structural regularizer (harder better faster stronger - *daft punk cit.*).
* avg 10x times faster in training, total loss around 0.01 at only 100 epochs compared to v3 wich required like 1000 epochs to achieve similar result.
