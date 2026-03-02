# DeepFilterNet MLX Implementation Roadmap

> **Goal**: Achieve full feature parity between `df_mlx/` (MLX/Apple Silicon) and `df/` (PyTorch)

## Current Status: 100% Parity ✅

### What's Implemented ✅
- DFNet4 + DFNet4Lite model architecture
- DFNet1/2/3 architectures (GroupedGRU, SqueezedGRU)
- DFNetMF multi-frame architecture
- Mamba/SqueezedMamba backbone
- HybridEncoder, LinearEncoder variants
- Core modules (Conv2dNormAct, GroupedLinear, DfOp, ERB filterbank)
- STFT/iSTFT signal processing
- Basic training loop
- GAN training (discriminators, adversarial losses)
- Multi-resolution spectral loss, SI-SDR loss, GAN losses
- Full evaluation pipeline (SI-SDR, STOI, PESQ, DNSMOS, composite metrics)
- LR schedulers (cosine, warmup)
- Multi-frame processing
- MLX quantization support
- Inference/enhancement utilities
- Visualization utilities
- Whisper adapter (ASR loss, transcription)
- Hardware detection/optimization
- Sharded NPZ data loading with dynamic mixing
- Full checkpointing with patience tracking

### What's Missing ❌
- *(None - Full parity achieved!)*

---

## Implementation Phases

### Phase 1: Training Foundations (Priority: HIGH)
*Estimated: 1-2 weeks*

| Task | File | Complexity | Status |
|------|------|------------|--------|
| 1.1 LR Schedulers (cosine, warmup) | `lr.py` | S | ✅ |
| 1.2 DF Alpha Loss | `loss.py` | S | ✅ |
| 1.3 Early Stopping | `train.py` | S | ✅ |
| 1.4 Checkpoint patience tracking | `checkpoint.py` | S | ✅ |
| 1.5 Training config validation | `config.py` | S | ✅ |

**Deliverable**: Training loop with proper LR scheduling and loss functions

---

### Phase 2: GAN Training (Priority: HIGH)
*Estimated: 3-5 weeks*

| Task | File | Complexity | Status |
|------|------|------------|--------|
| 2.1 PeriodDiscriminator | `discriminator.py` | M | ✅ |
| 2.2 ScaleDiscriminator | `discriminator.py` | M | ✅ |
| 2.3 MultiPeriodDiscriminator | `discriminator.py` | M | ✅ |
| 2.4 MultiScaleDiscriminator | `discriminator.py` | M | ✅ |
| 2.5 GAN losses (G/D/feature matching) | `loss.py` | M | ✅ |
| 2.6 Alternating GAN training loop | `train_dynamic.py` | L | ✅ |

**Deliverable**: Full GAN training capability matching PyTorch

---

### Phase 3: Evaluation Pipeline (Priority: HIGH)
*Estimated: 2-3 weeks*

| Task | File | Complexity | Status |
|------|------|------------|--------|
| 3.1 SI-SDR metric | `evaluation.py` | S | ✅ |
| 3.2 STOI metric | `stoi.py` | M | ✅ |
| 3.3 PESQ wrapper | `evaluation.py` | S | ✅ |
| 3.4 Composite metrics (CSIG/CBAK/COVL) | `evaluation.py` | M | ✅ |
| 3.5 DNSMOS integration | `dnsmos_proxy.py` | M | ✅ |
| 3.6 Evaluation loop with parallel workers | `evaluation.py` | M | ✅ |

**Deliverable**: Complete evaluation pipeline matching PyTorch

---

### Phase 4: Inference & Enhancement (Priority: MEDIUM)
*Estimated: 2 weeks*

| Task | File | Complexity | Status |
|------|------|------------|--------|
| 4.1 Model loading utilities | `enhance.py` | S | ✅ |
| 4.2 Single-file enhancement | `enhance.py` | M | ✅ |
| 4.3 Batch enhancement | `enhance.py` | M | ✅ |
| 4.4 Streaming inference | `enhance.py` | L | ✅ |
| 4.5 Model download/caching | `enhance.py` | M | ✅ |

**Deliverable**: CLI and programmatic enhancement interface

---

### Phase 5: Legacy Model Support (Priority: MEDIUM)
*Estimated: 3-4 weeks*

| Task | File | Complexity | Status |
|------|------|------------|--------|
| 5.1 GroupedGRU module | `modules.py` | L | ✅ |
| 5.2 SqueezedGRU module | `modules.py` | M | ✅ |
| 5.3 DFNet3 architecture | `deepfilternet3.py` | L | ✅ |
| 5.4 DFNet2 architecture | `deepfilternet2.py` | L | ✅ |
| 5.5 DFNet1 architecture | `deepfilternet.py` | M | ✅ |
| 5.6 PyTorch checkpoint conversion | `convert.py` | M | ✅ |

**Deliverable**: Support for loading/running all DFNet variants

---

### Phase 6: Advanced Features (Priority: LOW)
*Estimated: 4+ weeks*

| Task | File | Complexity | Status |
|------|------|------------|--------|
| 6.1 Multi-frame processing module | `multiframe.py` | L | ✅ |
| 6.2 DFNetMF architecture | `deepfilternetmf.py` | L | ✅ |
| 6.3 MLX quantization support | `quantization.py` | L | ✅ |
| 6.4 Visualization utilities | `visualization.py` | M | ✅ |
| 6.5 Whisper adapter (ASR loss) | `whisper_adapter.py` | XL | ✅ |
| 6.6 Hardware detection/optimization | `hardware.py` | S | ✅ |

**Deliverable**: Feature-complete MLX implementation

---

## File Structure Plan

```
df_mlx/
├── __init__.py
├── README.md
├── ROADMAP.md                    # This file
│
├── # Models
├── model.py                      # ✅ DFNet4, DFNet4Lite
├── deepfilternet.py              # ✅ DFNet1
├── deepfilternet2.py             # ✅ DFNet2
├── deepfilternet3.py             # ✅ DFNet3
├── deepfilternetmf.py            # ✅ DFNetMF
├── discriminator.py              # ✅ MPD, MSD
│
├── # Core modules
├── modules.py                    # ✅ Conv2dNormAct, GroupedLinear, etc.
├── mamba.py                      # ✅ Mamba, SqueezedMamba
├── ops.py                        # ✅ DfOp, ERB, Mask
├── multiframe.py                 # ✅ Multi-frame processing
│
├── # Training
├── train.py                      # ✅ Basic training (needs LR scheduler)
├── train_dynamic.py              # ✅ Dynamic mixing training
├── # (GAN training folded into train_dynamic.py)
├── loss.py                       # ✅ All loss functions
├── lr.py                         # ✅ LR schedulers
│
├── # Inference
├── enhance.py                    # ✅ Enhancement utilities
├── model_utils.py                # ✅ Model loading/downloading
│
├── # Evaluation
├── evaluation.py                 # ✅ Evaluation metrics/utilities
├── stoi.py                       # ✅ STOI metric
├── dnsmos.py                     # ✅ DNSMOS integration
│
├── # Data
├── datastore.py                  # ✅ Legacy datastore
├── dynamic_dataset.py            # ✅ Dynamic mixing dataset
├── build_audio_cache.py          # ✅ Cache builder
├── prepare_data.py               # ✅ Data preparation
├── generate_file_lists.py        # ✅ File list generation
│
├── # Utilities
├── config.py                     # ✅ Configuration
├── utils.py                      # ✅ General utilities
├── checkpoint.py                 # ✅ Checkpoint management
├── convert.py                    # ✅ PyTorch checkpoint conversion
├── visualization.py              # ✅ Visualization
├── quantization.py               # ✅ MLX quantization
├── hardware.py                   # ✅ Hardware detection
├── whisper_adapter.py            # ✅ Whisper integration (MLX-first)
│
└── # Tests
    ├── test_mlx.py               # ✅ Basic tests
    └── test_mlx_comprehensive.py # ✅ Comprehensive tests
```

---

## Complexity Legend

| Size | Hours | Description |
|------|-------|-------------|
| **S** | 2-8 | Single function/class, well-defined |
| **M** | 8-24 | Multiple components, integration needed |
| **L** | 24-60 | Major architecture, extensive testing |
| **XL** | 60+ | Major subsystem, external dependencies |

---

## Quick Wins (Do These First)

1. **LR Schedulers** (S) - Immediate training improvement
2. **DF Alpha Loss** (S) - Prevents over-filtering
3. **SI-SDR Metric** (S) - Simple eval metric
4. **PESQ Wrapper** (S) - External lib, just wrap it
5. **Early Stopping** (S) - Prevents overfitting

---

## Dependencies Graph

```
LR Schedulers ─────────────────────────────────────┐
                                                   │
DF Alpha Loss ─────────────────────────────────────┼──▶ Better Training
                                                   │
Early Stopping ────────────────────────────────────┘

PeriodDiscriminator ──┬──▶ MultiPeriodDiscriminator ──┐
                      │                               │
ScaleDiscriminator ───┴──▶ MultiScaleDiscriminator ───┼──▶ GAN Training
                                                      │
GAN Losses ───────────────────────────────────────────┘

SI-SDR ────────┬
STOI ──────────┼──▶ Composite Metrics ──▶ Evaluation Loop
PESQ ──────────┤
DNSMOS ────────┘

GroupedGRU ──┬──▶ SqueezedGRU ──▶ DFNet3
             │
             ├──▶ DFNet2
             │
             └──▶ DFNet1
```

---

## Notes

- **GAN training** is the highest-impact missing feature for audio quality
- **Evaluation metrics** can use PyTorch implementations via subprocess if needed
- **GRU-based models** require significant effort; prioritize only if legacy support needed
- **Whisper adapter** is complex and may require waiting for mlx-whisper maturity

---

## Contributing

When implementing a component:
1. Port from `df/` equivalent when possible
2. Add comprehensive tests
3. Update this roadmap
4. Document any MLX-specific adaptations

---

*Last updated: 2026-01-08*
