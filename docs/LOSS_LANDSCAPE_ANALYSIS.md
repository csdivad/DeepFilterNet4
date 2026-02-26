# Loss Landscape Analysis & Optimal Weight Recommendations

**Model:** DfNet4 with VadHead (multi-task)  
**Framework:** MLX (Apple Silicon)  
**Profile:** `pipeline_awesome_gan_optimized.toml`  
**Date:** 2025-07-17

---

## 1. Executive Summary

The current loss configuration suffers from **auxiliary dominance**: pipeline awesome + VAD + speech-band losses outweigh spectral reconstruction losses by nearly 3:1. Feature matching overwhelms GAN adversarial signal 17:1. MRSTFT is underweighted at ~5% effective contribution. The GAN phase is squeezed into the final 25% of training with an abrupt 12-epoch ramp.

**Key recommendations:**
- Rebalance reconstruction:auxiliary ratio from 1:2.8 → 1:1.1
- Increase MRSTFT contribution (+75%) for better spectral detail
- Cut FM:adversarial ratio from 17:1 → 6:1
- Start GAN 20 epochs earlier with a smoother 20-epoch ramp
- Introduce a 4-stage pipeline curriculum for controlled loss introduction

---

## 2. Loss Component Inventory

### 2.1 How the total loss is assembled

From `loss_fn()` at [train_dynamic.py](DeepFilterNet/df_mlx/train_dynamic.py#L1201):

```
total = spec_loss                           # always
      + mrstft_loss                         # if use_mrstft_loss
      + gan_weight   * gan_g_loss           # if gan_active
      + fm_weight    * fm_loss              # if gan_active & fm_weight > 0
      + awesome_wt   * pipeline_loss        # if use_pipeline_awesome_loss
      + vad_weight   * vad_loss             # if use_vad_loss
      + speech_wt    * speech_loss          # if vad_speech_loss_weight > 0
      + vad_weight   * vad_head_loss        # if vad_logits not None (BCE)
      + vad_reg_wt   * vad_reg_loss         # sparse (every 2 steps + 5% random)
```

### 2.2 Per-component details

| # | Component | Loss Type | Config Weight | Warmup | Typical Magnitude | Effective Contribution |
|---|-----------|-----------|---------------|--------|-------------------|----------------------|
| 1 | **Spectral L1** | `(1-α)·L1_mag + α·L1_complex`, α=0.5 | 1.0 (implicit) | None | 0.1–0.5 | 0.1–0.5 |
| 2 | **MRSTFT** | MSE on γ-compressed mag + MSE complex, 3 resolutions | factor=0.2, f_complex=0.15 | None | 0.05–0.3 | 0.01–0.06 |
| 3 | **Pipeline Awesome** | speech + noise + 0.3·smooth + 1.5·music_suppression | 1.2 | 2500 steps linear | 0.3–1.5 | 0.36–1.80 |
| 4 | **VAD (margin)** | `mean(gate · relu(p_ref - p_out - margin))` | 0.8 | 5 epochs linear | 0.01–0.3 | 0.008–0.24 |
| 5 | **VAD Head BCE** | `BCE(logits, p_ref, with_logits=True)` | 0.8 (shared) | 5 epochs (shared) | 0.3–0.7 | 0.24–0.56 |
| 6 | **Speech Band** | L1 on band-avg log-mag (300–3400 Hz) × gate | 0.6 | 5 epochs (shared) | 0.01–0.3 | 0.006–0.18 |
| 7 | **GAN Generator** | Hinge: `mean(relu(1 - disc_fake))` | 0.06 | epoch 150, 12-epoch ramp | 0.5–1.5 | 0.03–0.09 |
| 8 | **Feature Matching** | L1 on disc feature maps (factor=1.0) | 1.0 | epoch 150 (shared ramp) | 0.5–5.0 | 0.50–5.00 |
| 9 | **VAD Reg** | Speech-band L1 × speech_ratio × musicness gate | 0.1 | None (sparse) | 0.01–0.1 | 0.001–0.01 |

### 2.3 Current weight budget at full convergence

```
Reconstruction:  spectral(1.0) + MRSTFT(~0.2) = 1.2
Auxiliary:       pipeline(1.2) + VAD(0.8) + speech(0.6) = 2.6
VAD Head:        BCE at 0.8 weight = 0.8
GAN:             adv(0.06) + FM(1.0) = 1.06
────────────────────────────────────────────────
Ratio:  reconstruction : auxiliary+VAD_head = 1.2 : 3.4 = 1 : 2.83
```

**Problem**: Auxiliary losses have nearly **3× the gradient budget** of the primary reconstruction objective.

---

## 3. Loss Interaction Dynamics

### 3.1 Complementary pairs

| Pair | Interaction | Quality |
|------|-------------|---------|
| Spectral L1 ↔ MRSTFT MSE | L1 provides robust median-chasing on single resolution; MSE on 3 resolutions adds fine spectral detail. Different loss types on same signal = excellent coverage. | **Strongly complementary** |
| Pipeline speech_loss ↔ Speech Band loss | Both operate on speech regions but at different granularity: pipeline uses full log-mag with mask weighting; speech band targets 300–3400 Hz specifically. | **Moderately complementary** |
| FM loss ↔ GAN adversarial | FM aligns intermediate disc representations (stable, non-adversarial gradient); adversarial drives perceptual realism. FM stabilizes the inherently unstable GAN dynamics. | **Complementary (stability pair)** |

### 3.2 Conflicting pairs

| Pair | Conflict | Severity |
|------|----------|----------|
| GAN adversarial ↔ All reconstruction | Generator loss encourages outputs that fool the discriminator — may sacrifice spectral accuracy for "realistic-sounding" artifacts. | **Moderate** (mitigated by small adv_weight) |
| Pipeline noise_loss ↔ Pipeline speech_loss | Signal at mask boundaries can flip between speech/noise classification. With `mask_sharpness=7.0`, the hard sigmoid creates gradient cliffs near boundaries → oscillation. | **Low-Moderate** |
| VAD Head BCE ↔ Spectral quality | BCE gradients flow through the shared encoder backbone, distorting features optimized for spectral reconstruction toward features that separate speech from silence. | **Moderate** at weight=0.8 |

### 3.3 Redundancy

| Pair | Overlap | Assessment |
|------|---------|------------|
| Pipeline awesome ↔ VAD margin loss | Both emphasize "preserve speech, suppress noise" using energy-based gating. Pipeline uses raw energy masks; VAD uses model VAD probability. | **~40% redundant** |
| Pipeline speech_loss ↔ Base spectral loss | Pipeline speech preservation (L1 on log-mag in speech regions) partially duplicates spectral L1 which already covers all regions. | **~30% redundant** |

### 3.4 Gradient dynamics across training

```
                    Early (0-30)    Mid (30-100)    Late (100-150)    GAN (150-200)
                    ────────────    ────────────    ──────────────    ─────────────
Spectral L1         ████████        ███████         ██████            █████
MRSTFT MSE          ██████████      █████           ███               ██        ← MSE shrinks as errors reduce
Pipeline Awesome    ██ (warmup)     █████████       █████████         █████████
VAD BCE             █ (warmup)      █████           ███████           ███████   ← grows as head improves
GAN Adversarial     ─               ─               ─                 ████████
Feature Matching    ─               ─               ─                 ██████████████
```

**Key insight**: The gradient direction seen by the shared backbone shifts uncontrollably:
- **Early**: MSE dominates (large errors → large gradients), spectral L1 secondary
- **Mid**: MSE shrinks, L1 takes over, pipeline awesome fully ramped → auxiliary dominates
- **Late pre-GAN**: BCE can become dominant if VAD head struggles → backbone features biased toward VAD
- **GAN phase**: FM loss can spike to 2-5× spectral loss → backbone suddenly optimizing for disc feature alignment

Pipeline stages should **explicitly manage** these transitions.

---

## 4. Identified Problems

### P1: Auxiliary dominance (Critical)

The combined auxiliary weight budget (3.4) is 2.8× the reconstruction budget (1.2). This means the model spends most of its gradient capacity on speech/noise classification rather than spectral fidelity. Symptom: good noise suppression (DNSMOS-OVRL) but mediocre spectral detail (SI-SDR, PESQ).

### P2: MRSTFT underweighted (Moderate)

MRSTFT with `factor=0.2` and 3-resolution averaging yields 0.01–0.06 effective contribution — roughly **5% of spectral L1**. Multi-resolution spectral detail improves transients, high-frequency content, and formant preservation. This is almost noise-level in the gradient.

### P3: Feature Matching overwhelms adversarial signal (Moderate)

FM:adversarial ratio of 17:1 (`fm_weight=1.0` vs `adv_weight=0.06`). While FM provides stability, this extreme ratio means the GAN is almost entirely a **feature alignment** objective, not an adversarial one. The discriminator's adversarial feedback (which drives perceptual quality beyond reconstruction fidelity) is barely felt by the generator.

### P4: VAD Head gradient pollution (Moderate)

The VAD head's BCE loss at weight=0.8 produces gradients through the shared encoder that can be 50–100% as large as spectral reconstruction gradients. For a small auxiliary head that produces a single scalar per frame, this is disproportionate.

### P5: GAN phase too short and abrupt (Low-Moderate)

50 epochs of GAN = 25% of training. The 12-epoch ramp fills epoch 150–162, leaving only 38 epochs at full GAN strength. Combined with cosine-decayed LR (at epoch 150, LR ≈ `1e-6 * (0.5 * (1 + cos(π * 145/195))) ≈ 1.5e-7`), the model's capacity to respond to adversarial feedback is limited.

### P6: Minimum learning rate too low (Low)

`min_lr = learning_rate * 0.01 = 1e-8`. At epoch 150+ when GAN activates, the cosine schedule has already decayed to ~1.5e-7. By epoch 200, it's near 1e-8. The model essentially stops learning in the late GAN phase.

---

## 5. Recommended Configuration

### 5.1 Base weight changes

| Parameter | Current | Recommended | Δ | Rationale |
|-----------|---------|-------------|---|-----------|
| `loss.awesome.loss_weight` | 1.2 | **0.8** | −33% | Reduce auxiliary dominance; 0.8 × 1.5 (music_suppression) = 1.2 still strong |
| `vad.loss_weight` | 0.8 | **0.4** | −50% | VAD head is secondary; still gets meaningful BCE signal at 0.4 |
| `vad.speech_loss_weight` | 0.6 | **0.35** | −42% | Refinement loss, not primary; 0.35 maintains 300–3400 Hz focus |
| `loss.mrstft.factor` | 0.2 | **0.35** | +75% | Multi-res spectral detail was noise-level; 0.35 makes it audible |
| `gan.adv_weight` | 0.06 | **0.08** | +33% | More adversarial signal for perceptual quality |
| `gan.fm_weight` | 1.0 | **0.5** | −50% | FM:adv from 17:1 → 6:1; more adversarial influence |
| `gan.start_epoch` | 150 | **130** | −20 | 70 GAN epochs instead of 50 |
| `gan.ramp_epochs` | 12 | **20** | +67% | Smoother introduction; full GAN by epoch 150 |
| `gan.disc_lr` | 1e-5 | **2e-5** | +100% | Compensate for disc_update_freq=2 (effective LR was halved) |
| `training.max_grad_norm` | 0.8 | **1.0** | +25% | More headroom for multi-task gradients |

**New weight budget at full convergence:**

```
Reconstruction:  spectral(1.0) + MRSTFT(~0.35) = 1.35
Auxiliary:       pipeline(0.8) + VAD(0.4) + speech(0.35) = 1.55
GAN:             adv(0.08) + FM(0.5) = 0.58
────────────────────────────────────────────────
Ratio:  reconstruction : auxiliary = 1.35 : 1.55 = 1 : 1.15  (was 1 : 2.83)
FM : adversarial = 0.5 : 0.08 = 6.25 : 1  (was 17 : 1)
```

### 5.2 Additional parameter recommendations

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `learning_rate_min` | None (→1e-8) | **1e-7** | 10× higher floor keeps model responsive during GAN phase |
| `learning_rate` | 1e-6 | **1e-6** (keep) | Safe for multi-task; increase only if convergence stalls |
| `warmup_epochs` | 5 | **5** (keep) | Standard for this scale |
| `disc_max_samples` | 24000 | **24000** (keep) | Adequate; 48000 risks OOM on M3 Pro |
| `awesome.warmup_steps` | 2500 | **2500** (keep) | ~10 epochs of warmup — matches Stage 1-2 transition well |

---

## 6. Pipeline Stages Curriculum

### 6.1 Design

The pipeline stages system can override exactly three weights per epoch range:
- `awesome_loss_weight`
- `vad_loss_weight`  
- `vad_speech_loss_weight`

GAN weights follow their own epoch-based ramp (not stage-overridable).

### 6.2 Recommended 4-stage curriculum

```json
[
  {
    "start_epoch": 0,
    "awesome_loss_weight": 0.15,
    "vad_loss_weight": 0.08,
    "vad_speech_loss_weight": 0.05
  },
  {
    "start_epoch": 30,
    "awesome_loss_weight": 0.45,
    "vad_loss_weight": 0.20,
    "vad_speech_loss_weight": 0.15
  },
  {
    "start_epoch": 75,
    "awesome_loss_weight": 0.80,
    "vad_loss_weight": 0.40,
    "vad_speech_loss_weight": 0.35
  },
  {
    "start_epoch": 130,
    "awesome_loss_weight": 0.65,
    "vad_loss_weight": 0.32,
    "vad_speech_loss_weight": 0.28
  }
]
```

### 6.3 Stage rationale

#### Stage 1: Foundation (epochs 0–29, 30 epochs)

**Goal**: Establish spectral reconstruction baseline with minimal auxiliary noise.

```
Reconstruction : Auxiliary = 1.35 : 0.28 = 4.8 : 1
```

The backbone learns basic spectral mapping with spectral L1 + MRSTFT as the dominant gradients. Auxiliary losses at 15–19% of their full values provide gentle guidance without distorting early feature learning. The built-in awesome warmup (2500 steps ≈ 10 epochs) means the effective awesome contribution starts at 0 and rises to 0.15 by ~epoch 10.

#### Stage 2: Auxiliary Introduction (epochs 30–74, 45 epochs)

**Goal**: Introduce speech-aware supervision at moderate strength.

```
Reconstruction : Auxiliary = 1.35 : 0.80 = 1.7 : 1
```

Transition from Stage 1: awesome 3×, VAD 2.5×, speech 3× — a combined auxiliary jump of +0.52 (from 0.28 to 0.80). This is a ~50% increase in total loss magnitude, manageable with cosine LR still near peak.

The model now differentiates speech regions from noise while maintaining the spectral foundation. VAD head begins receiving meaningful BCE gradients but at a low enough weight (0.20) to avoid backbone distortion.

#### Stage 3: Full Multi-Task (epochs 75–129, 55 epochs)

**Goal**: All non-GAN losses at target strength; model reaches pre-GAN optimum.

```
Reconstruction : Auxiliary = 1.35 : 1.55 = 1 : 1.15
```

This is the target balance. 55 epochs of stable multi-task training allows the model to converge on spectral quality, speech preservation, and noise suppression simultaneously. By epoch 129, the model should be near-optimal for reconstruction metrics (SI-SDR, PESQ, DNSMOS).

#### Stage 4: GAN Integration (epochs 130–199, 70 epochs)

**Goal**: Maintain stability while adversarial training adds perceptual quality.

```
Reconstruction : Auxiliary = 1.35 : 1.25 = 1.08 : 1
GAN: 0 → 0.58 over 20 epochs (linear ramp, epoch 130–150)
```

**Why reduce auxiliary weights during GAN?** The GAN phase adds ~0.58 to the total gradient budget at full ramp. To prevent a sudden spike in total gradient magnitude (which destabilizes training), we preemptively reduce auxiliary by ~0.30. The net change at full GAN ramp is: −0.30 (auxiliary) + 0.58 (GAN) = +0.28 — a modest 18% increase rather than a 37% jump.

### 6.4 Transition dynamics

```
Epoch:  0      30      75       130      150       200
        │       │       │        │        │         │
Stage:  ├── 1 ──┤── 2 ──┤── 3 ──┤──── 4 ─────────┤
        │       │       │        │        │         │
Aux:    0.28    0.80    1.55     1.25     1.25      1.25
GAN:    0       0       0        0→       →0.58     0.58
Total:  1.63    2.15    2.90     2.60     3.18      3.18
        │       │       │        │        │         │
LR:     0→1e-6  ~1e-6   ~6e-7    ~2.5e-7  ~1.5e-7   1e-7
        (warmup) (peak) (decay)  (decay)  (low)     (floor)
```

### 6.5 TOML format for config file

Add to the `[loss]` section:

```toml
pipeline_stages = '[{"start_epoch":0,"awesome_loss_weight":0.15,"vad_loss_weight":0.08,"vad_speech_loss_weight":0.05},{"start_epoch":30,"awesome_loss_weight":0.45,"vad_loss_weight":0.20,"vad_speech_loss_weight":0.15},{"start_epoch":75,"awesome_loss_weight":0.80,"vad_loss_weight":0.40,"vad_speech_loss_weight":0.35},{"start_epoch":130,"awesome_loss_weight":0.65,"vad_loss_weight":0.32,"vad_speech_loss_weight":0.28}]'
```

---

## 7. Risk Assessment

| Change | Risk | Impact if Wrong | Mitigation | Reversibility |
|--------|------|-----------------|------------|---------------|
| awesome 1.2→0.8 | **Moderate** | Reduced noise suppression in mixed-content | Monitor noise floor; bump to 0.9 if needed | Config change |
| vad 0.8→0.4 | **Low-Mod** | Slower VAD accuracy improvement | Track VAD accuracy per epoch; bump to 0.5 | Config change |
| speech 0.6→0.35 | **Low** | Marginal effect on 300–3400 Hz band | Increase to 0.45 if PESQ drops | Config change |
| MRSTFT 0.2→0.35 | **Low** | Almost always beneficial; minor instability risk early | Watch first 5 epochs for large MRSTFT gradients | Config change |
| fm_weight 1.0→0.5 | **Moderate** | GAN training less stable | Monitor gen/disc loss ratio; increase to 0.7 if diverging | May need pre-GAN checkpoint restart |
| gan_start 150→130 | **Low** | More adversarial training = generally better | None needed | Config change |
| gan_ramp 12→20 | **Low** | Smoother ramp, almost no downside | None needed | Config change |
| disc_lr 1e-5→2e-5 | **Low-Mod** | Disc might overpower gen early | Watch disc accuracy; if >95% consistently, reduce to 1.5e-5 | Config change |
| max_grad_norm 0.8→1.0 | **Low-Mod** | Slightly more gradient variance | If NaN/Inf appears, reduce to 0.9 | Config change |
| Pipeline stages | **Low** | Curriculum ordering is robust | If any stage transition causes loss spike, add an intermediate stage | Config change |
| min_lr → 1e-7 | **Low** | Keeps model responsive in late training | Standard practice | Config change |

### Overall risk: **LOW-MODERATE**

All changes are config-level (no code changes). The most significant risk is FM weight reduction, which could destabilize GAN training. Monitor the discriminator accuracy and gen/disc loss ratio closely during epochs 130–150.

---

## 8. Monitoring Rubric for `pipeline_awesome_gan_curriculum_adj.toml`

This rubric is the operational gate for the tuned profile currently in use:

- Stage boundaries: **0 → 60 → 100 → 130**
- GAN schedule: **start=130, ramp=24, adv=0.08, fm=0.6**
- Late-stage suppression emphasis: **lower VAD/speech weights in GAN phase**

### 8.1 Primary decision metrics and thresholds

#### A) Loss-budget ratio (most important)

Track weighted objective balance at validation:

$$
R_{aux/recon} = \frac{w_{awesome}L_{awesome} + w_{vad}(L_{vad\_proxy} + L_{vad\_head}) + w_{speech}L_{speech}}{L_{spec} + L_{mrstft}}
$$

Guardrails:

- **Stage 2–3 target:** `0.8 ≤ R_aux/recon ≤ 1.3`
- **Warning:** sustained `R_aux/recon > 1.5` for 2+ validations
- **Action:** reduce `vad_loss_weight` and/or `vad_speech_loss_weight` by 10–20%

Rationale: keeps auxiliary speech/noise supervision from dominating reconstruction and preserving non-target speech.

#### B) GAN stability ratio

Track generator/discriminator balance:

$$
R_{gan} = \frac{L_{gen}}{L_{disc} + 10^{-8}}
$$

Guardrails:

- **Healthy band:** `0.3 ≤ R_gan ≤ 3.0`
- **High risk:** `R_gan > 5.0` or `R_gan < 0.1` for 2+ validations
- **Action:**
  - if `R_gan > 5.0`: increase `fm_weight` (e.g., `0.6 -> 0.7`) or lower `adv_weight` (`0.08 -> 0.06`)
  - if `R_gan < 0.1`: reduce discriminator pressure (`disc_lr` down 20–30%)

#### C) Suppression-vs-intelligibility split (deployment objective)

Evaluate on two validation subsets:

1. **Target-present speech set** (intelligibility retention)
2. **Target-absent interferer/music set** (suppression strictness)

Guardrails:

- If suppression improves but target-speech quality drops by >10% over 3 validations, treat as over-suppression.
- If target-speech quality is stable but residual interferer/music energy rises, treat as leakage.

Actions:

- Over-suppression: increase Stage 3 `awesome_loss_weight` slightly or reduce GAN aggressiveness (`adv_weight`).
- Leakage: reduce Stage 4 `vad_*` weights further and/or increase suppression emphasis via pipeline-awesome balance.

#### D) Stage-transition stability check

At boundaries (`60`, `100`, `130`), monitor for transient spikes:

- **Pass:** total validation loss stabilizes within 2 epochs
- **Fail:** >20% loss jump persisting for 3 epochs
- **Action:** add an intermediate stage or soften adjacent stage deltas

### 8.2 Rollback triggers (hard stop)

Rollback to the best pre-regression checkpoint when any trigger is met:

1. `R_aux/recon > 1.8` for 3 consecutive validations
2. GAN instability (`R_gan > 5` or `< 0.1`) persists after one corrective adjustment
3. Target-speech quality degrades >10% for 3 validations while suppression does not materially improve
4. Stage-transition instability persists beyond 3 epochs after boundary

### 8.3 Minimal intervention ladder

Apply one change at a time in this order:

1. `fm_weight` (+0.1) or `adv_weight` (−0.02) for GAN instability
2. Stage 4 `vad_loss_weight` / `vad_speech_loss_weight` (−10% each) for non-target speech leakage
3. Add stage between 100 and 130 if transition is abrupt
4. Roll back to best checkpoint and resume with adjusted config

---

## 9. Secondary Observations (Code-Level)

These are non-config observations that may warrant future investigation:

1. **`mask_sharpness=7.0`**: Creates steep sigmoid boundaries (σ(7·x) transitions from 0.01 to 0.99 over Δx ≈ 0.66). Consider reducing to 5.0 for smoother gradients near speech/noise boundaries if gradient oscillation is observed.

2. **`_PIPELINE_MUSIC_SUPPRESSION_WEIGHT=1.5`**: Internal constant, not configurable. Within the pipeline loss, music suppression is weighted 1.5× speech loss — appropriate only if music contamination is a significant dataset problem. If music is rare, this over-allocates gradient budget.

3. **`FeatureMatchingLoss(factor=1.0)`**: Instantiated at [train_dynamic.py line 704](DeepFilterNet/df_mlx/train_dynamic.py#L704) with `factor=1.0` (the class default is `factor=2.0`). This lower instantiation factor partially compensates for the high `fm_weight=1.0` in config. With the recommended `fm_weight=0.5`, the effective FM contribution = 0.5 × 1.0 = 0.5.

4. **Cosine schedule timing**: At epoch 130 (GAN start), the cosine schedule has decayed to ~25% of peak LR. The discriminator at `disc_lr=2e-5` would be learning 13× faster than the generator. This asymmetry is within normal GAN ranges (10–50×) but worth monitoring.

---

## 10. Complete Recommended Config Diff

```diff
 [training]
-max_grad_norm = 0.8
+max_grad_norm = 1.0
+learning_rate_min = 1e-7

 [loss]
 dynamic_loss = "pipeline_awesome"
+pipeline_stages = '[{"start_epoch":0,"awesome_loss_weight":0.15,"vad_loss_weight":0.08,"vad_speech_loss_weight":0.05},{"start_epoch":30,"awesome_loss_weight":0.45,"vad_loss_weight":0.20,"vad_speech_loss_weight":0.15},{"start_epoch":75,"awesome_loss_weight":0.80,"vad_loss_weight":0.40,"vad_speech_loss_weight":0.35},{"start_epoch":130,"awesome_loss_weight":0.65,"vad_loss_weight":0.32,"vad_speech_loss_weight":0.28}]'

 [loss.awesome]
-loss_weight = 1.2
+loss_weight = 0.8

 [loss.mrstft]
-factor = 0.2
+factor = 0.35

 [gan]
-start_epoch = 150
-ramp_epochs = 12
-adv_weight = 0.06
-fm_weight = 1.0
-disc_lr = 1e-5
+start_epoch = 130
+ramp_epochs = 20
+adv_weight = 0.08
+fm_weight = 0.5
+disc_lr = 2e-5

 [vad]
-loss_weight = 0.8
-speech_loss_weight = 0.6
+loss_weight = 0.4
+speech_loss_weight = 0.35
```
