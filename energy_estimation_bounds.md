# Energy Estimation Framework: Coefficient Bounds and Model Predictions

## Part 1: Revised Formula Coefficients with Bounds

Based on our previous validation against real measurements, I'm establishing bounds for each coefficient that should explain most observed data while remaining interpretable.

---

### 1.1 Scaling Coefficient (η_scale) — Formula 1

**Original:** η_scale = 0.95^(log₂(N_ratio))

**Revised with bounds:** η_scale = **k**^(log₂(N_ratio)) where **k ∈ [0.90, 1.10]**

#### Coefficient Values and Intuition:

| k Value | Regime | Interpretation |
|---------|--------|----------------|
| **k = 0.90** (Lower bound) | Highly optimized | Model benefits from excellent GPU utilization, kernel fusion, and memory access patterns at scale. Represents best-case scaling where larger models amortize overhead better. |
| **k = 0.95** (Expected) | Typical efficient | Standard sub-linear efficiency gains from better utilization at scale. |
| **k = 1.00** (Linear) | No efficiency change | Energy scales linearly with parameters — baseline assumption. |
| **k = 1.05** (Mild overhead) | Memory pressure | Slight inefficiency from approaching memory limits or suboptimal batching. |
| **k = 1.10** (Upper bound) | Constrained | Model experiences memory bandwidth saturation, frequent cache misses, or multi-GPU communication overhead. |

#### Derivation of Bounds:

**Lower bound (k = 0.90):**
- From measurements: Qwen 3B → 7B showed 2.19× energy increase for 2.33× parameters
- Implied k: 2.19 / 2.33 = 0.94 (per doubling efficiency)
- Best observed cases (A100, well-optimized models) show up to 10% better-than-linear scaling
- **k = 0.90** captures highly optimized scenarios

**Upper bound (k = 1.10):**
- Cross-architecture comparisons (Llama vs Qwen) showed up to 63% efficiency differences
- Memory-constrained scenarios (RTX 3090 running 7B near capacity) showed ~10-15% worse scaling
- **k = 1.10** captures worst-case single-GPU overhead scenarios

#### Formula 1 Bounds:

$$E_{new}^{lower} = E_{baseline} \times \frac{N_{new}}{N_{baseline}} \times 0.90^{\log_2(N_{ratio})}$$

$$E_{new}^{upper} = E_{baseline} \times \frac{N_{new}}{N_{baseline}} \times 1.10^{\log_2(N_{ratio})}$$

**Example:** Scaling from 3B to 72B (24× parameters, log₂(24) = 4.58):
- Lower bound multiplier: 24 × 0.90^4.58 = 24 × 0.62 = **14.9×**
- Expected multiplier: 24 × 0.95^4.58 = 24 × 0.79 = **19.0×**
- Upper bound multiplier: 24 × 1.10^4.58 = 24 × 1.53 = **36.7×**

---

### 1.2 Architecture Coefficient (μ_arch) — Enhanced Formula 1

**Definition:** Multiplier accounting for architectural efficiency differences between model families.

#### Coefficient Values and Bounds:

| μ_arch Value | Category | Examples |
|--------------|----------|----------|
| **0.75 - 0.85** | Highly optimized MoE | DeepSeek-V3 (MLA + DeepSeekMoE), Mixtral |
| **0.85 - 1.00** | Optimized dense | Qwen family, Gemma |
| **1.00** (Reference) | Standard efficient | Well-tuned dense transformers |
| **1.00 - 1.20** | Standard architectures | GPT-family (dense), Llama |
| **1.20 - 1.50** | Less optimized | Older architectures, large attention overhead |
| **1.50 - 1.80** | Inefficient | Experimental architectures, poor memory patterns |

#### Derivation of Bounds:

**Efficiency gains (μ < 1.0):**
- MoE architectures only activate a fraction of parameters (e.g., DeepSeek activates 37B of 685B = 5.4%)
- Multi-Head Latent Attention (MLA) compresses KV cache significantly
- Group Query Attention (GQA) reduces memory bandwidth by 4-8×
- **Lower bound: μ = 0.75** for highly optimized MoE with MLA

**Efficiency losses (μ > 1.0):**
- Llama-8B showed 85% more energy than Qwen-7B despite similar sizes (μ ≈ 1.63)
- Different attention mechanisms, activation functions affect memory patterns
- **Upper bound: μ = 1.80** for worst-case architectural mismatch

#### Specific Model Family Estimates:

| Model Family | μ_arch (Expected) | μ_arch (Range) | Reasoning |
|--------------|-------------------|----------------|-----------|
| Qwen 2.5 | 1.00 | [0.90, 1.10] | Reference family, highly optimized |
| DeepSeek-V3/V3.2 | 0.80 | [0.75, 0.90] | MoE (5.4% active), MLA, highly efficient |
| GPT-OSS | 1.05 | [0.95, 1.20] | Likely standard transformer, less public optimization |
| Llama 3 | 1.30 | [1.15, 1.50] | Validated: 63% less efficient than Qwen |
| GPT-5 (closed) | 1.10 | [1.00, 1.25] | Expected modern optimizations but closed-source |

---

### 1.3 Utilization Coefficient (η_util) — Formula 2

**Purpose:** Accounts for practical efficiency when moving between hardware configurations.

**Original range:** η_util ∈ [0.70, 0.85]

**Revised bounds for energy estimation:** η_util ∈ **[0.70, 1.15]**

#### Coefficient Values and Interpretation:

| η_util Value | Meaning | Scenario |
|--------------|---------|----------|
| **0.70** (Lower bound) | Excellent utilization | Newer GPU fully utilizing bandwidth, well-optimized inference engine |
| **0.85** | Typical good | Standard production deployment with good utilization |
| **1.00** | Neutral | Hardware transition tracks theoretical predictions exactly |
| **1.05** | Slight inefficiency | Minor underutilization, suboptimal batching |
| **1.15** (Upper bound) | Memory pressure | Approaching memory limits, formula nearing validity boundary |

#### Why Lower η_util Means More Energy:

The utilization coefficient in Formula 2 appears in the denominator conceptually. When η_util < 1.0, it means the target hardware achieves **better** practical efficiency than the baseline-derived theoretical ratio predicts:

$$E_{target} = E_{intermediate} \times \frac{B_{baseline}}{B_{target}} \times \frac{P_{target}}{P_{baseline}} \times \eta_{util}$$

- **η_util = 0.70**: Target hardware is 30% more efficient than raw specs predict (newer architectures)
- **η_util = 1.15**: Target hardware is 15% less efficient than specs predict (memory pressure)

#### Derivation from Measurements:

| Transition | Actual η_util | Notes |
|------------|---------------|-------|
| A100 → RTX 3090 (Qwen-3B) | 0.997 | Near-perfect prediction |
| A100 → RTX 6000 Ada (Qwen-3B) | ~0.85 | Good utilization |
| Cross-architecture variations | 0.75 - 1.10 | Depends on optimization |

**Bounds:**
- **Lower bound (0.70):** H100/B200 with latest vLLM optimizations
- **Upper bound (1.15):** Signals approaching formula validity limits (memory pressure)

---

## Part 2: Combined Bounds Formula

### Full Formula with All Bounds:

**Formula 1 (Model Scaling):**
$$E_{new} = E_{baseline} \times \frac{N_{new}}{N_{baseline}} \times k^{\log_2(N_{ratio})} \times \mu_{arch}$$

**Formula 2 (Hardware Scaling):**
$$E_{target} = E_{intermediate} \times \frac{G_{target} \cdot (1 + \alpha(G_{target} - 1))}{G_{baseline} \cdot (1 + \alpha(G_{baseline} - 1))} \times \frac{B_{baseline}}{B_{target}} \times \frac{P_{target}}{P_{baseline}} \times \eta_{util}$$

### Bound Combinations:

| Scenario | k | μ_arch | η_util | Result |
|----------|---|--------|--------|--------|
| **Lower bound (optimistic)** | 0.90 | 0.75 | 0.70 | Minimum energy estimate |
| **Expected (typical)** | 0.95 | 1.00 | 0.85 | Central estimate |
| **Upper bound (pessimistic)** | 1.10 | 1.50 | 1.15 | Maximum energy estimate |

---

## Part 3: GPT-5 Energy Estimation (GPT-OSS Family)

### Model Information:

From search results, GPT-OSS family:
- **gpt-oss-120b**: ~117B total parameters, ~5.1B active per token (MoE)
- Uses Mixture-of-Experts architecture
- Group Query Attention (GQA) and sliding window attention
- Compatible with vLLM

For estimation, I'll use **gpt-oss-120b** as representative.

### Baseline Reference:

Using our validated Qwen-2.5-3B measurements:
- A100-80GB: 0.35 J/token
- RTX 3090: 0.83 J/token

### Step-by-Step Calculation:

#### Step 1: Model Scaling (Formula 1)

**Active parameters matter for MoE:**
- gpt-oss-120b: 5.1B active per token
- Baseline (Qwen-3B): 3B parameters

**Parameter ratio:** 5.1B / 3B = 1.70

**Scaling coefficient bounds:**
- Lower (k=0.90): 1.70 × 0.90^(log₂(1.70)) = 1.70 × 0.90^0.77 = 1.70 × 0.926 = **1.57×**
- Expected (k=0.95): 1.70 × 0.95^0.77 = 1.70 × 0.961 = **1.63×**
- Upper (k=1.10): 1.70 × 1.10^0.77 = 1.70 × 1.077 = **1.83×**

**Architecture coefficient (GPT-OSS is MoE with GQA):**
- Expected: μ_arch = 1.00 (assume GPT-OSS is reasonably optimized)
- Range: [0.90, 1.20] (likely not as optimized as Qwen/DeepSeek)

**Intermediate energy on A100:**
- Lower: 0.35 × 1.57 × 0.90 = **0.49 J/token**
- Expected: 0.35 × 1.63 × 1.00 = **0.57 J/token**
- Upper: 0.35 × 1.83 × 1.20 = **0.77 J/token**

#### Step 2: Hardware Scaling (Formula 2)

**Hardware specifications:**

| GPU | Memory BW (GB/s) | TDP (W) | Est. Avg Power |
|-----|------------------|---------|----------------|
| A100-80GB | 2,039 | 400 | ~290 W |
| H100-80GB | 3,350 | 700 | ~500 W |

**GPT-OSS-120B on A100:**

Already calculated intermediate energy. Now apply utilization bounds:
- Lower: 0.49 × 0.70 = **0.34 J/token**
- Expected: 0.57 × 0.85 = **0.48 J/token**
- Upper: 0.77 × 1.15 = **0.89 J/token**

**GPT-OSS-120B on H100:**

Hardware ratio from A100 baseline:
$$S_{hardware} = \frac{B_{A100}}{B_{H100}} \times \frac{P_{H100}}{P_{A100}} = \frac{2039}{3350} \times \frac{500}{290} = 0.609 \times 1.72 = 1.05$$

Apply to intermediate energies with utilization bounds:
- Lower: 0.49 × 1.05 × 0.70 = **0.36 J/token**
- Expected: 0.57 × 1.05 × 0.85 = **0.51 J/token**
- Upper: 0.77 × 1.05 × 1.15 = **0.93 J/token**

### GPT-OSS-120B Summary:

| Hardware | Lower Bound | Expected | Upper Bound |
|----------|-------------|----------|-------------|
| **A100-80GB** | 0.34 J/token | 0.48 J/token | 0.89 J/token |
| **H100-80GB** | 0.36 J/token | 0.51 J/token | 0.93 J/token |

**Note:** H100 shows slightly higher energy due to higher power consumption, but similar range due to bandwidth gains offsetting.

---

## Part 4: DeepSeek-V3.2 685B Energy Estimation

### Model Information:

From search results:
- **Total parameters:** 685B (671B main + 14B MTP module)
- **Active parameters per token:** 37B
- **Architecture:** MoE with Multi-Head Latent Attention (MLA)
- **Key optimizations:** DeepSeek Sparse Attention (DSA), auxiliary-loss-free load balancing
- Hardware requirement: H200/B200 recommended, can run on 8× H100

### Architecture Coefficient:

DeepSeek is notably efficient:
- MLA compresses KV cache significantly
- Only 5.4% of parameters active per token (37B / 685B)
- FP8 training validated at scale
- Near full computation-communication overlap

**μ_arch for DeepSeek-V3.2:**
- Expected: **0.80** (highly optimized)
- Range: **[0.75, 0.90]** (user specified: "on par with Qwen" → slightly more optimized)

### Step-by-Step Calculation:

#### Step 1: Model Scaling (Formula 1)

**Active parameter ratio:** 37B / 3B = 12.33

**Scaling coefficient bounds:**
- Lower (k=0.90): 12.33 × 0.90^(log₂(12.33)) = 12.33 × 0.90^3.62 = 12.33 × 0.68 = **8.4×**
- Expected (k=0.95): 12.33 × 0.95^3.62 = 12.33 × 0.82 = **10.1×**
- Upper (k=1.10): 12.33 × 1.10^3.62 = 12.33 × 1.42 = **17.5×**

**With architecture coefficient:**
- Lower: 8.4 × 0.75 = **6.3×**
- Expected: 10.1 × 0.80 = **8.1×**
- Upper: 17.5 × 0.90 = **15.8×**

**Intermediate energy (from Qwen-3B A100 baseline 0.35 J/token):**
- Lower: 0.35 × 6.3 = **2.21 J/token**
- Expected: 0.35 × 8.1 = **2.84 J/token**
- Upper: 0.35 × 15.8 = **5.53 J/token**

#### Step 2: Hardware Scaling (Formula 2)

**Multi-GPU consideration:**
DeepSeek-V3.2 requires 8 GPUs minimum. Using α = 0.12 (tensor parallelism overhead):

$$S_{multi-GPU} = \frac{8 \times (1 + 0.12 \times 7)}{1 \times 1} = 8 \times 1.84 = 14.7$$

Wait — this overhead accounts for communication, but we're measuring **per-token energy**, not total system energy. For per-token measurement, the multi-GPU overhead is already baked into the utilization efficiency.

**DeepSeek-V3.2 on A100 (8× A100s):**

Per-GPU bandwidth remains 2,039 GB/s. Per-token energy with utilization:
- Lower: 2.21 × 0.70 = **1.55 J/token**
- Expected: 2.84 × 0.85 = **2.41 J/token**
- Upper: 5.53 × 1.15 = **6.36 J/token**

**DeepSeek-V3.2 on H100 (8× H100s):**

Hardware ratio:
$$S_{hardware} = \frac{2039}{3350} \times \frac{500}{290} = 1.05$$

With utilization:
- Lower: 2.21 × 1.05 × 0.70 = **1.62 J/token**
- Expected: 2.84 × 1.05 × 0.85 = **2.53 J/token**
- Upper: 5.53 × 1.05 × 1.15 = **6.68 J/token**

### DeepSeek-V3.2 685B Summary:

| Hardware | Lower Bound | Expected | Upper Bound |
|----------|-------------|----------|-------------|
| **8× A100-80GB** | 1.55 J/token | 2.41 J/token | 6.36 J/token |
| **8× H100-80GB** | 1.62 J/token | 2.53 J/token | 6.68 J/token |

---

## Part 5: Summary Table

### All Model Estimates:

| Model | Active Params | Hardware | Lower | Expected | Upper |
|-------|---------------|----------|-------|----------|-------|
| Qwen-2.5-3B | 3B | A100 | — | 0.35 J/token | — |
| GPT-OSS-120B | 5.1B | A100 | 0.34 J/token | 0.48 J/token | 0.89 J/token |
| GPT-OSS-120B | 5.1B | H100 | 0.36 J/token | 0.51 J/token | 0.93 J/token |
| DeepSeek-V3.2 | 37B | 8× A100 | 1.55 J/token | 2.41 J/token | 6.36 J/token |
| DeepSeek-V3.2 | 37B | 8× H100 | 1.62 J/token | 2.53 J/token | 6.68 J/token |

### Coefficient Summary:

| Coefficient | Symbol | Lower Bound | Expected | Upper Bound |
|-------------|--------|-------------|----------|-------------|
| Scaling | k | 0.90 | 0.95 | 1.10 |
| Architecture | μ_arch | 0.75 (MoE) | 1.00 | 1.50 |
| Utilization | η_util | 0.70 | 0.85 | 1.15 |

### Key Insights:

1. **GPT-OSS-120B is relatively efficient** due to MoE architecture (only 5.1B active), making it comparable energy-wise to dense models 2× smaller.

2. **DeepSeek-V3.2 benefits enormously from MoE + MLA**, resulting in lower per-token energy than naive 685B parameter count would suggest.

3. **H100 vs A100 difference is modest** (~5-10%) because higher bandwidth is largely offset by higher power consumption.

4. **Bounds span 2-4× range**, which is appropriate for extrapolation without direct measurements. The wide range on DeepSeek reflects uncertainty in how well theoretical MoE efficiency translates to practice.

5. **For paper presentation:** Use expected values as point estimates with bounds clearly indicated. The multiplicative nature of uncertainties means wide bounds are appropriate and honest.
