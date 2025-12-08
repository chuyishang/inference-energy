# Inference Energy Toolkit

Measure the environmental impact of running inference on machine learning models, with a concrete vLLM-oriented protocol and utilities that emphasize reproducibility and realistic serving conditions.

## Contents
- Protocol for vLLM energy and carbon measurement
- Alternative (non-runtime) energy estimation approaches
- Revised metric family to replace ESS for LLM serving

---

## 1) vLLM Energy & Carbon Measurement Protocol

### Design goals
- Target energy per token/request under real serving conditions
- Capture batching, KV cache, and concurrency effects
- Reproduce across models/prompts with steady-state load
- Attribute energy to requests proportionally to tokens

Resulting constraints:
- Run vLLM as a server (not a single script)
- Use steady-state load with realistic traffic
- Measure GPU + CPU power at ≥5–10 Hz
- Attribute energy to requests by token counts

### Experimental setup
Document your environment:
- GPU: model (e.g., A100-40GB), SM count, memory size, driver version
- CPU: cores and base/Turbo clocks
- RAM amount
- Software: Python, CUDA, PyTorch, vLLM commit/branch, model name & revision
- OS + power settings (e.g., nvidia-persistence, power mode)

Launch vLLM as usual, for example:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <model_name> \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --tensor-parallel-size 1
```

Enable the performance tweaks you would use in production (chunked prefill, paged attention, etc.), since efficiency depends on them.

### Workload specification
- Prompt length distribution (e.g., mean 256, std 64 tokens)
- Target output length (e.g., max 256, stop on EOS)
- Concurrency / QPS target (e.g., 64 QPS with Poisson arrivals)
- Duration: ~10–15 minutes steady state after warmup

Load generation guidance:
- Use the OpenAI-compatible vLLM endpoint
- Pre-generate a prompt pool and sample randomly
- Record per request: `prompt_tokens`, `completion_tokens`, `latency`, `timestamp`

### Instrumentation: measuring power
**GPU power (preferred)**  
Use NVML (e.g., `pynvml`) or `nvidia-smi` in streaming mode:
- Sample at ~10 Hz (minimum 5 Hz)
- Log: timestamp, GPU power (W), GPU utilization (%), memory used

Example loop:

```python
import time, csv, pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

with open("gpu_power_log.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "power_W", "gpu_util", "mem_used_bytes"])
    start = time.time()
    while time.time() - start < MEASUREMENT_DURATION:
        p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        writer.writerow([time.time(), p, util.gpu, mem.used])
        time.sleep(0.1)
```

**CPU/system power (optional)**  
Use `intel_rapl`/`pcm-power` or a smart plug/PDU for whole-system power. Whole-system power can be attributed to vLLM during the measurement window (minus idle baseline).

### Experimental procedure
1. **Baseline idle**  
   - Start the power logger with the system idle (vLLM off) for ~2–5 minutes.  
   - Compute average idle power `P_idle`.
2. **Warmup**  
   - Launch vLLM.  
   - Send modest load (e.g., 10 QPS) for 2–3 minutes to warm caches/JIT.  
   - Exclude this window from analysis.
3. **Steady state**  
   - Run the target workload for 10–15 minutes.  
   - Log power samples and per-request stats (tokens, timestamps, latency).
4. **Cool down (optional)**  
   - Post-measure idle to confirm `P_idle` consistency.

### Converting power to energy
Given power samples `(t_i, P_i)` over `[T_start, T_end]`:
1. Compute energy via trapezoidal integration over the window.
2. Subtract idle baseline using `P_idle` to get active energy.
3. Convert Joules → kWh as needed.

### Allocating energy to tokens/requests
From request logs over `[T_start, T_end]`:
- Total generated tokens: `T_gen = sum(completion_tokens_r)`
- Optional prefill tokens: `T_prefill = sum(prompt_tokens_r)`

Then:
- Energy per generated token: `e_per_gen_tok = E_active / T_gen` (J/token)
- Energy per request `r`: `E_r = e_per_gen_tok * completion_tokens_r`

Prefill vs decode split:
- Run prefill-only (0 generated tokens) and decode-only (reuse cache) benches.
- Fit `E_active ≈ α * T_prefill + β * T_gen` via linear regression across workloads.

### Carbon and water conversion
Prefer dynamic inputs over static ML-EcoLyzer values:
- Grid carbon intensity (CI) from a regional API
- PUE from your provider or a conservative assumption

```
CO2_g = E_kWh * CI_g_per_kWh * PUE
CO2_per_1k_tokens = CO2_g / (T_gen / 1000)
```

Apply the same pattern for water if you have defensible water-intensity data.

---

## Toolkit Usage
The repository includes simple helpers that map directly to the protocol above.

- Install dependencies: `pip install -e .`
- Log GPU power (e.g., idle baseline): `inference-energy log-power --duration 120 --interval 0.1 --output logs/idle.csv`
- Run a load test against vLLM (prompts file: one prompt per line):  
  `inference-energy load-test --endpoint http://localhost:8000 --model <model> --prompts prompts.txt --duration 600 --concurrency 8 --output logs/requests.csv`

Captured logs can be used to integrate power over time, subtract `P_idle`, and attribute active energy to generated tokens per the protocol above.
