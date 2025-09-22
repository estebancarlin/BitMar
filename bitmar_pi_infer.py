#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BitMar CPU-only inference + telemetry (Raspberry Pi Zero friendly)

Features
- Loads BitMar from Hugging Face (repo: estebancarlin/bitmar-attention-multimodal, revision: epoch-10 by default)
- CPU-only inference with quantized BitNet layers enabled by model.eval()
- Greedy or temperature/top-k/top-p sampling
- Multimodal: accepts optional DiNOv2 768-D vision feature vector (.npy). Defaults to text-only (zero vector).
- Metrics captured and saved:
    * Throughput (tokens/sec)
    * Latency (ms per token & total response latency)
    * Memory footprint (RSS/peak in RAM; on-disk model size)
    * Power (mW) & Energy (mJ) — via INA219 if available, otherwise estimated from CPU% (configurable)
    * Thermal profile (°C) and CPU frequency (MHz)
    * Optional aggregation of CodeCarbon training energy logs if provided

Outputs
- <metrics_dir>/response.txt                  : generated text
- <metrics_dir>/metrics_summary.json          : roll-up summary
- <metrics_dir>/metrics_timeseries.csv        : time series samples during generation
- <metrics_dir>/token_latencies_ms.csv        : per-token latency breakdown
- <metrics_dir>/disk_footprint.json           : on-disk model + snapshot sizes
- (optional) <metrics_dir>/training_energy.json : computed from a CodeCarbon CSV if supplied

Notes
- Designed for Pi Zero (single core) → limits threads, avoids heavy background work.
- If you have a current/voltage sensor (INA219/INA3221) on I2C, enable with --ina219 to get real power readings.
- Otherwise, power is estimated from CPU utilization between an idle and full-power point you can tune with flags.

# 1) Install what you need (examples; adjust to your environment)
pip3 install numpy huggingface_hub transformers psutil
# Install a CPU-only PyTorch wheel that works on your Pi (armv6/armv7); use your preferred source.
# Optional (if you wired a power sensor):
pip3 install adafruit-circuitpython-ina219

# 2) Run text-only (downloads the model once, then caches it)
python3 bitmar_pi_infer.py \
  --prompt "Describe the scene, then continue a short story about it." \
  --max-new-tokens 48 \
  --metrics-dir run_text_only

# 3) If you have a precomputed DiNOv2 768-D feature vector:
python3 bitmar_pi_infer.py \
  --prompt "Describe the image briefly, then continue the story:" \
  --vision-npy my_image_dino_v2_768.npy \
  --max-new-tokens 64 \
  --metrics-dir run_mm

# 4) If you have an INA219 wired on I2C for real power readings:
python3 bitmar_pi_infer.py \
  --prompt "What do you see?" \
  --ina219 \
  --metrics-dir run_power

# 5) For another model repo or revision, use --repo and --revision flags.
python3 bitmar_pi_infer.py \
  --prompt "Describe the scene, then continue a short story about it." \
  --repo estebancarlin/bitmar-no-memory \
  --revision main \
  --metrics-dir bitmar_no_memory_metrics


###########################################################################################
USING IMAGE : 
###########################################################################################


1. Extract vision features from your image

BitMar doesn't take raw images directly; it expects DiNOv2 features (dim=768).
On your laptop, you can use Hugging Face's facebook/dino-v2-base to generate these:


###########################################################################################
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import numpy as np

# Load DiNOv2
processor = AutoImageProcessor.from_pretrained("facebook/dino-v2-base")
model = AutoModel.from_pretrained("facebook/dino-v2-base")

# Load your image
image = Image.open("my_photo.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    # Use CLS token or average pooled embedding (dim=768)
    features = outputs.last_hidden_state[:,0,:].numpy()

np.save("my_image_dino_v2_768.npy", features)  # Save for BitMar
###########################################################################################


This gives you a my_image_dino_v2_768.npy file with shape (1, 768).

2. Run inference with your prompt + vision vector

Now call the inference script with both the text prompt and the vision .npy file:

###########################################################################################
python3 bitmar_pi_infer.py \
  --prompt "Describe this image and continue with a story about it:" \
  --vision-npy my_image_dino_v2_768.npy \
  --max-new-tokens 64 \
  --metrics-dir test_with_image
###########################################################################################

--vision-npy points to your extracted embedding
The script validates the shape (must be 768-D)
If provided, BitMar sets has_vision=True and fuses the visual features with the text prompt during inference
"""
import os
import sys
import time
import json
import math
import csv
import argparse
import pathlib
import importlib.util
import shutil
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Hard runtime constraints for Pi Zero
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import numpy as np  # type: ignore
except Exception as e:
    print("This script requires numpy. Install with: pip install numpy", file=sys.stderr)
    raise

try:
    import torch  # type: ignore
except Exception:
    print("This script requires PyTorch (CPU build). On Raspberry Pi Zero you may need a community wheel.", file=sys.stderr)
    raise

try:
    from huggingface_hub import snapshot_download  # type: ignore
except Exception:
    snapshot_download = None

# Optional sensors
INA_AVAILABLE = False
try:
    # Prefer adafruit library; if not available, will fall back to estimation
    from adafruit_ina219 import INA219  # type: ignore
    import board  # type: ignore
    import busio  # type: ignore
    INA_AVAILABLE = True
except Exception:
    INA_AVAILABLE = False

# Optional: safetensors
SAFE_TENSORS_AVAILABLE = False
try:
    from safetensors.torch import load_file as safe_load_file  # type: ignore
    SAFE_TENSORS_AVAILABLE = True
except Exception:
    SAFE_TENSORS_AVAILABLE = False


# --------------------------- Utilities ---------------------------

def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n)
    while x >= 1024.0 and i < len(units)-1:
        x /= 1024.0
        i += 1
    return f"{x:.1f} {units[i]}"


def dir_size_bytes(path: str) -> int:
    total = 0
    p = pathlib.Path(path)
    if not p.exists():
        return 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except Exception:
                pass
    return total


def read_cpu_temp_c() -> Optional[float]:
    # Prefer sysfs
    candidates = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
    ]
    for c in candidates:
        try:
            with open(c, "r") as fh:
                milli = int(fh.read().strip())
                return milli / 1000.0
        except Exception:
            continue
    # Fallback: vcgencmd (if installed)
    try:
        import subprocess
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        # format: temp=43.8'C
        if "=" in out and "'C" in out:
            val = out.split("=")[1].split("'C")[0]
            return float(val)
    except Exception:
        pass
    return None


def read_cpu_freq_mhz() -> Optional[float]:
    candidates = [
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq",
    ]
    for c in candidates:
        try:
            with open(c, "r") as fh:
                kHz = int(fh.read().strip())
                return kHz / 1000.0
        except Exception:
            continue
    return None


def get_cpu_percent() -> Optional[float]:
    if psutil is None:
        return None
    try:
        # non-blocking; rely on previous call to warm up
        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


def process_memory_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


@dataclass
class PowerReading:
    timestamp: float
    power_mw: Optional[float]
    voltage_v: Optional[float]
    current_ma: Optional[float]


class PowerMonitor:
    """
    Power monitor using INA219 if available, else estimation from CPU%.
    """
    def __init__(self, use_ina: bool, idle_mw: float, full_mw: float):
        self.use_ina = use_ina and INA_AVAILABLE
        self.idle_mw = float(idle_mw)
        self.full_mw = float(full_mw)
        self._ina = None
        self.readings: List[PowerReading] = []
        self.energy_mJ: float = 0.0
        self._last_ts: Optional[float] = None
        if self.use_ina:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self._ina = INA219(i2c)
                # Configure shunt & bus range conservatively
                self._ina.bus_adc_resolution = self._ina.ADCRES_12BIT_32S
                self._ina.shunt_adc_resolution = self._ina.ADCRES_12BIT_32S
                self._ina.bus_voltage_range = self._ina.BUS_RANGE_16V
                print("✅ INA219 power monitor enabled")
            except Exception as e:
                print(f"⚠️  INA219 init failed ({e}); falling back to estimated power")
                self.use_ina = False
                self._ina = None

    def sample(self) -> PowerReading:
        ts = time.perf_counter()
        if self.use_ina and self._ina is not None:
            try:
                voltage = float(self._ina.bus_voltage) + float(self._ina.shunt_voltage) / 1000.0
                current = float(self._ina.current)  # mA
                power_mw = voltage * current
                pr = PowerReading(ts, power_mw, voltage, current)
            except Exception:
                pr = self._estimate_power(ts)
        else:
            pr = self._estimate_power(ts)
        # Integrate energy (trapezoid using last sample if exists)
        if self.readings:
            dt = (pr.timestamp - self.readings[-1].timestamp)  # seconds
            p_prev = self.readings[-1].power_mw or 0.0
            p_now = pr.power_mw or 0.0
            # mW * s = mJ (because W*s = J, mW*s = mJ)
            self.energy_mJ += (p_prev + p_now) * 0.5 * dt
        self.readings.append(pr)
        return pr

    def _estimate_power(self, ts: float) -> PowerReading:
        cpu = get_cpu_percent()
        if cpu is None:
            # Best-effort: assume mid-load
            est = (self.idle_mw + self.full_mw) * 0.5
        else:
            cpu_clamped = max(0.0, min(100.0, cpu))
            est = self.idle_mw + (self.full_mw - self.idle_mw) * (cpu_clamped / 100.0)
        return PowerReading(ts, est, None, None)


# --------------------------- Model loading ---------------------------

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    Returns logits with filtered values set to -inf for softmax.
    """
    top_k = int(top_k)
    top_p = float(top_p)
    if top_k > 0:
        v, _ = torch.topk(logits, top_k)
        min_keep = v[..., -1, None]
        logits = torch.where(logits < min_keep, torch.tensor(float("-inf"), device=logits.device), logits)
    if top_p < 1.0:
        # sort and accumulate softmax probabilities
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        # shift mask right to always keep at least one token
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits[mask] = float("-inf")
        # invert sorting
        unsorted_logits = torch.full_like(sorted_logits, float("-inf"))
        unsorted_logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        logits = unsorted_logits
    return logits


def generate_text(
    model,
    tokenizer,
    prompt: str,
    vision_vec: Optional[np.ndarray],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    stop_at_eos: bool = True,
    verbose: bool = True,
    telemetry_fn=None,
    sample_interval: float = 0.25,
) -> Dict[str, Any]:
    device = torch.device("cpu")
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Determine expected vision dim from model config
    expected_vdim = int(getattr(model.config, "vision_encoder_dim", 768))
    
    # ---- Build vision features as 3D [B, S, D] ----
    # BitMar's modeling_bitmar.create_episode() averages over dim=1 for vision_latent,
    # so vision_features must include a sequence dimension (S). We'll use S=1 here.
    if vision_vec is None:
        v2d = np.zeros((1, expected_vdim), dtype=np.float32)  # [1, D]
    else:
        v2d = np.asarray(vision_vec, dtype=np.float32)
        # Always enforce [1, D] first
        if v2d.ndim == 1:
            v2d = v2d.reshape(1, -1)
        elif v2d.ndim > 2:
            v2d = v2d.reshape(1, -1)
        if v2d.shape[1] != expected_vdim:
            raise ValueError(f"Vision feature dim {v2d.shape[1]} != expected {expected_vdim}")
    # Expand to [B=1, S=1, D]
    v3d = v2d.reshape(1, 1, expected_vdim)
    vision_features = torch.from_numpy(v3d).to(device)

    has_vision = torch.tensor([bool(np.any(v2d))], dtype=torch.bool, device=device)

    # Warm-up CPU utilization reading (non-blocking) for better power estimation
    if psutil is not None:
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    # Per-token latency capture
    token_latencies_ms: List[float] = []
    token_ids: List[int] = []
    t_start = time.perf_counter()

    model.eval()  # ensures quantized inference path in BitNet layers
    # Generation loop
    last_sample_t = time.perf_counter()
    for step in range(max_new_tokens):
        step_t0 = time.perf_counter()

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features,
            labels=None,
            mode="inference",
            has_vision=has_vision,
        )
        logits = out["logits"]
        next_token_logits = logits[:, -1, :]

        if temperature <= 0.0:
            next_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            scaled = next_token_logits / max(1e-8, float(temperature))
            filtered = top_k_top_p_filtering(scaled, top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        # Append
        input_ids = torch.cat([input_ids, next_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)
        token_ids.append(int(next_id.item()))

        # Latency per token
        step_dt_ms = (time.perf_counter() - step_t0) * 1000.0
        token_latencies_ms.append(step_dt_ms)
        # Telemetry per token
        if telemetry_fn is not None:
            now = time.perf_counter()
            if (now - last_sample_t) >= sample_interval:
                telemetry_fn(note=f"tok-{step}", now_ts=now)
                last_sample_t = now

        if stop_at_eos and tokenizer.eos_token_id is not None and int(next_id.item()) == int(tokenizer.eos_token_id):
            break

    total_dt_s = time.perf_counter() - t_start
    gen_ids = input_ids[0].tolist()
    prompt_len = len(enc["input_ids"][0])
    new_ids = gen_ids[prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    metrics = {
        "generated_tokens": len(new_ids),
        "total_time_s": total_dt_s,
        "throughput_tokens_per_s": (len(new_ids) / total_dt_s) if len(new_ids) > 0 and total_dt_s > 0 else 0.0,
        "avg_latency_ms_per_token": (sum(token_latencies_ms) / len(token_latencies_ms)) if token_latencies_ms else None,
        "p50_latency_ms_per_token": float(np.median(token_latencies_ms)) if token_latencies_ms else None,
        "p95_latency_ms_per_token": float(np.percentile(token_latencies_ms, 95)) if token_latencies_ms else None,
        "token_latencies_ms": token_latencies_ms,
    }
    return {"text": text, "metrics": metrics, "token_ids": token_ids}


# --------------------------- Main ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="CPU-only BitMar inference and telemetry (Pi Zero)")
    ap.add_argument("--repo", type=str, default="estebancarlin/bitmar-attention-multimodal", help="Hugging Face repo id")
    ap.add_argument("--revision", type=str, default="epoch-10", help="HF branch/tag/commit")
    ap.add_argument("--hf-cache", type=str, default=None, help="HF cache directory (optional)")
    ap.add_argument("--offline", action="store_true", help="Use local HF cache only (no network)")
    ap.add_argument("--model-dir", type=str, default=None, help="Load model from an existing snapshot directory (skip download)")

    ap.add_argument("--prompt", type=str, default="Describe the image briefly, then continue the story:", help="Input prompt")
    ap.add_argument("--vision-npy", type=str, default=None, help="Optional .npy file with a (768,) or (1,768) DiNOv2 feature vector")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--no-eos-stop", action="store_true", help="Do not stop at EOS token")

    ap.add_argument("--metrics-dir", type=str, default="bitmar_metrics", help="Directory to save outputs")
    ap.add_argument("--ina219", action="store_true", help="Use INA219 sensor for real power readings (requires hardware & library)")
    ap.add_argument("--idle-mw", type=float, default=500.0, help="Estimated idle power (mW) used when INA219 is not available")
    ap.add_argument("--full-mw", type=float, default=1200.0, help="Estimated full-load power (mW) used when INA219 is not available")
    ap.add_argument("--sample-interval", type=float, default=0.25, help="Min seconds between telemetry samples during generation")
    ap.add_argument("--training-emissions-csv", type=str, default=None, help="Optional CodeCarbon CSV to summarize training energy")

    return ap.parse_args()


def summarize_training_energy(csv_path: str) -> Optional[Dict[str, Any]]:
    p = pathlib.Path(csv_path)
    if not p.exists():
        return None
    try:
        import pandas as pd  # optional; but if not installed, simple CSV parse
        df = pd.read_csv(p)
        total_kwh = float(df.get("energy_consumed", df.get("energy_kwh", 0.0)).sum())
        emissions_kg = float(df.get("emissions", df.get("emissions_kg", 0.0)).sum())
        duration_h = float(df.get("duration", df.get("duration_s", 0.0)).sum())
        if "duration" not in df and "duration_s" in df:
            duration_h = duration_h / 3600.0
        return {
            "total_energy_kwh": total_kwh,
            "emissions_kg": emissions_kg,
            "duration_h": duration_h,
        }
    except Exception:
        # Fallback small parser
        total_kwh = 0.0
        emissions_kg = 0.0
        duration_h = 0.0
        with open(p, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ek = row.get("energy_consumed") or row.get("energy_kwh") or "0"
                em = row.get("emissions") or row.get("emissions_kg") or "0"
                du = row.get("duration") or row.get("duration_s") or "0"
                try:
                    total_kwh += float(ek)
                except Exception:
                    pass
                try:
                    emissions_kg += float(em)
                except Exception:
                    pass
                try:
                    d = float(du)
                    if "duration_s" in row:
                        d = d / 3600.0
                    duration_h += d
                except Exception:
                    pass
        return {"total_energy_kwh": total_kwh, "emissions_kg": emissions_kg, "duration_h": duration_h}


def main():
    args = parse_args()
    metrics_dir = pathlib.Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Fix torch for CPU-only and single-thread
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    # Load model directly from Hugging Face
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.repo,
        revision=args.revision,
        cache_dir=args.hf_cache,
        local_files_only=args.offline,
        torch_dtype="float32",
        device_map="cpu",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.repo,
        revision=args.revision,
        cache_dir=args.hf_cache,
        local_files_only=args.offline,
        trust_remote_code=True
    )
    model.eval()
    
    snapshot_dir = model.name_or_path
    snapshot_bytes = dir_size_bytes(snapshot_dir)
    
    print(f"📦 Model loaded from: {snapshot_dir}")

    # Memory before generation
    rss_before = process_memory_rss_mb()

    # Optional vision vector
    vision_vec = None
    if args.vision_npy:
        arr = np.load(args.vision_npy)
        arr = np.asarray(arr, dtype=np.float32)
        
        # Ensure correct shape (1, 768)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(1, -1)
        
        vision_vec = arr

    # Telemetry monitors
    power = PowerMonitor(use_ina=args.ina219, idle_mw=args.idle_mw, full_mw=args.full_mw)

    # Timeseries CSV
    timeseries_path = metrics_dir / "metrics_timeseries.csv"
    ts_writer = None
    ts_fh = open(timeseries_path, "w", newline="")
    ts_writer = csv.writer(ts_fh)
    ts_writer.writerow(["t_s", "dt_s", "cpu_percent", "cpu_temp_c", "cpu_freq_mhz", "rss_mb", "power_mw", "energy_mJ", "note"])

    # Warm up telemetry
    t_prev = time.perf_counter()
    cpu_temp = read_cpu_temp_c()
    cpu_freq = read_cpu_freq_mhz()
    cpu_pct = get_cpu_percent()
    rss_now = process_memory_rss_mb()
    pr = power.sample()
    ts_writer.writerow([t_prev, 0.0, cpu_pct, cpu_temp, cpu_freq, rss_now, pr.power_mw, power.energy_mJ, "start"])

    # ---- Generate ----
        # Define telemetry callback
    def _telemetry_cb(note: str, now_ts: float):
        cpu_temp = read_cpu_temp_c()
        cpu_freq = read_cpu_freq_mhz()
        cpu_pct = get_cpu_percent()
        rss_now = process_memory_rss_mb()
        pr = power.sample()
        nonlocal t_prev
        dt = now_ts - t_prev
        ts_writer.writerow([now_ts, dt, cpu_pct, cpu_temp, cpu_freq, rss_now, pr.power_mw, power.energy_mJ, note])
        t_prev = now_ts

    gen_result = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        vision_vec=vision_vec,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_at_eos=(not args.no_eos_stop),
        verbose=True,
        telemetry_fn=_telemetry_cb,
        sample_interval=args.sample_interval,
    )

    # While generation was running, we measured per token internally.
    # Take a few final telemetry samples (end-of-run)
    for note in ["end-1", "end-2", "end-3"]:
        now = time.perf_counter()
        dt = now - t_prev
        if dt < args.sample_interval:
            time.sleep(args.sample_interval - dt)
            now = time.perf_counter()
            dt = now - t_prev
        cpu_temp = read_cpu_temp_c()
        cpu_freq = read_cpu_freq_mhz()
        cpu_pct = get_cpu_percent()
        rss_now = process_memory_rss_mb()
        pr = power.sample()
        ts_writer.writerow([now, dt, cpu_pct, cpu_temp, cpu_freq, rss_now, pr.power_mw, power.energy_mJ, note])
        t_prev = now

    ts_fh.flush()
    ts_fh.close()

    # Memory after generation + peak
    rss_after = process_memory_rss_mb()
    peak_rss_mb = None
    if psutil is not None:
        try:
            peak_rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
        except Exception:
            pass

    # Disk footprint details
    disk_footprint = {
        "snapshot_dir": snapshot_dir,
        "snapshot_bytes": snapshot_bytes,
        "snapshot_human": human_bytes(snapshot_bytes)#,
    }
    with open(metrics_dir / "disk_footprint.json", "w") as fh:
        json.dump(disk_footprint, fh, indent=2)

    # Save response
    with open(metrics_dir / "response.txt", "w") as fh:
        fh.write(gen_result["text"])

    # Token latency CSV
    token_lat_path = metrics_dir / "token_latencies_ms.csv"
    with open(token_lat_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["token_index", "latency_ms"])
        for i, ms in enumerate(gen_result["metrics"]["token_latencies_ms"] or []):
            w.writerow([i, f"{ms:.3f}"])

    # Optional training energy summary
    training_energy = None
    if args.training_emissions_csv:
        training_energy = summarize_training_energy(args.training_emissions_csv)
        if training_energy:
            with open(metrics_dir / "training_energy.json", "w") as fh:
                json.dump(training_energy, fh, indent=2)

    # Summary
    summary = {
        "repo": args.repo,
        "revision": args.revision,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "generated_text_path": str(metrics_dir / "response.txt"),
        "generated_tokens": gen_result["metrics"]["generated_tokens"],
        "throughput_tokens_per_sec": gen_result["metrics"]["throughput_tokens_per_s"],
        "latency_ms_avg_per_token": gen_result["metrics"]["avg_latency_ms_per_token"],
        "latency_ms_p50_per_token": gen_result["metrics"]["p50_latency_ms_per_token"],
        "latency_ms_p95_per_token": gen_result["metrics"]["p95_latency_ms_per_token"],
        "total_latency_ms_response": gen_result["metrics"]["total_time_s"] * 1000.0,
        "ram_rss_mb_before": rss_before,
        "ram_rss_mb_after": rss_after,
        "ram_peak_mb": peak_rss_mb,
        "disk_snapshot_human": disk_footprint["snapshot_human"],
        "power_measurement": "INA219" if (args.ina219 and INA_AVAILABLE) else "estimated",
        "energy_mJ_deployment": power.energy_mJ,
        "avg_power_mW_deployment": (power.energy_mJ / (gen_result["metrics"]["total_time_s"] if gen_result['metrics']['total_time_s']>0 else 1e-6)) if gen_result['metrics']['total_time_s']>0 else None,
        "thermal_last_C": read_cpu_temp_c(),
        "cpu_freq_last_MHz": read_cpu_freq_mhz(),
        "timeseries_path": str(timeseries_path),
        "token_latencies_path": str(token_lat_path),
        "disk_footprint_path": str(metrics_dir / "disk_footprint.json"),
        "training_energy": training_energy,
    }
    with open(metrics_dir / "metrics_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== BitMar Inference Summary ===")
    print(f"Text saved to:        {summary['generated_text_path']}")
    print(f"Tokens generated:     {summary['generated_tokens']}")
    print(f"Throughput:           {summary['throughput_tokens_per_sec']:.3f} tok/s")
    print(f"Avg latency/token:    {summary['latency_ms_avg_per_token']:.2f} ms")
    print(f"Total latency:        {summary['total_latency_ms_response']:.2f} ms")
    print(f"RAM (RSS) before/after/peak: {summary['ram_rss_mb_before']:.1f} / {summary['ram_rss_mb_after']:.1f} / {summary['ram_peak_mb']:.1f} MB")
    print(f"On-disk snapshot:     {summary['disk_snapshot_human']}")
    print(f"Energy (deployment):  {summary['energy_mJ_deployment']:.1f} mJ (power={summary['avg_power_mW_deployment']:.1f} mW, method={summary['power_measurement']})")
    print(f"Thermal last:         {summary['thermal_last_C']} °C   CPU freq last: {summary['cpu_freq_last_MHz']} MHz")
    print(f"Telemetry timeseries: {summary['timeseries_path']}")
    if training_energy:
        print(f"Training energy:      {training_energy['total_energy_kwh']} kWh, emissions {training_energy['emissions_kg']} kg CO₂e")

if __name__ == "__main__":
    main()
