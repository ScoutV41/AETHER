"""
========================================================
  AETHER - AI & Hardware Benchmark Runner
  Open Source Local AI Compute Benchmark
========================================================

What this script collects and why:
  - CPU model, core count, RAM         → context for performance scaling
  - GPU model, VRAM, driver version     → primary performance variable
  - ROCm / CUDA version (if present)   → affects inference speed significantly
  - Tokens/sec on a standardized prompt → the actual benchmark metric
  - Quantization level of model tested  → apples-to-apples comparisons

Nothing is uploaded automatically. Results are saved locally to
benchmark_result.json. Share them however you'd like.
"""

import time
import platform
import subprocess
import json
import os
import sys
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    import psutil
except ImportError:
    print("[!] psutil not found. Run: pip install psutil")
    sys.exit(1)

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import requests
except ImportError:
    print("[!] requests not found. Run: pip install requests")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────
OLLAMA_BASE     = "http://localhost:11434"
LMS_BASE        = "http://localhost:1234"   # LM Studio default
BENCHMARK_PROMPT = (
    "Write a detailed 300-word essay explaining quantum computing to a five-year-old. "
    "Use simple analogies and avoid technical jargon."
)
CONTEXT_WINDOW  = 2048
OUTPUT_FILE     = "benchmark_result.json"

BANNER = """
╔══════════════════════════════════════════════════════╗
║         AETHER - AI Compute Benchmark Runner         ║
║      Open Source · Local Inference · No Uploads      ║
╚══════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════════════════════════════════════
# HARDWARE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _rocm_version() -> str:
    """Try to detect installed ROCm version."""
    # Method 1: rocminfo
    try:
        out = subprocess.check_output(
            ["rocminfo"], stderr=subprocess.DEVNULL, timeout=10
        ).decode()
        match = re.search(r"ROCm Version:\s*([\d.]+)", out)
        if match:
            return match.group(1)
    except Exception:
        pass

    # Method 2: /opt/rocm version file
    try:
        with open("/opt/rocm/.info/version") as f:
            return f.read().strip()
    except Exception:
        pass

    # Method 3: rocm-smi
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--version"], stderr=subprocess.DEVNULL, timeout=5
        ).decode()
        match = re.search(r"([\d.]+)", out)
        if match:
            return match.group(1)
    except Exception:
        pass

    return "Not detected"


def _cuda_version() -> str:
    """Try to detect installed CUDA version."""
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.DEVNULL
        ).decode()
        match = re.search(r"release ([\d.]+)", out)
        if match:
            return match.group(1)
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.DEVNULL
        ).decode()
        match = re.search(r"CUDA Version: ([\d.]+)", out)
        if match:
            return match.group(1)
    except Exception:
        pass

    return "Not detected"


def _get_cpu_name() -> str:
    """Get human-readable CPU name instead of generic Family/Model/Stepping string."""
    system = platform.system()

    if system == "Windows":
        try:
            out = subprocess.check_output(
                "wmic cpu get name /format:value",
                shell=True, stderr=subprocess.DEVNULL
            ).decode()
            for line in out.splitlines():
                if line.startswith("Name=") and line.strip() != "Name=":
                    return line.split("=", 1)[1].strip()
        except Exception:
            pass

    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    elif system == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            if out:
                return out
        except Exception:
            pass

    return platform.processor() or "Unknown"


def _detect_nvidia() -> dict | None:
    """Returns GPU info dict if an NVIDIA GPU is found via GPUtil."""
    if not GPUTIL_AVAILABLE:
        return None
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            return {
                "gpu_model":      g.name,
                "gpu_vram_gb":    round(g.memoryTotal / 1024, 2),
                "gpu_driver":     g.driver,
                "gpu_vendor":     "NVIDIA",
                "cuda_version":   _cuda_version(),
                "rocm_version":   "N/A",
            }
    except Exception:
        pass
    return None


def _detect_amd_windows() -> dict:
    """AMD GPU detection on Windows using dxdiag XML with wmic fallback.
    
    Selects the discrete GPU by picking the AMD device with the most VRAM,
    which avoids grabbing the iGPU on systems with both integrated and discrete AMD graphics.
    """
    result = {"gpu_model": "Unknown", "gpu_vram_gb": 0.0, "gpu_driver": "Unknown"}

    # Primary: dxdiag XML (most reliable for VRAM)
    dxdiag_xml = "dxdiag_temp.xml"
    try:
        subprocess.run(
            f"dxdiag /x {dxdiag_xml}",
            shell=True, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30
        )
        if os.path.exists(dxdiag_xml):
            tree = ET.parse(dxdiag_xml)
            root = tree.getroot()

            # Collect ALL AMD devices, then pick the one with the most VRAM
            # This avoids grabbing the iGPU on systems with integrated + discrete AMD
            candidates = []
            for device in root.findall(".//DisplayDevice"):
                card_name = device.find("CardName")
                chip      = device.find("ChipType")
                dname     = device.find("DeviceName")
                dmem_ded  = device.find("DedicatedMemory")  # actual VRAM, not shared
                dmem_all  = device.find("DisplayMemory")    # fallback (includes shared)
                dver      = device.find("DriverVersion")
                dev_type  = device.find("DeviceType")

                # CardName is the human-readable name (e.g. "AMD Radeon RX 9070 XT")
                # ChipType is a generic hex placeholder — don't use it for filtering
                name = (card_name.text if card_name is not None and card_name.text
                        else chip.text if chip is not None
                        else dname.text if dname is not None else "")

                # Skip non-AMD and display-only virtual adapters (e.g. LuminonCore IDDCX)
                is_amd = "AMD" in name or "Radeon" in name
                is_virtual = (dev_type is not None and
                              "Display-Only" in (dev_type.text or ""))
                if not is_amd or is_virtual:
                    continue

                # Use DedicatedMemory for real VRAM — avoids shared RAM inflating iGPU values
                vram_gb = 0.0
                for mem_field in [dmem_ded, dmem_all]:
                    if mem_field is not None and mem_field.text:
                        try:
                            mb = int(mem_field.text.split()[0].replace(",", ""))
                            vram_gb = round(mb / 1024, 2)
                            break
                        except ValueError:
                            pass

                driver = dver.text if dver is not None else "Unknown"
                candidates.append({
                    "gpu_model":   name,
                    "gpu_vram_gb": vram_gb,
                    "gpu_driver":  driver,
                })

            os.remove(dxdiag_xml)

            if candidates:
                # Pick discrete GPU by highest dedicated VRAM
                # DedicatedMemory excludes shared system RAM so iGPUs naturally rank lower
                best = max(candidates, key=lambda x: x["gpu_vram_gb"])
                result.update(best)

                if len(candidates) > 1:
                    skipped = [c["gpu_model"] for c in candidates if c != best]
                    print(f"  [i] {len(candidates)} AMD devices found, selected: {best['gpu_model']} ({best['gpu_vram_gb']} GB)")
                    print(f"  [i] Skipped: {', '.join(skipped)}")
                return result

    except Exception:
        if os.path.exists(dxdiag_xml):
            try:
                os.remove(dxdiag_xml)
            except Exception:
                pass

    # Fallback: wmic — picks device with most AdapterRAM
    try:
        out = subprocess.check_output(
            "wmic path win32_VideoController get Name,DriverVersion,AdapterRAM /format:csv",
            shell=True
        ).decode()
        candidates = []
        for line in out.splitlines():
            if "AMD" in line or "Radeon" in line:
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        vram_gb = round(int(parts[1].strip()) / (1024 ** 3), 2)
                    except ValueError:
                        vram_gb = 0.0
                    candidates.append({
                        "gpu_model":   parts[3].strip(),
                        "gpu_driver":  parts[2].strip(),
                        "gpu_vram_gb": vram_gb,
                    })
        if candidates:
            best = max(candidates, key=lambda x: x["gpu_vram_gb"])
            result.update(best)
    except Exception:
        pass

    return result


def _detect_amd_linux() -> dict:
    """AMD GPU detection on Linux using rocm-smi with lspci fallback."""
    result = {"gpu_model": "Unknown", "gpu_vram_gb": 0.0, "gpu_driver": "Unknown"}

    # Primary: rocm-smi JSON
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.DEVNULL, timeout=10
        ).decode()
        data = json.loads(out)
        card = list(data.keys())[0] if data else None
        if card:
            result["gpu_model"] = data[card].get("Card series", "AMD ROCm GPU")
            vram_bytes = data[card].get("VRAM Total Memory (B)", 0)
            if vram_bytes:
                result["gpu_vram_gb"] = round(int(vram_bytes) / (1024 ** 3), 2)
            return result
    except Exception:
        pass

    # Fallback: lspci
    try:
        out = subprocess.check_output(
            "lspci | grep -iE 'vga|3d' | grep -iE 'AMD|ATI'",
            shell=True
        ).decode().strip()
        if out:
            result["gpu_model"] = out.split(":")[-1].strip()
    except Exception:
        pass

    return result


def get_system_specs() -> dict:
    """Gather full hardware profile with vendor-specific GPU detection."""
    print("  Profiling CPU...")
    specs = {
        "os":                platform.system(),
        "os_release":        platform.release(),
        "os_version":        platform.version(),
        "python_version":    platform.python_version(),
        "cpu_model":         _get_cpu_name(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical":  psutil.cpu_count(logical=True),
        "ram_gb":            round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu_model":         "Unknown/CPU-Only",
        "gpu_vram_gb":       0.0,
        "gpu_driver":        "Unknown",
        "gpu_vendor":        "Unknown",
        "cuda_version":      "N/A",
        "rocm_version":      "N/A",
    }

    print("  Profiling GPU...")

    # Try NVIDIA first
    nvidia = _detect_nvidia()
    if nvidia:
        specs.update(nvidia)
        print(f"  NVIDIA GPU detected: {specs['gpu_model']}")
        return specs

    # Try AMD
    if specs["os"] == "Windows":
        amd = _detect_amd_windows()
    else:
        amd = _detect_amd_linux()

    if amd["gpu_model"] != "Unknown":
        specs.update(amd)
        specs["gpu_vendor"]   = "AMD"
        specs["rocm_version"] = _rocm_version()
        print(f"  AMD GPU detected: {specs['gpu_model']}")
    else:
        print("  No discrete GPU detected — CPU-only mode.")

    return specs


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_backend() -> tuple[str, str] | tuple[None, None]:
    """
    Auto-detects whether Ollama or LM Studio is running.
    Returns (backend_name, base_url) or (None, None).
    """
    # Check Ollama
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if r.status_code == 200:
            return "ollama", OLLAMA_BASE
    except Exception:
        pass

    # Check LM Studio
    try:
        r = requests.get(f"{LMS_BASE}/v1/models", timeout=3)
        if r.status_code == 200:
            return "lmstudio", LMS_BASE
    except Exception:
        pass

    return None, None


def list_models(backend: str, base_url: str) -> list[str]:
    """Returns available model names for the detected backend."""
    try:
        if backend == "ollama":
            r = requests.get(f"{base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]

        elif backend == "lmstudio":
            r = requests.get(f"{base_url}/v1/models", timeout=5)
            if r.status_code == 200:
                return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        pass
    return []


# ══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_model_info(backend: str, base_url: str, model_name: str) -> dict:
    """Fetch quantization and format info for the selected model."""
    if backend == "ollama":
        try:
            r = requests.post(
                f"{base_url}/api/show",
                json={"name": model_name}, timeout=5
            )
            if r.status_code == 200:
                details = r.json().get("details", {})
                return {
                    "quantization": details.get("quantization_level", "Unknown"),
                    "format":       details.get("format", "Unknown"),
                    "family":       details.get("family", "Unknown"),
                }
        except Exception:
            pass

    elif backend == "lmstudio":
        # LM Studio doesn't expose quant via API — parse from model ID first
        quant = "Unknown"
        for tag in ["Q4_K_M", "Q4_0", "Q8_0", "Q5_K_M", "Q6_K", "Q2_K", "fp16", "bf16", "IQ4_XS", "IQ3_M"]:
            if tag.lower() in model_name.lower():
                quant = tag
                break
        # If still unknown, ask the user — important for accurate benchmark comparisons
        if quant == "Unknown":
            print(f"\n  [?] Could not detect quantization from model ID: '{model_name}'")
            print("      Common values: Q4_K_M, Q4_0, Q8_0, Q5_K_M, fp16, bf16")
            user_quant = input("      Enter quantization level (or press Enter to skip): ").strip()
            if user_quant:
                quant = user_quant
        return {"quantization": quant, "format": "GGUF", "family": "Unknown"}

    return {"quantization": "Unknown", "format": "Unknown", "family": "Unknown"}


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_ollama_benchmark(base_url: str, model_name: str) -> dict:
    """Run inference benchmark against Ollama backend."""
    payload = {
        "model":  model_name,
        "prompt": BENCHMARK_PROMPT,
        "stream": False,
        "options": {"num_ctx": CONTEXT_WINDOW}
    }
    start = time.time()
    try:
        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=180)
        elapsed = time.time() - start

        if r.status_code != 200:
            return {"status": "Failed", "error": f"HTTP {r.status_code}"}

        data = r.json()

        # Ollama field names — eval = generation, prompt_eval = prompt processing
        gen_tokens   = data.get("eval_count", 0)
        gen_ns       = data.get("eval_duration", 1)
        prompt_tokens = data.get("prompt_eval_count", 0)
        prompt_ns    = data.get("prompt_eval_duration", 1)

        return {
            "status":                  "Success",
            "wall_time_sec":           round(elapsed, 2),
            "prompt_tokens":           prompt_tokens,
            "generation_tokens":       gen_tokens,
            "prompt_tokens_per_sec":   round(prompt_tokens / (prompt_ns / 1e9), 2) if prompt_ns > 0 else 0,
            "generation_tokens_per_sec": round(gen_tokens / (gen_ns / 1e9), 2) if gen_ns > 0 else 0,
            "context_window":          CONTEXT_WINDOW,
        }
    except requests.exceptions.ConnectionError:
        return {"status": "Failed", "error": "Could not connect to Ollama"}
    except Exception as e:
        return {"status": "Failed", "error": str(e)}


def run_lmstudio_benchmark(base_url: str, model_name: str) -> dict:
    """Run inference benchmark against LM Studio (OpenAI-compatible) backend."""
    payload = {
        "model":      model_name,
        "messages":   [{"role": "user", "content": BENCHMARK_PROMPT}],
        "max_tokens": 400,
        "stream":     False,
    }
    start = time.time()
    try:
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=180)
        elapsed = time.time() - start

        if r.status_code != 200:
            return {"status": "Failed", "error": f"HTTP {r.status_code}"}

        data  = r.json()
        usage = data.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        gen_tokens    = usage.get("completion_tokens", 0)

        # LM Studio doesn't return per-component durations — derive gen t/s from wall time
        gen_tps = round(gen_tokens / elapsed, 2) if elapsed > 0 else 0

        return {
            "status":                    "Success",
            "wall_time_sec":             round(elapsed, 2),
            "prompt_tokens":             prompt_tokens,
            "generation_tokens":         gen_tokens,
            "prompt_tokens_per_sec":     "N/A (LM Studio)",
            "generation_tokens_per_sec": gen_tps,
            "context_window":            CONTEXT_WINDOW,
            "note": "LM Studio wall-time derived t/s — includes prompt processing overhead"
        }
    except requests.exceptions.ConnectionError:
        return {"status": "Failed", "error": "Could not connect to LM Studio"}
    except Exception as e:
        return {"status": "Failed", "error": str(e)}


def run_benchmark(backend: str, base_url: str, model_name: str) -> dict:
    if backend == "ollama":
        return run_ollama_benchmark(base_url, model_name)
    elif backend == "lmstudio":
        return run_lmstudio_benchmark(base_url, model_name)
    return {"status": "Failed", "error": "Unknown backend"}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(BANNER)

    # ── Step 1: Hardware ──────────────────────────────────────────────────────
    print("[1/4] Profiling hardware...\n")
    hw = get_system_specs()
    print(f"\n  OS:     {hw['os']} {hw['os_release']}")
    print(f"  CPU:    {hw['cpu_model']} ({hw['cpu_cores_physical']}P / {hw['cpu_cores_logical']}L cores)")
    print(f"  RAM:    {hw['ram_gb']} GB")
    print(f"  GPU:    {hw['gpu_model']} | {hw['gpu_vram_gb']} GB VRAM | Driver {hw['gpu_driver']}")
    if hw['rocm_version'] != "N/A":
        print(f"  ROCm:   {hw['rocm_version']}")
    if hw['cuda_version'] != "N/A":
        print(f"  CUDA:   {hw['cuda_version']}")

    # ── Step 2: Backend ───────────────────────────────────────────────────────
    print("\n[2/4] Detecting inference backend...\n")
    backend, base_url = detect_backend()

    if backend is None:
        print("  [!] No inference backend detected.")
        print("      Start Ollama (ollama serve) or LM Studio before running.")
        sys.exit(1)

    print(f"  Backend: {backend.upper()} at {base_url}")

    models = list_models(backend, base_url)
    if not models:
        print("  [!] No models found. Load a model in your inference backend first.")
        sys.exit(1)

    print(f"  Available models ({len(models)}):")
    for i, m in enumerate(models):
        print(f"    [{i}] {m}")

    # Model selection
    if len(models) == 1:
        chosen = models[0]
        print(f"\n  Auto-selected: {chosen}")
    else:
        while True:
            try:
                idx = int(input(f"\n  Select model [0-{len(models)-1}]: "))
                if 0 <= idx < len(models):
                    chosen = models[idx]
                    break
                print("  Invalid selection.")
            except ValueError:
                print("  Enter a number.")

    model_info = get_model_info(backend, base_url, chosen)
    print(f"  Quantization: {model_info['quantization']} | Format: {model_info['format']}")

    # ── Step 3: Benchmark ─────────────────────────────────────────────────────
    print(f"\n[3/4] Running inference benchmark on '{chosen}'...")
    print(f"      Context window: {CONTEXT_WINDOW} tokens\n")
    print("      This may take a few minutes depending on your hardware...\n")

    results = run_benchmark(backend, base_url, chosen)

    if results["status"] == "Success":
        print(f"  Generation speed:  {results['generation_tokens_per_sec']} tok/s")
        print(f"  Prompt eval speed: {results['prompt_tokens_per_sec']} tok/s")
        print(f"  Wall time:         {results['wall_time_sec']}s")
        print(f"  Tokens generated:  {results['generation_tokens']}")
    else:
        print(f"  [!] Benchmark failed: {results['error']}")

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving results to {OUTPUT_FILE}...")

    payload = {
        "aether_version":  "0.1.0",
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "backend":         backend,
        "model":           chosen,
        "model_info":      model_info,
        "hardware":        hw,
        "benchmark":       results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=4)

    print(f"\n  Done! Results saved to {OUTPUT_FILE}")
    print("\n  ─────────────────────────────────────────────────────")
    print("  Share your results to contribute to the database!")
    print("  Discord: [your server link here]")
    print("  GitHub:  [your repo link here]")
    print("  ─────────────────────────────────────────────────────\n")

    print(json.dumps(payload, indent=4))


if __name__ == "__main__":
    main()
