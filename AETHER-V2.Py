"""
========================================================
  AETHER - AI & Hardware Benchmark Runner
  Open Source Local AI Compute Benchmark
========================================================
Usage:
  python AETHER-V2.py                          Interactive CUI menu
  python AETHER-V2.py --headless               Run once, no menu, using defaults/auto-detect
  python AETHER-V2.py --model llama3:8b        Headless, specific model
  python AETHER-V2.py --backend ollama --model llama3:8b --runs 5 --quiet
  python AETHER-V2.py --difficulty crazy       Headless with specific test difficulty
"""

import time
import platform
import subprocess
import json
import os
import sys
import re
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import argparse

# == Optional dependencies ======================================================
try:
    import psutil  # type: ignore
except ImportError:
    print("[!] psutil not found. Run: pip install psutil")
    sys.exit(1)

try:
    import GPUtil  # type: ignore
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import requests
except ImportError:
    print("[!] requests not found. Run: pip install requests")
    sys.exit(1)

# == Constants & Colors =========================================================
OLLAMA_BASE     = "http://localhost:11434"
LMS_BASE        = "http://localhost:1234"

# Test Difficulties Definition
TEST_DIFFICULTIES = {
    "easy": {
        "name": "Easy (Short Essay)",
        "context": 2048,
        "max_tokens": 400,
        "prompt": (
            "Write a detailed 300-word essay explaining quantum computing to a five-year-old. "
            "Use simple analogies and avoid technical jargon."
        )
    },
    "medium": {
        "name": "Medium (Complex & Moderate)",
        "context": 4096,
        "max_tokens": 800,
        "prompt": (
            "Write a comprehensive architectural analysis comparing microservices and monolithic design patterns. "
            "Discuss scalability, deployment complexity, latency, fault isolation, and state management trade-offs. "
            "Provide concrete examples of when to choose each pattern in modern cloud-native enterprise environments."
        )
    },
    "hard": {
        "name": "Hard (Complex & Long)",
        "context": 8192,
        "max_tokens": 1500,
        "prompt": (
            "Provide an exhaustive technical breakdown of modern Large Language Model training paradigms, "
            "focusing specifically on the transition from standard autoregressive causal decoding to Reinforcement Learning from "
            "Human Feedback (RLHF) and Direct Preference Optimization (DPO). Detail the mathematical underpinnings of the reward model, "
            "PPO clipping mechanisms, KL-divergence penalties, and memory bottlenecks associated with gradient accumulation across "
            "distributed GPU clusters (FSDP and Tensor Parallelism)."
        )
    },
    "crazy": {
        "name": "Crazy (Massive 4k+ Token Prompt)",
        "context": 16384,
        "max_tokens": 2500,
        "prompt": (
            "You are an elite principal systems architect and distributed computing expert. Below is an exhaustive multi-system "
            "specification brief detailing a hypothetical global-scale financial ledger system.\n\n" +
            (
                "System Component Specification Alpha: The ingestion pipeline handles 500,000 requests per second via Apache Kafka clusters "
                "replicated across three cloud regions with strict eventual consistency guarantees. Paxos and Raft consensus algorithms are "
                "utilized for state machine replication across partitioned nodes. Network jitter, Byzantine fault tolerance, partition tolerance, "
                "and cap-theorem limitations must be fully accounted for under extreme network stress scenarios. "
            ) * 15 +  # Padding to push prompt size past 4k+ tokens for high VRAM high-context setups
            "\n\nTask: Analyze the entire specification above. Enumerate every single architectural vulnerability regarding network partitioning, "
            "state divergence, deadlock potential, and leader election latency. Propose a mathematically sound remediation strategy for each point "
            "along with pseudo-code implementation blocks for the consensus validation layer."
        )
    }
}

DEFAULT_DIFFICULTY    = "easy"
INFERENCE_TIMEOUT_SEC = 120
NUM_RUNS              = 3
WARMUP_RUN            = True
OUTPUT_FILE           = "benchmark_result.json"
AETHER_VERSION        = "0.4.0"

# ANSI Colors
C_CYAN    = "\033[96m"
C_GREEN   = "\033[92m"
C_YELLOW  = "\033[93m"
C_RED     = "\033[91m"
C_MAGENTA = "\033[95m"
C_BOLD    = "\033[1m"
C_RESET   = "\033[0m"

BANNER = f"""{C_CYAN}{C_BOLD}
╔══════════════════════════════════════════════════════╗
║         AETHER - AI Compute Benchmark Runner         ║
║      Open Source - Local Inference - No Uploads      ║
╚══════════════════════════════════════════════════════╝{C_RESET}
"""


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def parse_args():
    p = argparse.ArgumentParser(description="AETHER - AI & Hardware Benchmark Runner")
    p.add_argument("--headless", action="store_true",
                    help="Run once without the interactive menu, then exit")
    p.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of benchmark runs")
    p.add_argument("--model", type=str, default=None, help="Model name (skips interactive selection)")
    p.add_argument("--backend", type=str, choices=["ollama", "lmstudio"], default=None,
                    help="Force a specific backend instead of auto-detecting")
    p.add_argument("--difficulty", type=str, choices=list(TEST_DIFFICULTIES.keys()), default=None,
                    help="Select test difficulty (easy, medium, hard, crazy)")
    p.add_argument("--ctx", type=int, default=None, help="Override context window size")
    p.add_argument("--no-warmup", action="store_true", help="Skip warm-up run")
    p.add_argument("--cpu-only", action="store_true", help="Force CPU-only hardware profiling (skip GPU detection)")
    p.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output JSON path")
    p.add_argument("--timeout", type=int, default=INFERENCE_TIMEOUT_SEC,
                    help="Seconds to wait for a benchmark response before timing out")
    p.add_argument("--quiet", action="store_true",
                    help="Headless mode only: suppress the full JSON dump to stdout")
    return p.parse_args()


# ==============================================================================
# HARDWARE DETECTION
# ==============================================================================

def _rocm_version() -> str:
    try:
        out = subprocess.check_output(
            ["rocminfo"], stderr=subprocess.DEVNULL, timeout=10
        ).decode()
        match = re.search(r"ROCm Version:\s*([\d.]+)", out)
        if match:
            return match.group(1)
    except Exception:
        pass

    try:
        with open("/opt/rocm/.info/version") as f:
            return f.read().strip()
    except Exception:
        pass

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
    result = {"gpu_model": "Unknown", "gpu_vram_gb": 0.0, "gpu_driver": "Unknown"}
    dxdiag_xml = os.path.join(tempfile.gettempdir(), f"aether_dxdiag_{os.getpid()}.xml")
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

            candidates = []
            for device in root.findall(".//DisplayDevice"):
                card_name = device.find("CardName")
                chip      = device.find("ChipType")
                dname     = device.find("DeviceName")
                dmem_ded  = device.find("DedicatedMemory")
                dmem_all  = device.find("DisplayMemory")
                dver      = device.find("DriverVersion")
                dev_type  = device.find("DeviceType")

                name = (card_name.text if card_name is not None and card_name.text
                        else chip.text if chip is not None
                        else dname.text if dname is not None else "")

                is_amd = "AMD" in name or "Radeon" in name
                is_virtual = (dev_type is not None and
                              "Display-Only" in (dev_type.text or ""))
                if not is_amd or is_virtual:
                    continue

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
                best = max(candidates, key=lambda x: x["gpu_vram_gb"])
                result.update(best)
                return result
    except Exception:
        if os.path.exists(dxdiag_xml):
            try:
                os.remove(dxdiag_xml)
            except Exception:
                pass

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
    result = {"gpu_model": "Unknown", "gpu_vram_gb": 0.0, "gpu_driver": "Unknown"}
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


def get_system_specs(cpu_only_override=False) -> dict:
    specs = {
        "os":                 platform.system(),
        "os_release":         platform.release(),
        "os_version":         platform.version(),
        "python_version":     platform.python_version(),
        "cpu_model":          _get_cpu_name(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical":  psutil.cpu_count(logical=True),
        "ram_gb":             round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu_model":          "Unknown/CPU-Only",
        "gpu_vram_gb":        0.0,
        "gpu_driver":         "Unknown",
        "gpu_vendor":         "Unknown",
        "cuda_version":       "N/A",
        "rocm_version":       "N/A",
    }

    if cpu_only_override:
        return specs

    nvidia = _detect_nvidia()
    if nvidia:
        specs.update(nvidia)
        return specs

    if specs["os"] == "Windows":
        amd = _detect_amd_windows()
    else:
        amd = _detect_amd_linux()

    if amd["gpu_model"] != "Unknown":
        specs.update(amd)
        specs["gpu_vendor"]   = "AMD"
        specs["rocm_version"] = _rocm_version()

    return specs


# ==============================================================================
# BACKEND DETECTION
# ==============================================================================

def detect_backend() -> tuple[str, str] | tuple[None, None]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if r.status_code == 200:
            return "ollama", OLLAMA_BASE
    except Exception:
        pass

    try:
        r = requests.get(f"{LMS_BASE}/v1/models", timeout=3)
        if r.status_code == 200:
            return "lmstudio", LMS_BASE
    except Exception:
        pass

    return None, None


def list_models(backend: str, base_url: str) -> list[str]:
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


def get_model_info(backend: str, base_url: str, model_name: str, interactive: bool = True) -> dict:
    if backend == "ollama":
        try:
            r = requests.post(f"{base_url}/api/show", json={"name": model_name}, timeout=5)
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
        quant = "Unknown"
        for tag in ["Q4_K_M", "Q4_0", "Q8_0", "Q5_K_M", "Q6_K", "Q2_K", "fp16", "bf16", "IQ4_XS", "IQ3_M"]:
            if tag.lower() in model_name.lower():
                quant = tag
                break
        return {"quantization": quant, "format": "GGUF", "family": "Unknown"}

    return {"quantization": "Unknown", "format": "Unknown", "family": "Unknown"}


# ==============================================================================
# BENCHMARK RUNNER
# ==============================================================================

def run_ollama_benchmark(base_url: str, model_name: str, difficulty_key: str, ctx_override: int | None) -> dict:
    cfg = TEST_DIFFICULTIES[difficulty_key]
    ctx_size = ctx_override if ctx_override else cfg["context"]

    payload = {
        "model":  model_name,
        "prompt": cfg["prompt"],
        "stream": False,
        "options": {"num_ctx": ctx_size}
    }
    start = time.time()
    try:
        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=INFERENCE_TIMEOUT_SEC)
        elapsed = time.time() - start

        if r.status_code != 200:
            return {"status": "Failed", "error": f"HTTP {r.status_code}"}

        data = r.json()
        gen_tokens = data.get("eval_count", 0)
        gen_ns = data.get("eval_duration", 1)
        prompt_tokens = data.get("prompt_eval_count", 0)
        prompt_ns = data.get("prompt_eval_duration", 1)

        return {
            "status": "Success",
            "wall_time_sec": round(elapsed, 2),
            "prompt_tokens": prompt_tokens,
            "generation_tokens": gen_tokens,
            "prompt_tokens_per_sec": round(prompt_tokens / (prompt_ns / 1e9), 2) if prompt_ns > 0 else 0,
            "generation_tokens_per_sec": round(gen_tokens / (gen_ns / 1e9), 2) if gen_ns > 0 else 0,
            "context_window": ctx_size,
        }
    except requests.exceptions.Timeout:
        return {"status": "Failed", "error": "Request timed out (model may be too slow or stuck loading)"}
    except requests.exceptions.ConnectionError:
        return {"status": "Failed", "error": "Could not connect to Ollama"}
    except Exception as e:
        return {"status": "Failed", "error": str(e)}


def run_lmstudio_benchmark(base_url: str, model_name: str, difficulty_key: str, ctx_override: int | None) -> dict:
    cfg = TEST_DIFFICULTIES[difficulty_key]
    ctx_size = ctx_override if ctx_override else cfg["context"]

    payload = {
        "model":      model_name,
        "messages":   [{"role": "user", "content": cfg["prompt"]}],
        "max_tokens": cfg["max_tokens"],
        "stream":     False,
    }
    start = time.time()
    try:
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=INFERENCE_TIMEOUT_SEC)
        elapsed = time.time() - start

        if r.status_code != 200:
            return {"status": "Failed", "error": f"HTTP {r.status_code}"}

        data  = r.json()
        usage = data.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        gen_tokens    = usage.get("completion_tokens", 0)
        gen_tps = round(gen_tokens / elapsed, 2) if elapsed > 0 else 0

        return {
            "status":                    "Success",
            "wall_time_sec":             round(elapsed, 2),
            "prompt_tokens":             prompt_tokens,
            "generation_tokens":         gen_tokens,
            "prompt_tokens_per_sec":     "N/A (LM Studio)",
            "generation_tokens_per_sec": gen_tps,
            "context_window":            ctx_size,
            "note": "LM Studio wall-time derived t/s -- includes prompt processing overhead"
        }
    except requests.exceptions.Timeout:
        return {"status": "Failed", "error": "Request timed out (model may be too slow or stuck loading)"}
    except requests.exceptions.ConnectionError:
        return {"status": "Failed", "error": "Could not connect to LM Studio"}
    except Exception as e:
        return {"status": "Failed", "error": str(e)}


def run_benchmark(backend: str, base_url: str, model_name: str, difficulty_key: str, ctx_override: int | None) -> dict:
    if backend not in ("ollama", "lmstudio"):
        return {"status": "Failed", "error": "Unknown backend"}

    all_runs = []
    total_runs = NUM_RUNS + (1 if WARMUP_RUN else 0)

    for i in range(total_runs):
        label = "Warm-up" if (WARMUP_RUN and i == 0) else f"Run {i if not WARMUP_RUN else i}/{NUM_RUNS}"
        print(f"    {C_CYAN}[{label}]{C_RESET} Running inference...")
        
        if backend == "ollama":
            result = run_ollama_benchmark(base_url, model_name, difficulty_key, ctx_override)
        else:
            result = run_lmstudio_benchmark(base_url, model_name, difficulty_key, ctx_override)

        if WARMUP_RUN and i == 0:
            continue
        if result["status"] == "Success":
            all_runs.append(result)
        else:
            print(f"    {C_RED}[!] Run failed: {result.get('error')}{C_RESET}")

    if not all_runs:
        return {"status": "Failed", "error": "All runs failed"}

    gen_speeds = [r["generation_tokens_per_sec"] for r in all_runs]
    wall_times = [r["wall_time_sec"] for r in all_runs]

    return {
        "status":                        "Success",
        "runs_completed":                len(all_runs),
        "generation_tokens_per_sec":     round(sorted(gen_speeds)[len(gen_speeds) // 2], 2),
        "generation_tokens_per_sec_avg": round(sum(gen_speeds) / len(gen_speeds), 2),
        "generation_tokens_per_sec_min": round(min(gen_speeds), 2),
        "generation_tokens_per_sec_max": round(max(gen_speeds), 2),
        "prompt_tokens_per_sec":         all_runs[0].get("prompt_tokens_per_sec", "N/A"),
        "wall_time_sec_avg":             round(sum(wall_times) / len(wall_times), 2),
        "prompt_tokens":                 all_runs[0]["prompt_tokens"],
        "generation_tokens":             all_runs[0]["generation_tokens"],
        "context_window":                all_runs[0]["context_window"],
        "all_runs":                      all_runs,
    }


def compute_aether_score(hw: dict, results: dict) -> dict:
    if results["status"] != "Success":
        return {"score": 0, "tier": "N/A"}

    gen_tps = results["generation_tokens_per_sec"]
    vram = hw.get("gpu_vram_gb", 0) or 1

    raw_score = gen_tps * 10
    efficiency = gen_tps / vram if vram else 0
    efficiency_bonus = efficiency * 5

    total = round(raw_score + efficiency_bonus, 1)

    if total >= 500:
        tier = "S - Exceptional"
    elif total >= 300:
        tier = "A - High Performance"
    elif total >= 150:
        tier = "B - Solid"
    elif total >= 50:
        tier = "C - Usable"
    else:
        tier = "D - Limited"

    return {
        "score":                total,
        "tier":                 tier,
        "raw_speed_component":  round(raw_score, 1),
        "efficiency_component": round(efficiency_bonus, 1),
    }


def build_payload(backend, chosen_model, model_info, hw, results, aether_score, difficulty_key) -> dict:
    return {
        "aether_version":   AETHER_VERSION,
        "run_id":           f"{platform.node()}-{int(time.time())}",
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "backend":          backend,
        "model":            chosen_model,
        "model_info":       model_info,
        "test_difficulty":  difficulty_key,
        "hardware":         hw,
        "benchmark":        results,
        "aether_score":     aether_score,
        "timeout_sec_used": INFERENCE_TIMEOUT_SEC,
    }


# ==============================================================================
# HEADLESS MODE
# ==============================================================================

def run_headless(args) -> int:
    print(BANNER)
    print(f"{C_CYAN}[Headless Mode]{C_RESET} Running non-interactively\n")

    difficulty_key = args.difficulty or DEFAULT_DIFFICULTY
    print(f"{C_GREEN}[+] Test Difficulty: {TEST_DIFFICULTIES[difficulty_key]['name']}{C_RESET}")

    if args.backend:
        backend = args.backend
        base_url = OLLAMA_BASE if backend == "ollama" else LMS_BASE
        check_url = f"{base_url}/api/tags" if backend == "ollama" else f"{base_url}/v1/models"
        try:
            r = requests.get(check_url, timeout=3)
            if r.status_code != 200:
                return 1
        except Exception:
            return 1
    else:
        backend, base_url = detect_backend()
        if not backend:
            return 1

    print(f"{C_GREEN}[+] Backend: {backend.upper()} at {base_url}{C_RESET}")

    models = list_models(backend, base_url)
    if not models:
        return 1

    if args.model:
        if args.model not in models:
            return 1
        chosen_model = args.model
    elif len(models) == 1:
        chosen_model = models[0]
    else:
        return 1

    print(f"{C_GREEN}[+] Model: {chosen_model}{C_RESET}")

    print(f"{C_CYAN}[+] Profiling hardware...{C_RESET}")
    hw = get_system_specs(cpu_only_override=args.cpu_only)

    print(f"\n{C_CYAN}[+] Running benchmark ({difficulty_key.upper()}) -- {NUM_RUNS} run(s)...{C_RESET}\n")
    results = run_benchmark(backend, base_url, chosen_model, difficulty_key, args.ctx)

    if results["status"] != "Success":
        return 1

    model_info = get_model_info(backend, base_url, chosen_model, interactive=False)
    aether_score = compute_aether_score(hw, results)

    print(f"\n  {C_BOLD}Generation speed (median):{C_RESET} {results['generation_tokens_per_sec']} tok/s")
    print(f"  {C_GREEN}{C_BOLD}AETHER Score: {aether_score['score']} ({aether_score['tier']}){C_RESET}")

    payload = build_payload(backend, chosen_model, model_info, hw, results, aether_score, difficulty_key)

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=4)

    print(f"\n{C_CYAN}[+] Results saved to {args.output}{C_RESET}")
    return 0


# ==============================================================================
# CUI MAIN MENU
# ==============================================================================

def main_menu():
    hw = None
    backend, base_url = detect_backend()
    chosen_model = None
    cpu_only_mode = False
    current_difficulty = DEFAULT_DIFFICULTY

    while True:
        clear_screen()
        print(BANNER)

        print(f"{C_YELLOW}--- Current Configuration ---{C_RESET}")

        if hw is None:
            print(f"  {C_BOLD}Hardware:{C_RESET} [Not scanned yet]")
        else:
            if cpu_only_mode:
                print(f"  {C_BOLD}Hardware:{C_RESET} {hw['cpu_model']} {C_RED}[CPU ONLY MODE]{C_RESET}")
            else:
                gpu_str = f"{hw['gpu_model']} ({hw['gpu_vram_gb']}GB)" if hw['gpu_model'] != "Unknown/CPU-Only" else "None Detected"
                print(f"  {C_BOLD}Hardware:{C_RESET} {hw['cpu_model']} | GPU: {gpu_str}")

        b_str = backend.upper() if backend else f"{C_RED}Not Detected{C_RESET}"
        print(f"  {C_BOLD}Backend:{C_RESET}  {b_str}")

        m_str = chosen_model if chosen_model else f"{C_RED}None Selected{C_RESET}"
        print(f"  {C_BOLD}Model:{C_RESET}    {m_str}")
        
        diff_info = TEST_DIFFICULTIES[current_difficulty]
        print(f"  {C_BOLD}Difficulty:{C_RESET}{C_MAGENTA} {diff_info['name']} (Ctx: {diff_info['context']}){C_RESET}")
        print(f"{C_YELLOW}-----------------------------{C_RESET}\n")

        print(f"{C_CYAN}1.{C_RESET} Run Benchmark")
        print(f"{C_CYAN}2.{C_RESET} Change Test Difficulty (Easy / Medium / Hard / Crazy)")
        print(f"{C_CYAN}3.{C_RESET} Toggle CPU-Only Mode")
        print(f"{C_CYAN}4.{C_RESET} Scan / Re-Scan Hardware")
        print(f"{C_CYAN}5.{C_RESET} Detect / Change Backend")
        print(f"{C_CYAN}6.{C_RESET} Select Model")
        print(f"{C_CYAN}0.{C_RESET} Exit\n")

        choice = input(f"Select an option {C_CYAN}[0-6]{C_RESET}: ").strip()

        if choice == '1':
            if not backend or not base_url:
                input(f"\n{C_RED}[!] No backend detected. Press Enter...{C_RESET}")
                continue
            if not chosen_model:
                input(f"\n{C_RED}[!] No model selected. Press Enter...{C_RESET}")
                continue

            if hw is None:
                hw = get_system_specs(cpu_only_override=cpu_only_mode)

            print(f"\n{C_GREEN}[+] Starting Benchmark ({diff_info['name']}) on {chosen_model}...{C_RESET}\n")

            results = run_benchmark(backend, base_url, chosen_model, current_difficulty, None)

            if results["status"] == "Success":
                print(f"\n  {C_BOLD}Generation speed (median):{C_RESET} {results['generation_tokens_per_sec']} tok/s")
                print(f"  {C_BOLD}Prompt eval speed:{C_RESET}         {results['prompt_tokens_per_sec']} tok/s")
                print(f"  {C_BOLD}Wall time (avg):{C_RESET}           {results['wall_time_sec_avg']}s")

                model_info = get_model_info(backend, base_url, chosen_model)
                aether_score = compute_aether_score(hw, results)

                print(f"\n  {C_GREEN}{C_BOLD}AETHER Score: {aether_score['score']} ({aether_score['tier']}){C_RESET}")

                payload = build_payload(backend, chosen_model, model_info, hw, results, aether_score, current_difficulty)
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(payload, f, indent=4)
                print(f"\n  {C_CYAN}[+] Results saved to {OUTPUT_FILE}{C_RESET}")
            else:
                print(f"\n  {C_RED}[!] Benchmark failed: {results['error']}{C_RESET}")

            input("\nPress Enter to return to menu...")

        elif choice == '2':
            clear_screen()
            print(f"{C_CYAN}--- Select Test Difficulty ---{C_RESET}\n")
            keys = list(TEST_DIFFICULTIES.keys())
            for idx, k in enumerate(keys):
                d = TEST_DIFFICULTIES[k]
                active_marker = " (Active)" if k == current_difficulty else ""
                print(f"  {C_YELLOW}[{idx}]{C_RESET} {d['name']}{active_marker}")
                print(f"      Context: {d['context']} tokens | Max Output: {d['max_tokens']} tokens")
            
            try:
                sel = int(input(f"\nSelect option {C_CYAN}[0-{len(keys)-1}]{C_RESET}: "))
                if 0 <= sel < len(keys):
                    current_difficulty = keys[sel]
            except ValueError:
                pass

        elif choice == '3':
            cpu_only_mode = not cpu_only_mode
            if hw is not None:
                hw = get_system_specs(cpu_only_override=cpu_only_mode)

        elif choice == '4':
            clear_screen()
            hw = get_system_specs(cpu_only_override=cpu_only_mode)
            input("\nPress Enter to return to menu...")

        elif choice == '5':
            backend, base_url = detect_backend()
            chosen_model = None

        elif choice == '6':
            if not backend or not base_url:
                continue
            models = list_models(backend, base_url)
            if not models:
                continue

            clear_screen()
            print(f"{C_CYAN}--- Available Models ---{C_RESET}")
            for i, m in enumerate(models):
                print(f"  {C_YELLOW}[{i}]{C_RESET} {m}")

            try:
                idx = int(input(f"\nSelect model {C_CYAN}[0-{len(models)-1}]{C_RESET}: "))
                if 0 <= idx < len(models):
                    chosen_model = models[idx]
            except ValueError:
                pass

        elif choice == '0':
            clear_screen()
            sys.exit(0)


def main():
    args = parse_args()

    global NUM_RUNS, OUTPUT_FILE, WARMUP_RUN, INFERENCE_TIMEOUT_SEC
    NUM_RUNS = args.runs
    OUTPUT_FILE = args.output
    WARMUP_RUN = not args.no_warmup
    INFERENCE_TIMEOUT_SEC = args.timeout

    headless_triggered = args.headless or args.model or args.backend or args.difficulty

    if headless_triggered:
        exit_code = run_headless(args)
        sys.exit(exit_code)

    try:
        main_menu()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Exiting AETHER...{C_RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
