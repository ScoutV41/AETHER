"""
========================================================
  AETHER - AI & Hardware Benchmark Runner
  Open Source Local AI Compute Benchmark
========================================================

What this script collects and why:
  - CPU model, core count, RAM         -> context for performance scaling
  - GPU model, VRAM, driver version     -> primary performance variable
  - ROCm / CUDA version (if present)   -> affects inference speed significantly
  - Tokens/sec on a standardized prompt -> the actual benchmark metric
  - Quantization level of model tested  -> apples-to-apples comparisons

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
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="AETHER - AI & Hardware Benchmark Runner")
    p.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of benchmark runs")
    p.add_argument("--model", type=str, default=None, help="Model name (skips interactive selection)")
    p.add_argument("--backend", type=str, choices=["ollama", "lmstudio"], default=None)
    p.add_argument("--ctx", type=int, default=CONTEXT_WINDOW, help="Context window size")
    p.add_argument("--no-warmup", action="store_true", help="Skip warm-up run")
    p.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output JSON path")
    p.add_argument("--timeout", type=int, default=INFERENCE_TIMEOUT_SEC,
                    help="Seconds to wait for a benchmark response before timing out (default: 100)")
    return p.parse_args()


# == Optional dependencies ======================================================
try:
    import psutil  # type: ignore
except ImportError:
    print("[!] psutil not found. Run: pip install psutil")
    sys.exit(1)

try:
    import GPUtil # type: ignore
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
BENCHMARK_PROMPT = (
    "Write a detailed 300-word essay explaining quantum computing to a five-year-old. "
    "Use simple analogies and avoid technical jargon."
)
CONTEXT_WINDOW  = 2048
INFERENCE_TIMEOUT_SEC = 100 
NUM_RUNS    = 3 
WARMUP_RUN  = True 
OUTPUT_FILE     = "benchmark_result.json"

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


def get_model_info(backend: str, base_url: str, model_name: str) -> dict:
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
        quant = "Unknown"
        for tag in ["Q4_K_M", "Q4_0", "Q8_0", "Q5_K_M", "Q6_K", "Q2_K", "fp16", "bf16", "IQ4_XS", "IQ3_M"]:
            if tag.lower() in model_name.lower():
                quant = tag
                break
        if quant == "Unknown":
            print(f"\n  [?] Could not detect quantization from model ID: '{model_name}'")
            user_quant = input("      Enter quantization level (or press Enter to skip): ").strip()
            if user_quant:
                quant = user_quant
        return {"quantization": quant, "format": "GGUF", "family": "Unknown"}

    return {"quantization": "Unknown", "format": "Unknown", "family": "Unknown"}


# ==============================================================================
# BENCHMARK RUNNER
# ==============================================================================

def run_ollama_benchmark(base_url: str, model_name: str) -> dict:
    payload = {
        "model":  model_name,
        "prompt": BENCHMARK_PROMPT,
        "stream": False,
        "options": {"num_ctx": CONTEXT_WINDOW}
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
            "context_window": CONTEXT_WINDOW,
        }
    except requests.exceptions.Timeout:
        return {"status": "Failed", "error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return {"status": "Failed", "error": "Could not connect to Ollama"}
    except Exception as e:
        return {"status": "Failed", "error": str(e)}


def run_lmstudio_benchmark(base_url: str, model_name: str) -> dict:
    payload = {
        "model":      model_name,
        "messages":   [{"role": "user", "content": BENCHMARK_PROMPT}],
        "max_tokens": 400,
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
            "context_window":            CONTEXT_WINDOW,
            "note": "LM Studio wall-time derived t/s"
        }
    except requests.exceptions.Timeout:
        return {"status": "Failed", "error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return {"status": "Failed", "error": "Could not connect to LM Studio"}
    except Exception as e:
        return {"status": "Failed", "error": str(e)}


def run_benchmark(backend: str, base_url: str, model_name: str) -> dict:
    runner = run_ollama_benchmark if backend == "ollama" else run_lmstudio_benchmark # type: ignore
    if backend not in ("ollama", "lmstudio"): # type: ignore
        return {"status": "Failed", "error": 'Unknown backend'}
    
    all_runs = []
    total_runs = NUM_RUNS + (1 if WARMUP_RUN else 0)

    for i in range(total_runs):
        label = "Warm-up" if (WARMUP_RUN and i == 0) else f"Run {i if not WARMUP_RUN else i}/{NUM_RUNS}"
        print(f"    {C_CYAN}[{label}]{C_RESET} Running inference...")
        result = runner(base_url, model_name)
        if WARMUP_RUN and i == 0:
            continue 
        if result["status"] == "Success":
            all_runs.append(result)
        else:
            print(f"    {C_RED}[!] Run failed: {result.get('error')}{C_RESET}")

    if not all_runs:
        return {"status": "Failed", "error": "All runs failed"}
    
    gen_speeds = [r["generation_tokens_per_sec"] for r in all_runs] # type: ignore
    wall_times = [r["wall_time_sec"] for r in all_runs] # type: ignore

    return {
        "status":                           "Success",
        "runs_completed":                   len(all_runs),
        "generation_tokens_per_sec":        round(sorted(gen_speeds)[len(gen_speeds)//2], 2),
        "generation_tokens_per_sec_avg":    round(sum(gen_speeds), 2),
        "generation_tokens_per_sec_min":    round(min(gen_speeds), 2),
        "generation_tokens_per_sec_max":    round(max(gen_speeds), 2),
        "prompt_tokens_per_sec":            all_runs[0].get("prompt_tokens_per_sec", "N/A"),
        "wall_time_sec":                    round(sum(wall_times)/len(wall_times), 2),
        "prompt_tokens":                    all_runs[0]["prompt_tokens"],
        "generation_tokens":                all_runs[0]["generation_tokens"],
        'context_window':                   CONTEXT_WINDOW,
        "all_runs":                         all_runs,
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
        "score":                    total,
        "tier":                     tier,
        "raw_speed_component":      round(raw_score, 1),
        "efficiency_component":     round(efficiency_bonus, 1),
    }

# ==============================================================================
# CUI MAIN MENU
# ==============================================================================

def main_menu():
    hw = get_system_specs(cpu_only_override=False)
    backend, base_url = detect_backend()
    chosen_model = None
    cpu_only_mode = False

    while True:
        clear_screen()
        print(BANNER)
        
        # Display Current Status
        print(f"{C_YELLOW}--- Current Configuration ---{C_RESET}")
        
        if cpu_only_mode:
            print(f"  {C_BOLD}Hardware:{C_RESET} {hw['cpu_model']} {C_RED}[CPU ONLY MODE]{C_RESET}")
        else:
            gpu_str = f"{hw['gpu_model']} ({hw['gpu_vram_gb']}GB)" if hw['gpu_model'] != "Unknown/CPU-Only" else "None Detected"
            print(f"  {C_BOLD}Hardware:{C_RESET} {hw['cpu_model']} | GPU: {gpu_str}")
            
        b_str = backend.upper() if backend else f"{C_RED}Not Detected{C_RESET}"
        print(f"  {C_BOLD}Backend:{C_RESET}  {b_str} {f'({base_url})' if base_url else ''}")
        
        m_str = chosen_model if chosen_model else f"{C_RED}None Selected{C_RESET}"
        print(f"  {C_BOLD}Model:{C_RESET}    {m_str}")
        print(f"{C_YELLOW}-----------------------------{C_RESET}\n")

        # Menu Options
        print(f"{C_CYAN}1.{C_RESET} Run Benchmark")
        print(f"{C_CYAN}2.{C_RESET} Toggle CPU-Only Mode")
        print(f"{C_CYAN}3.{C_RESET} Re-Scan Hardware")
        print(f"{C_CYAN}4.{C_RESET} Detect / Change Backend")
        print(f"{C_CYAN}5.{C_RESET} Select Model")
        print(f"{C_CYAN}0.{C_RESET} Exit\n")

        choice = input(f"Select an option {C_CYAN}[0-5]{C_RESET}: ").strip()

        if choice == '1':
            if not backend or not base_url:
                input(f"\n{C_RED}[!] No backend detected. Press Enter to return to menu...{C_RESET}")
                continue
            if not chosen_model:
                input(f"\n{C_RED}[!] No model selected. Press Enter to return to menu...{C_RESET}")
                continue
            
            print(f"\n{C_GREEN}[+] Starting Benchmark on {chosen_model}...{C_RESET}")
            print(f"    Context window: {CONTEXT_WINDOW} tokens")
            print("    This may take a few minutes...\n")
            
            results = run_benchmark(backend, base_url, chosen_model)

            if results["status"] == "Success":
                print(f"\n  {C_BOLD}Generation speed:{C_RESET}  {results['generation_tokens_per_sec']} tok/s")
                print(f"  {C_BOLD}Prompt eval speed:{C_RESET} {results['prompt_tokens_per_sec']} tok/s")
                print(f"  {C_BOLD}Wall time:{C_RESET}         {results['wall_time_sec']}s")
                print(f"  {C_BOLD}Tokens generated:{C_RESET}  {results['generation_tokens']}")
                
                model_info = get_model_info(backend, base_url, chosen_model)
                aether_score = compute_aether_score(hw, results)
                
                print(f"\n  {C_GREEN}{C_BOLD}AETHER Score: {aether_score['score']} ({aether_score['tier']}){C_RESET}")

                # Save Routine
                payload = {
                    "aether_version":  "0.2.1",
                    "run_id":           f"{platform.node()}-{int(time.time())}",
                    "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
                    "backend":         backend,
                    "model":           chosen_model,
                    "model_info":      model_info,
                    "hardware":        hw,
                    "benchmark":       results,
                    "aether_score":    aether_score,
                    "timeout_sec_used": INFERENCE_TIMEOUT_SEC,
                }
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(payload, f, indent=4)
                print(f"\n  {C_CYAN}[+] Results saved to {OUTPUT_FILE}{C_RESET}")
            else:
                print(f"\n  {C_RED}[!] Benchmark failed: {results['error']}{C_RESET}")
            
            input("\nPress Enter to return to menu...")

        elif choice == '2':
            cpu_only_mode = not cpu_only_mode
            hw = get_system_specs(cpu_only_override=cpu_only_mode)

        elif choice == '3':
            clear_screen()
            print(f"{C_CYAN}Re-scanning hardware...{C_RESET}\n")
            hw = get_system_specs(cpu_only_override=cpu_only_mode)
            input("\nPress Enter to return to menu...")

        elif choice == '4':
            backend, base_url = detect_backend()
            chosen_model = None 
            if backend:
                input(f"\n{C_GREEN}[+] Detected {backend.upper()}{C_RESET}. Press Enter to continue...")
            else:
                input(f"\n{C_RED}[!] No backend found.{C_RESET} Press Enter to continue...")

        elif choice == '5':
            if not backend or not base_url:
                input(f"\n{C_RED}[!] Detect a backend first.{C_RESET} Press Enter...")
                continue
            
            models = list_models(backend, base_url)
            if not models:
                input(f"\n{C_RED}[!] No models found in backend.{C_RESET} Press Enter...")
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
    global NUM_RUNS, CONTEXT_WINDOW, OUTPUT_FILE, WARMUP_RUN, INFERENCE_TIMEOUT_SEC
    NUM_RUNS = args.runs
    CONTEXT_WINDOW = args.ctx
    OUTPUT_FILE = args.output
    WARMUP_RUN = not args.no_warmup
    INFERENCE_TIMEOUT_SEC = args.timeout

    # If run with command line args, we bypass the menu for scripting automation
    if args.model or args.backend:
        print("[!] Running in headless CLI mode due to provided arguments...")
        # ... You can re-implement the headless linear flow here if needed
        # For now, it will just jump to the menu unless you strictly want to bypass it
        print("Headless mode logic goes here (omitted for CUI focus).")
        sys.exit(0)

    try:
        main_menu()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Exiting AETHER...{C_RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main()
