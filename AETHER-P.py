"""
========================================================
  AETHER Portable -  Hardware Benchmark CLI
  Fully Configurable Non-AI Benchmark Suite
========================================================
"""

import time
import platform
import subprocess
import json
import os
import sys
import math
import concurrent.futures
from datetime import datetime, timezone

# == Optional Dependencies ======================================================
try:
    import psutil
except ImportError:
    print("[!] psutil not found. Please run: pip install psutil")
    sys.exit(1)

NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

# == Formatting & Constants ====================================================
OUTPUT_FILE = "aether_benchmark_result.json"

C_CYAN    = "\033[96m"
C_GREEN   = "\033[92m"
C_YELLOW  = "\033[93m"
C_RED     = "\033[91m"
C_MAGENTA = "\033[95m"
C_BOLD    = "\033[1m"
C_RESET   = "\033[0m"

BANNER = f"""{C_CYAN}{C_BOLD}
╔══════════════════════════════════════════════════════╗
║             AETHER -  Benchmark CLI                  ║
║       CPU • RAM • GPU Compute Benchmark Suite        ║
╚══════════════════════════════════════════════════════╝{C_RESET}
"""

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ==============================================================================
# HARDWARE PROFILE
# ==============================================================================

def get_cpu_name() -> str:
    system = platform.system()
    if system == "Windows":
        try:
            out = subprocess.check_output("wmic cpu get name /format:value", shell=True, stderr=subprocess.DEVNULL).decode()
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
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"

def detect_hardware() -> dict:
    specs = {
        "os":                 f"{platform.system()} {platform.release()}",
        "cpu_model":          get_cpu_name(),
        "cpu_cores_logical":  psutil.cpu_count(logical=True) or 1,
        "ram_gb":             round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu_model":          "CPU-Only / Integrated",
        "cuda_available":     False
    }
    if TORCH_AVAILABLE and torch.cuda.is_available():
        specs["gpu_model"] = torch.cuda.get_device_name(0)
        specs["cuda_available"] = True
    return specs

# ==============================================================================
# DEFAULT SETTINGS
# ==============================================================================

class BenchmarkSettings:
    def __init__(self, hw_threads):
        self.passes = 3
        self.detailed_metrics = True
        
        # CPU Settings
        self.cpu_threads = hw_threads
        self.cpu_iterations = 1_500_000  # Default workload
        
        # RAM Settings
        self.ram_size_mb = 512
        
        # GPU Settings
        self.gpu_matrix_size = 4096

    def set_duration_preset(self, preset: str):
        if preset == "Quick":
            self.passes = 1
            self.cpu_iterations = 750_000
            self.ram_size_mb = 256
            self.gpu_matrix_size = 2048
        elif preset == "Standard":
            self.passes = 3
            self.cpu_iterations = 1_500_000
            self.ram_size_mb = 512
            self.gpu_matrix_size = 4096
        elif preset == "Stress":
            self.passes = 5
            self.cpu_iterations = 5_000_000
            self.ram_size_mb = 1024
            self.gpu_matrix_size = 8192

# ==============================================================================
# BENCHMARK ENGINE
# ==============================================================================

def _cpu_worker(iterations):
    acc = 0.0
    for i in range(1, iterations):
        acc += math.sin(i) * math.cos(i) + math.sqrt(i)
    return acc

def run_cpu_test(settings: BenchmarkSettings) -> dict:
    times = []
    total_ops = settings.cpu_threads * settings.cpu_iterations

    for run_idx in range(settings.passes):
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=settings.cpu_threads) as executor:
            futures = [executor.submit(_cpu_worker, settings.cpu_iterations) for _ in range(settings.cpu_threads)]
            concurrent.futures.wait(futures)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    ops_per_sec = total_ops / avg_time
    score = round(ops_per_sec / 100_000, 2)

    return {
        "status": "Success",
        "score": score,
        "avg_time_sec": round(avg_time, 3),
        "min_time_sec": round(min(times), 3),
        "max_time_sec": round(max(times), 3),
        "ops_per_sec": round(ops_per_sec, 2),
        "threads_used": settings.cpu_threads
    }

def run_ram_test(settings: BenchmarkSettings) -> dict:
    num_elements = (settings.ram_size_mb * 1024 * 1024) // 8
    times = []

    for _ in range(settings.passes):
        start = time.time()
        if NUMPY_AVAILABLE:
            arr1 = np.ones(num_elements, dtype=np.float64)
            arr2 = arr1 * 2.5
            _ = arr1 + arr2
        else:
            data = [1.0] * (num_elements // 4)
            _ = [x * 2.5 for x in data]
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    bandwidth_gb_s = round((settings.ram_size_mb / 1024) / avg_time, 2)
    score = round(bandwidth_gb_s * 20, 2)

    return {
        "status": "Success",
        "score": score,
        "bandwidth_gb_s": bandwidth_gb_s,
        "avg_time_sec": round(avg_time, 3),
        "min_time_sec": round(min(times), 3),
        "max_time_sec": round(max(times), 3),
        "size_mb": settings.ram_size_mb
    }

def run_gpu_test(settings: BenchmarkSettings) -> dict:
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {"status": "Skipped", "reason": "PyTorch with CUDA support not found."}

    try:
        device = torch.device("cuda")
        N = settings.gpu_matrix_size
        x = torch.randn(N, N, device=device)
        y = torch.randn(N, N, device=device)

        # Warmup pass
        _ = torch.matmul(x, y)
        torch.cuda.synchronize()

        times = []
        for _ in range(settings.passes):
            start = time.time()
            for _ in range(5):
                _ = torch.matmul(x, y)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed / 5)  # Per-multiplication average

        avg_time = sum(times) / len(times)
        tflops = round((2 * (N ** 3)) / (avg_time * 1e12), 2)
        score = round(tflops * 100, 2)

        return {
            "status": "Success",
            "score": score,
            "tflops": tflops,
            "avg_time_sec": round(avg_time, 4),
            "min_time_sec": round(min(times), 4),
            "max_time_sec": round(max(times), 4),
            "matrix_size": f"{N}x{N}"
        }
    except Exception as e:
        return {"status": "Failed", "error": str(e)}

# ==============================================================================
# MENU CONTROLLERS
# ==============================================================================

def display_header(hw):
    clear_screen()
    print(BANNER)
    print(f"  {C_BOLD}System Specs:{C_RESET}")
    print(f"    • CPU : {hw['cpu_model']} ({hw['cpu_cores_logical']} Logical Threads)")
    print(f"    • RAM : {hw['ram_gb']} GB")
    print(f"    • GPU : {hw['gpu_model']}")
    print(f"{C_CYAN}──────────────────────────────────────────────────────{C_RESET}\n")

def settings_menu(settings: BenchmarkSettings, hw: dict):
    while True:
        display_header(hw)
        print(f"{C_YELLOW}{C_BOLD}[ SETTINGS & METRICS MENU ]{C_RESET}\n")
        print(f"  {C_CYAN}1.{C_RESET} Preset Profiles        : Choose Quick, Standard, or Stress")
        print(f"  {C_CYAN}2.{C_RESET} Test Runs / Passes     : {settings.passes} run(s)")
        print(f"  {C_CYAN}3.{C_RESET} CPU Threads            : {settings.cpu_threads} of {hw['cpu_cores_logical']} Cores")
        print(f"  {C_CYAN}4.{C_RESET} CPU Iteration Load     : {settings.cpu_iterations:,} ops/thread")
        print(f"  {C_CYAN}5.{C_RESET} RAM Allocation Size    : {settings.ram_size_mb} MB")
        print(f"  {C_CYAN}6.{C_RESET} GPU Matrix Resolution  : {settings.gpu_matrix_size} x {settings.gpu_matrix_size}")
        print(f"  {C_CYAN}7.{C_RESET} Toggle Detailed Metrics: {'ON' if settings.detailed_metrics else 'OFF'}")
        print(f"\n  {C_CYAN}0.{C_RESET} Back to Main Menu\n")

        choice = input(f"Select option {C_CYAN}[0-7]{C_RESET}: ").strip()

        if choice == '1':
            print("\nChoose Preset Profile:")
            print(" 1) Quick Test (Fast iteration, minimal runs)")
            print(" 2) Standard Test (Balanced performance test)")
            print(" 3) Stress Test (High memory & heavy iteration stress)")
            preset_choice = input("Select [1-3]: ").strip()
            if preset_choice == '1': settings.set_duration_preset("Quick")
            elif preset_choice == '2': settings.set_duration_preset("Standard")
            elif preset_choice == '3': settings.set_duration_preset("Stress")

        elif choice == '2':
            try:
                val = int(input("\nEnter pass count (1-10): "))
                if 1 <= val <= 10: settings.passes = val
            except ValueError: pass

        elif choice == '3':
            try:
                val = int(input(f"\nEnter CPU thread count (1-{hw['cpu_cores_logical']}): "))
                if 1 <= val <= hw['cpu_cores_logical']: settings.cpu_threads = val
            except ValueError: pass

        elif choice == '4':
            try:
                val = int(input("\nEnter CPU iterations per thread (e.g. 1000000): "))
                if val >= 100000: settings.cpu_iterations = val
            except ValueError: pass

        elif choice == '5':
            try:
                val = int(input("\nEnter RAM allocation size in MB (e.g. 256, 512, 1024): "))
                if val >= 64: settings.ram_size_mb = val
            except ValueError: pass

        elif choice == '6':
            try:
                val = int(input("\nEnter GPU matrix size (e.g. 2048, 4096, 8192): "))
                if val >= 512: settings.gpu_matrix_size = val
            except ValueError: pass

        elif choice == '7':
            settings.detailed_metrics = not settings.detailed_metrics

        elif choice == '0':
            break

def individual_tests_menu(settings: BenchmarkSettings, hw: dict):
    while True:
        display_header(hw)
        print(f"{C_YELLOW}{C_BOLD}[ INDIVIDUAL TEST SELECTION ]{C_RESET}\n")
        print(f"  {C_CYAN}1.{C_RESET} Run CPU Benchmark Only ({settings.cpu_threads} Threads)")
        print(f"  {C_CYAN}2.{C_RESET} Run RAM Bandwidth Test ({settings.ram_size_mb} MB)")
        print(f"  {C_CYAN}3.{C_RESET} Run GPU Matrix Compute Test ({settings.gpu_matrix_size}x{settings.gpu_matrix_size})")
        print(f"\n  {C_CYAN}0.{C_RESET} Back to Main Menu\n")

        choice = input(f"Select option {C_CYAN}[0-3]{C_RESET}: ").strip()

        if choice == '1':
            print(f"\n{C_CYAN}[+] Running CPU Test...{C_RESET}")
            res = run_cpu_test(settings)
            print(f"  -> Score        : {C_GREEN}{res['score']} pts{C_RESET}")
            print(f"  -> Avg Latency  : {res['avg_time_sec']}s")
            if settings.detailed_metrics:
                print(f"  -> Min/Max Time : {res['min_time_sec']}s / {res['max_time_sec']}s")
                print(f"  -> Ops/Sec      : {res['ops_per_sec']:,}")
            input("\nPress Enter to continue...")

        elif choice == '2':
            print(f"\n{C_CYAN}[+] Running RAM Test...{C_RESET}")
            res = run_ram_test(settings)
            print(f"  -> Score        : {C_GREEN}{res['score']} pts{C_RESET}")
            print(f"  -> Throughput   : {res['bandwidth_gb_s']} GB/s")
            if settings.detailed_metrics:
                print(f"  -> Min/Max Time : {res['min_time_sec']}s / {res['max_time_sec']}s")
            input("\nPress Enter to continue...")

        elif choice == '3':
            print(f"\n{C_CYAN}[+] Running GPU Test...{C_RESET}")
            res = run_gpu_test(settings)
            if res["status"] == "Success":
                print(f"  -> Score        : {C_GREEN}{res['score']} pts{C_RESET}")
                print(f"  -> Compute Rate : {res['tflops']} TFLOPS")
                if settings.detailed_metrics:
                    print(f"  -> Avg Latency  : {res['avg_time_sec']}s")
                    print(f"  -> Min/Max Time : {res['min_time_sec']}s / {res['max_time_sec']}s")
            else:
                print(f"  {C_RED}[!] {res.get('reason', 'GPU test failed')}{C_RESET}")
            input("\nPress Enter to continue...")

        elif choice == '0':
            break

def run_full_suite(settings: BenchmarkSettings, hw: dict):
    display_header(hw)
    print(f"{C_MAGENTA}{C_BOLD}[ EXECUTING FULL BENCHMARK SUITE ]{C_RESET}\n")

    print(f"{C_CYAN}[1/3] Testing CPU Multithreaded Performance ({settings.cpu_threads} threads)...{C_RESET}")
    cpu_res = run_cpu_test(settings)
    print(f"      Score: {C_GREEN}{cpu_res['score']} pts{C_RESET} ({cpu_res['avg_time_sec']}s avg)")

    print(f"\n{C_CYAN}[2/3] Testing RAM Memory Bandwidth ({settings.ram_size_mb} MB array)...{C_RESET}")
    ram_res = run_ram_test(settings)
    print(f"      Score: {C_GREEN}{ram_res['score']} pts{C_RESET} ({ram_res['bandwidth_gb_s']} GB/s)")

    print(f"\n{C_CYAN}[3/3] Testing GPU Compute Performance...{C_RESET}")
    gpu_res = run_gpu_test(settings)
    if gpu_res["status"] == "Success":
        print(f"      Score: {C_GREEN}{gpu_res['score']} pts{C_RESET} ({gpu_res['tflops']} TFLOPS)")
    else:
        print(f"      Status: {C_YELLOW}{gpu_res.get('reason', 'Skipped')}{C_RESET}")

    gpu_score = gpu_res.get('score', 0) if gpu_res["status"] == "Success" else 0
    total_score = round(cpu_res['score'] + ram_res['score'] + gpu_score, 1)

    print(f"\n{C_CYAN}──────────────────────────────────────────────────────{C_RESET}")
    print(f"  {C_BOLD}TOTAL AETHER BENCHMARK SCORE : {C_GREEN}{total_score}{C_RESET}")
    print(f"{C_CYAN}──────────────────────────────────────────────────────{C_RESET}")

    output_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hw,
        "total_score": total_score,
        "results": {"cpu": cpu_res, "ram": ram_res, "gpu": gpu_res}
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_payload, f, indent=4)
    print(f"\n[+] Detailed benchmark results saved to {OUTPUT_FILE}")
    input("\nPress Enter to return to main menu...")

# ==============================================================================
# MAIN BOOT LOGIC
# ==============================================================================

def main():
    hw = detect_hardware()
    settings = BenchmarkSettings(hw['cpu_cores_logical'])

    while True:
        display_header(hw)
        print(f"{C_YELLOW}{C_BOLD}[ MAIN DASHBOARD ]{C_RESET}\n")
        print(f"  {C_CYAN}1.{C_RESET} Run Full Benchmark Suite (CPU + RAM + GPU)")
        print(f"  {C_CYAN}2.{C_RESET} Select Individual Tests")
        print(f"  {C_CYAN}3.{C_RESET} Benchmark Settings & Metrics Configuration")
        print(f"\n  {C_CYAN}0.{C_RESET} Exit\n")

        choice = input(f"Select option {C_CYAN}[0-3]{C_RESET}: ").strip()

        if choice == '1':
            run_full_suite(settings, hw)
        elif choice == '2':
            individual_tests_menu(settings, hw)
        elif choice == '3':
            settings_menu(settings, hw)
        elif choice == '0':
            clear_screen()
            sys.exit(0)

if __name__ == "__main__":
    main()
