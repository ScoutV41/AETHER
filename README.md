
### ROCm Crowdsourced Benchmark Runner
A standardized, community-driven hardware profiling and LLM inference benchmark tool. This project gathers real-world telemetry from volunteers to build an open, comparative index of local AI performance, with special attention given to native AMD ROCm optimization.

By crowdsourcing performance data across varied systems, this tool eliminates guesswork and creates an authentic baseline for consumer and enterprise hardware alike.

Key Features
Robust AMD GPU Profiling: Moves past primitive OS naming conventions. It executes native rocm-smi JSON queries on Linux and parses structured dxdiag XML trees on Windows to extract exact architecture series and true VRAM metrics.

Quantization Tagging: Automatically queries Ollama's local /api/show endpoint prior to execution. This maps the exact quantization family (such as Q4_K_M or FP16) to ensure fair data categorization on the backend.

Deterministic Baselines: Enforces a rigid 2048 context window payload limit. This locks down the attention matrix memory footprint, guaranteeing a true apples-to-apples performance comparison across all volunteer systems.

NVIDIA Fallback Support: Seamlessly falls back to GPUtil if an NVIDIA card is present, allowing cross-vendor comparative analysis.

Prerequisites
Before running the benchmark, ensure you have the following components set up on your machine:

Python 3.8 or Higher

Ollama Inference Engine: Installed and running locally.

Target Model: The default benchmark runs against qwen2.5:7b. You can pull it using your terminal:

Bash
ollama run qwen2.5:7b
Quick Start
1. Clone the Repository
Bash
git clone https://github.com/your-username/rocm-crowdsourced-benchmark.git
cd rocm-crowdsourced-benchmark
2. Install Dependencies
Install the required hardware tracking and networking packages:

Bash
pip install psutil gputil requests
3. Run the Benchmark
Execute the script to profile your hardware and run the local inference test:

Bash
python benchmark_runner.py
Telemetry Payload Structure
The script compiles a structured JSON object ready for database submission. It categorizes host hardware and performance separately:

JSON
{
    "timestamp": 1718541913,
    "hardware": {
        "os": "Linux",
        "os_release": "6.8.0-amd64",
        "cpu_model": "AMD Ryzen 7 7800X3D 8-Core Processor",
        "cpu_cores_physical": 8,
        "cpu_cores_logical": 16,
        "ram_gb": 31.24,
        "gpu_model": "AMD Radeon RX 7900 XTX",
        "gpu_vram_gb": 24.0,
        "gpu_driver": "Unknown"
    },
    "performance": {
        "benchmark_status": "Success",
        "model_tested": "qwen2.5:7b",
        "quantization_level": "Q4_K_M",
        "model_format": "gguf",
        "context_window_set": 2048,
        "total_wall_time_sec": 4.12,
        "prompt_tokens": 18,
        "generation_tokens": 312,
        "prompt_eval_tokens_per_sec": 410.5,
        "generation_tokens_per_sec": 92.15
    }
}

 ## License
 This project is licensed under the Apache License 2.0.What this means for you:

- You Can: Use, modify, distribute, and sell this software for personal or commercial projects completely free of charge.
 - Patent Protection: Every contributor grants you a royalty-free license to any patents they hold on the code. If anyone sues you over patents in this software, their license is instantly revoked.
 - Keep Notices: You must include a copy of the original license, copyright, and NOTICE file in any distribution.
 - Track Changes: If you modify any existing files in this repository, you must prominently state inside those files that you changed them.
 - No Trademark Rights: This license does not grant you the right to use the project's name, logos, or trademarks.
