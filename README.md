# AETHER - Local AI Inference Benchmark

**Cross-platform, crowdsourced performance benchmarking for local AI inference.**  
Run a standardized test on your hardware, share your results, help build the first real-world database of local LLM performance across AMD, NVIDIA, and CPU setups.

 · Tested on AMD RX 9070 XT

---

## Why this exists

Every local AI benchmark that exists today either:
- Tests raw GPU compute (FLOPS, memory bandwidth) - not inference speed
- Only covers NVIDIA/CUDA
- Has no community dataset to compare against

AETHER benchmarks what actually matters: **how fast does a real model generate tokens on your real hardware**, and how does that compare to everyone else with the same card?

---

## What it measures

- **Generation speed** (tokens/sec) - the number that actually matters for usability
- **Prompt evaluation speed** - how fast your hardware processes the input
- **Hardware profile** - GPU model, VRAM, driver, ROCm/CUDA version, CPU, RAM
- **Acceleration backend** - DirectML, Vulkan, ROCm, or CUDA (whatever your setup uses)
- **Model info** - quantization level, format, family

All results are saved locally to `benchmark_result.json`. Nothing is uploaded automatically - you share what you want, when you want.

---

## Supported setups

| Hardware | OS | Backend |
|---|---|---|
| AMD GPU (RX 6000+) | Windows | DirectML / Vulkan via LM Studio |
| AMD GPU (RX 6000+) | Linux | ROCm via Ollama or LM Studio |
| NVIDIA GPU | Windows / Linux | CUDA via Ollama or LM Studio |
| CPU only | Any | Ollama or LM Studio |
| Apple Silicon | macOS | Metal via Ollama or LM Studio |

---

## Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai) **or** [Ollama](https://ollama.com) running locally with at least one vision-capable model loaded
- A loaded model (recommended: `qwen2.5-vl-7b-instruct` or `qwen2.5:7b`)

```bash
pip install psutil GPUtil requests
```

> **Note:** `GPUtil` is for NVIDIA detection. If you're on AMD or CPU-only, it's optional - AETHER will warn you if it's missing but continue fine.

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/ScoutV41/AETHER
cd AETHER
```

**2. Install dependencies**
```bash
pip install psutil GPUtil requests
```

**3. Start your inference backend**

For LM Studio: open the app, load a model, and make sure the local server is running (default port 1234)

For Ollama:
```bash
ollama serve
ollama pull qwen2.5:7b
```

**4. Run the benchmark**
```bash
python benchmark.py
```

---


---

## Sharing your results

Results are saved to `benchmark_result.json` in the same folder you ran the script from.

Share them in the [Discord server](https://discord.gg/yhQeCRyc) in the `#benchmark-results` channel. So I can crowdsource results for comparison charts.

Paste the full JSON or attach the file. Include:
- Your GPU model
- Your OS
- The model + quantization you tested
- Anything unusual about your setup (undervolted, custom drivers, etc.)

---

## What we're building toward

The goal is a community-maintained database of real-world local inference performance, searchable by GPU, model, quantization, and OS. Think [UserBenchmark](https://www.userbenchmark.com) but for local AI + Performance metrics - and actually trustworthy.

Planned features:
- [ ] Web dashboard to browse and compare submissions
- [ ] Multi-model test suite (run 3-5 models in one session)
- [ ] Acceleration backend auto-detection (DirectML vs ROCm vs CUDA vs Metal)
- [ ] Optimization recommendations if your score is below average for your card
- [ ] Automated submission endpoint (opt-in)

---

## Contributing

Pull requests welcome. If you find a bug with GPU detection on your specific setup, open an issue and include your `benchmark_result.json` - hardware detection edge cases are the hardest part of this project and real examples help a lot.

Known gaps:
- Linux ROCm path is untested (need volunteers with ROCm setups)
- macOS Metal path is untested  
- Dual discrete GPU setups pick the highest VRAM card

---

## License

Apache L v.2
---

