# GAM-Agent

Official codebase for **GAM-Agent: Game-Theoretic and Uncertainty-Aware Collaboration for Complex Visual Reasoning**.

> NeurIPS 2025 Poster  
> Paper: arXiv 2505.23399

## Overview

GAM-Agent is a multi-agent framework for visual reasoning. The repository implements specialized expert agents, uncertainty-aware aggregation, and debate-style collaboration for image/video understanding and benchmark evaluation.

This cleaned release is organized as a research codebase rather than a personal working directory:
- removed local caches, git metadata, output artifacts, and private environment files
- sanitized hard-coded API keys and local absolute paths
- kept the main source code, configs, demos, and evaluation integration

## Highlights

- Multi-agent expert collaboration for visual reasoning
- Uncertainty-aware expert selection and aggregation
- Debate-based refinement for stronger final responses
- Support for local image/video demos and benchmark-oriented evaluation
- Integration with `VLMEvalKit` for standardized testing workflows

## Repository Structure

```text
GAM-Agent/
├── configs/            # Main experiment and demo configs
├── demo/               # Minimal demo assets and demo entry scripts
├── scripts/            # Training / evaluation / utility scripts
├── src/                # Core implementation
│   ├── api/
│   ├── datasets/
│   ├── models/
│   ├── trainer/
│   └── utils/
├── VLMEvalKit/         # Evaluation toolkit integration
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-org-or-username>/GAM-Agent.git
cd GAM-Agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For API-based experiments, set your key through an environment variable:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

## Quick Start

### 1) Local image demo

```bash
python demo/multi_agent_image_local_demo.py
```

Main config:
- `configs/multi_agent_local_image.yaml`
- `configs/multi_agent_local_image_enhanced_debate.yaml`

### 2) Local video demo

```bash
python demo/multi_agent_video_local_demo.py
```

### 3) API-based test script

```bash
python scripts/test_api_agent.py
```

Default benchmark-style config:
- `configs/MMbench_config_Choice.yaml`

## Configuration Notes

This repository intentionally ships **sanitized** configs.

Before running experiments, update paths and model settings in the YAML files under `configs/`:
- local model checkpoints
- dataset paths
- optional demo image/video paths
- API provider/model fields when using hosted VLMs

Recommended practice:
- keep secrets in environment variables
- avoid committing machine-specific absolute paths
- keep experimental outputs outside the tracked repository

## Evaluation

Benchmark-oriented evaluation entry points are kept in:
- `run.sh`
- `run_enhanced.sh`
- `round.sh`
- `VLMEvalKit/run.py`
- `VLMEvalKit/run_new.py`

Because evaluation pipelines depend on external datasets and local model checkpoints, you should revise the corresponding configs before use.

## Citation

If you use this repository, please cite:

```bibtex
@article{zhang2025gamagent,
  title   = {GAM-Agent: Game-Theoretic and Uncertainty-Aware Collaboration for Complex Visual Reasoning},
  author  = {Jusheng Zhang and Yijia Fan and Wenjun Lin and Ruiqi Chen and Haoyi Jiang and Wenhao Chai and Jian Wang and Keze Wang},
  journal = {arXiv preprint arXiv:2505.23399},
  year    = {2025}
}
```

## Release Notes

This package is a cleaned public-facing release prepared from the original research workspace. Non-essential artifacts such as private keys, cached outputs, nested git metadata, temporary files, and machine-specific paths were removed or sanitized.

## Acknowledgements

This repository includes or integrates `VLMEvalKit` for evaluation-related workflows. Please also follow the licensing and citation requirements of upstream dependencies.
