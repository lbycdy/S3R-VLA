<h1 align="center">S3R-VLA: Self-Supervised Skill Learning for Robust Vision-Language-Action Models</h1>

<p align="center">
Bo Liu<sup>1,2</sup>, Qiyan Zhao<sup>3,4</sup>, Liming Zhang<sup>1</sup>, Jingtao Yang<sup>1</sup>, Jun Huang<sup>1</sup>, Xu-Yao Zhang<sup>3,4</sup>, Yong Liu<sup>1,2,*</sup>
</p>
<p align="center">
<sup>1</sup>Nanjing University of Science and Technology &nbsp;
<sup>2</sup>State Key Laboratory of Intelligent Manufacturing of Advanced Construction Machinery &nbsp;
<sup>3</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences &nbsp;
<sup>4</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences
<br><sup>*</sup>Corresponding author
</p>

<p align="center">
  <a href="https://lbycdy.github.io/S3R-VLA/"><img src="https://img.shields.io/badge/Project-Page-Green.svg" alt="Project Page"></a>
</p>

## Overview

S3R-VLA learns discrete skills during VLA policy training. A training-only **posterior branch** maps the actions and observation changes within a window to a skill code in a shared codebook, and a deployable **prior branch** learns to predict this code from the current observation and language instruction. The code builds on [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) and [starVLA](https://github.com/starVLA/starVLA).

## Versions and Checkpoints

All versions are implemented in [`S3R_VLA.py`](./starVLA/model/framework/S3R_VLA.py), share one launcher, and are initialized from the VLA-JEPA pretrained checkpoint. v3 follows the method of the paper; v1 and v2 are earlier versions (their differences are listed in the class docstrings).

| Version | Config | Global batch | Weights (LIBERO) |
|---|---|---|---|
| v1 | [`s3r_vla_v1_libero_ft.yaml`](./scripts/configs/s3r_vla_v1_libero_ft.yaml) | 64 (4 × 16) | <!-- TODO: link --> coming soon |
| v2 | [`s3r_vla_v2_libero_ft.yaml`](./scripts/configs/s3r_vla_v2_libero_ft.yaml) | 64 (4 × 16) | <!-- TODO: link --> coming soon |
| v3 | [`s3r_vla_v3_libero_ft.yaml`](./scripts/configs/s3r_vla_v3_libero_ft.yaml) | 256 (8 × 32) | config only |

A released checkpoint keeps the layout written by the trainer, which the evaluation scripts expect:
```
s3r_vla_v1_libero_ft/
├── config.yaml
├── dataset_statistics.json
└── checkpoints/steps_120000_pytorch_model.pt
```

## Installation

```bash
git clone https://github.com/lbycdy/S3R-VLA.git && cd S3R-VLA
conda create -n s3r_vla python=3.10 -y && conda activate s3r_vla
pip install -r requirements.txt && pip install -e .
pip install tyro matplotlib mediapy websockets msgpack numpy==1.24.4   # evaluation client / server
```

## Preparation

Download the pretrained models into `./models` and the LIBERO datasets in LeRobot format ([collection](https://huggingface.co/collections/IPEC-COMMUNITY/libero-benchmark-dataset)) into `./datasets`, and copy [`examples/LIBERO/modality.json`](./examples/LIBERO/modality.json) into the `meta/` directory of each dataset:
```
models/
├── Qwen3-VL-2B-Instruct/                    # https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
├── vjepa2-vitl-fpc64-256/                   # https://huggingface.co/facebook/vjepa2-vitl-fpc64-256
└── VLA-JEPA-pretrain/VLA-JEPA-pretrain.pt   # https://huggingface.co/ginwind/VLA-JEPA
datasets/LeRobot/LIBERO/
├── libero_spatial_no_noops_1.0.0_lerobot/
├── libero_object_no_noops_1.0.0_lerobot/
├── libero_goal_no_noops_1.0.0_lerobot/
└── libero_10_no_noops_1.0.0_lerobot/
```
Other locations can be set in [`s3r_vla_libero_base.yaml`](./scripts/configs/s3r_vla_libero_base.yaml).

## Training

```bash
SMOKE=1 bash scripts/train.sh v3    # 30-step smoke test
bash scripts/train.sh v3            # v1 / v2 / v3, or the path of a config
```

Each run takes 120k steps and writes a checkpoint every 10k steps to `checkpoints/<run_id>/`. The number of GPUs and the per-GPU batch size come from the config and can be overridden with `NPROC` and `BS`; extra arguments override config entries. The ablations of the paper are config overrides of v3, e.g.

```bash
bash scripts/train.sh v3 --framework.skill_model.skill_full_attention false --run_id s3r_vla_v3_wo_full_attention
```

| Ablation | Override (`framework.skill_model`) |
|---|---|
| w/o obs. change / w/o action | `posterior_inputs: [action]` / `posterior_inputs: [obs]` |
| w/o frame difference | `frame_difference: false` |
| w/o full skill attention / w/o world model | `skill_full_attention: false` / `wm_weight: 0` |
| DASQ without kernel update / restart / regularization | `kernel_weight: 0` / `revive_every: 0` / `diversity_weight: 0` |

## Evaluation

Each script starts one policy server per GPU in `GPUS` (about 6 GB each) and one simulator client per task suite or perturbation dimension. The paper reports the mean of `SEED=3`, `5`, and `7` (default 7). Results and videos are written to `results/`.

**LIBERO.** Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) in a separate environment, then run
```bash
CKPT=/path/to/s3r_vla_v1_libero_ft/checkpoints/steps_120000_pytorch_model.pt \
LIBERO_HOME=/path/to/LIBERO SIM_PYTHON=/path/to/libero_env/bin/python \
GPUS="0 1 2 3" bash examples/LIBERO/eval_libero.sh
```

**LIBERO-Plus.** Install [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) in a separate environment and replace `LIBERO-plus/libero/libero/benchmark/__init__.py` with [`libero_plus_init.py`](./examples/LIBERO-Plus/libero_plus_init.py), then run
```bash
CKPT=/path/to/s3r_vla_v1_libero_ft/checkpoints/steps_120000_pytorch_model.pt \
LIBERO_HOME=/path/to/LIBERO-plus SIM_PYTHON=/path/to/libero_plus_env/bin/python \
GPUS="0 1 2 3 4 5 6" bash examples/LIBERO-Plus/eval_libero_plus.sh
```

## Acknowledgement

We thank [VLA-JEPA](https://github.com/ginwind/VLA-JEPA), [starVLA](https://github.com/starVLA/starVLA), [V-JEPA 2](https://github.com/facebookresearch/vjepa2), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), and [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) for their open-source contributions.

## License

<!-- TODO: choose a license for S3R-VLA and add a LICENSE file. -->
This repository contains code from starVLA (MIT License), VLA-JEPA (Apache-2.0 License), and V-JEPA 2 (MIT License); the original copyright notices are kept in the corresponding files.

## Citation

```bibtex
@misc{liu2026s3rvla,
  title        = {{S3R-VLA}: Self-Supervised Skill Learning for Robust Vision-Language-Action Models},
  author       = {Liu, Bo and Zhao, Qiyan and Zhang, Liming and Yang, Jingtao and Huang, Jun and Zhang, Xu-Yao and Liu, Yong},
  year         = {2026},
  howpublished = {\url{https://lbycdy.github.io/S3R-VLA/}}
}
```
