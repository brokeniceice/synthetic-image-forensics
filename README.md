# Synthetic Image Forensics

Synthetic-image classification, perceptual artifact localization, and auxiliary explanation generation with **C1 + frozen P1 SAM decoding + NoRelative spatial correction**.

The inference model is distributed as one Hugging Face repository. One download includes the GLaMM/tokenizer and CLIP snapshots plus a combined file containing the selected C1, RINE, and spatial tensors. The loader assembles the model automatically; users do not prepare separate checkpoints or merge adapters.

## Install

Follow [environment setup](docs/SETUP.md), then run commands from the repository root. Prepare image manifests as described in [data formats](docs/DATA.md).

## Run

Model weights: [broken-ice/synthetic-image-forensics](https://huggingface.co/broken-ice/synthetic-image-forensics). A downloaded local model directory is also accepted:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --model broken-ice/synthetic-image-forensics --task classification \
  --manifest data/classification.jsonl --output outputs/classification

CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --model broken-ice/synthetic-image-forensics --task localization \
  --manifest data/localization.jsonl --output outputs/localization

CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --model broken-ice/synthetic-image-forensics --task explanation \
  --manifest data/explanation.jsonl --output outputs/explanation
```

For one image, replace `--manifest ...` with `--image path/to/image.png`. `--revision` pins a Hugging Face commit or tag; `--local-files-only` uses the local cache.

```python
from forensics.runtime import Runtime

runtime = Runtime.from_pretrained("broken-ice/synthetic-image-forensics", device="cuda:0")
```

## Evaluate

```bash
python scripts/evaluate_classification.py \
  --predictions outputs/classification/predictions.jsonl \
  --references data/classification_labels.jsonl \
  --output outputs/classification/metrics.json

python scripts/evaluate_masks.py \
  --manifest data/localization_masks.jsonl \
  --output outputs/localization/metrics.json

python scripts/evaluate_explanation.py \
  --predictions outputs/explanation/predictions.jsonl \
  --references data/explanation_references.jsonl \
  --output outputs/explanation/metrics
```

| Task | Frozen paper setting |
|---|---|
| Classification | Canonical fixed CLS question; TRAIN center −9.190834884678093; batch 8 |
| Localization | Frozen P1 decoder and final NoRelative correction; known-Fake G1; 400 continuation tokens |
| Explanation | E03 autonomous G0; 512 tokens; pinned MiniLM CSS and ROUGE-L F1 |

LoRA tensors retain their original values and computation. They are not numerically merged into the base weights. Checkpoint identities, prompts, metric rules and expected results are in [`configs/paper/protocol.json`](configs/paper/protocol.json).

## Code

| Directory | Contents |
|---|---|
| `forensics/` | Hugging Face loading and the final inference path |
| `model/` | GLaMM/RINE, dense evidence, utility, spatial attention and SAM |
| `dataset/` | Unified forensic samples and exact batch collation |
| `scripts/` | Inference, three task evaluators, TRAIN center and model export |
| `training/` | Detector, RINE and three-stage spatial training; checkpoint preparation |
| `eval/`, `tools/` | Shared metric, geometry and training utilities |
| `configs/` | C1, P1 decoder and final spatial recipes |

The inherited ROI extractor is retained as one attributed source file; the unrelated detector registry and other model branches are removed. See [training](docs/TRAINING.md), [model packaging](docs/CHECKPOINTS.md), [evaluation](docs/EVALUATION.md), and [validation](docs/VALIDATION.md).

## Acknowledgments

Built on GLaMM, LLaVA, Segment Anything, RINE and the OpenMMLab ROI implementation. See [NOTICE](NOTICE) and [LICENSE](LICENSE). Upstream weights retain their original terms and model cards. Dataset images and annotations are not included.
