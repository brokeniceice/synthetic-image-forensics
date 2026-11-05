# Synthetic Image Forensics

Code for synthetic-image classification, perceptual artifact localization, and auxiliary explanation generation. The model combines global RINE evidence with a GLaMM language backbone and uses dense forensic evidence to correct spatial embeddings before SAM mask decoding.

The frozen paper model is **C1 + the frozen P1 SAM decoder + NoRelative R1**. Checkpoint identities, prompts, decision rules, and expected results are recorded in [`configs/paper/protocol.json`](configs/paper/protocol.json).

## Setup and checkpoints

Follow [environment setup](docs/SETUP.md), [checkpoint assembly](docs/CHECKPOINTS.md), and [data preparation](docs/DATA.md). Model weights and datasets are separate assets and are not included in this code repository. Download links for the trained checkpoints have not been published in this snapshot.

Run commands from the repository root. Inference requires the base GLaMM and CLIP checkpoints in addition to the trained adapters.

## Classification

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --task classification --manifest data/classification.jsonl \
  --checkpoint checkpoints/c1.pt --output outputs/classification

python scripts/evaluate_classification.py \
  --predictions outputs/classification/predictions.jsonl \
  --references data/classification_labels.jsonl \
  --output outputs/classification/metrics.json
```

Classification uses the canonical forensic question, a fixed `[CLS]` query, and the Fake-minus-Real logit margin. The final decision center is the mean TRAIN margin, **−9.190834884678093**; it is fixed across internal and OOD data. Evaluation reports raw and centered results. There is no validation threshold search.

## Localization

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --task localization --manifest data/localization.jsonl \
  --checkpoint checkpoints/c1.pt --spatial-bundle checkpoints/spatial.pt \
  --output outputs/localization

python scripts/evaluate_masks.py \
  --manifest data/localization_masks.jsonl \
  --output outputs/localization/metrics.json
```

The localization main table uses known-Fake G1 generation: 400 continuation tokens, control-token suppression, and one appended `[SEG]` query only if none is generated naturally. C1 provides language queries; the frozen P1 SAM path decodes spatial features corrected by the final NoRelative R1. Masks are unioned at original resolution with logit threshold zero.

## Explanation

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --task explanation --manifest data/explanation.jsonl \
  --checkpoint checkpoints/c1.pt --output outputs/explanation

python scripts/evaluate_explanation.py \
  --predictions outputs/explanation/predictions.jsonl \
  --references data/explanation_references.jsonl \
  --output outputs/explanation/metrics
```

The explanation main table uses **autonomous G0**, the E03 explanation prompt, and 512 tokens. Its CSS and ROUGE-L implementation is preserved, including the pinned MiniLM revision and 128-token encoder limit. This route does not prepend the known-Fake prefix used for localization. R1 changes masks and does not change explanation text.

## Code layout

| Directory | Contents |
|---|---|
| `forensics/` | Final inference, checkpoint loading, geometry gating and text metrics |
| `model/` | GLaMM, RINE, dense forensic adapter, utility network, rectifier and SAM |
| `dataset/` | Unified forensic data and inherited GLaMM loaders |
| `scripts/` | Inference and the three task evaluators; asset/center utilities |
| `training/` | Training implementations and their required internal helpers |
| `configs/` | Training recipes and the final paper protocol |
| `mmdet/` | Vendored MMDetection dependency used by the inherited region encoder |

[Training and selection](docs/TRAINING.md) describes the separate P1/C1 branches and three-stage R1 training. [Evaluation](docs/EVALUATION.md) defines input formats and metric conventions. The initial release has been checked against frozen result assets; validation details are in [release checks](docs/VALIDATION.md).

## Acknowledgments and license

Built on GLaMM, LLaVA, Segment Anything, RINE, and MMDetection. See [NOTICE](NOTICE) and the retained source notices. Project code is distributed under [Apache-2.0](LICENSE); upstream components and model/data assets retain their respective terms.
