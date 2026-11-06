# Synthetic Image Forensics

Classification, artifact localization and explanation generation for synthetic images.

## Install

Use Linux, Python 3.10 and a CUDA GPU supporting BF16. Run commands from the repository root.

```bash
conda create -n image-forensics python=3.10 -y
conda activate image-forensics
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install mmcv-full==1.4.7 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
pip install -r requirements.txt
```

## Inference

The complete model is downloaded automatically from [Hugging Face](https://huggingface.co/broken-ice/synthetic-image-forensics). Its frozen CLIP encoder is downloaded separately at the version specified in the model configuration. Both are cached automatically.

Run the joint system on one image:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --model broken-ice/synthetic-image-forensics \
  --image path/to/image.png --output outputs/predictions
```

The system predicts image authenticity, generates an explanation, and produces the final artifact mask. The predicted class drives explanation generation; localization uses queries from that same generated explanation. Images predicted as real receive an empty artifact mask.

Each record in `predictions.jsonl` contains the predicted class, explanation and mask path. Binary PNG masks are saved under `masks/`. Use a new output directory for each run.

For multiple images, create a JSONL manifest with one record per line:

```json
{"sample_id":"example_001","image_path":"images/example_001.png"}
{"sample_id":"example_002","image_path":"images/example_002.png"}
```

Use unique sample IDs. Relative image paths are resolved against the manifest directory.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --model broken-ice/synthetic-image-forensics \
  --manifest data/images.jsonl --output outputs/predictions
```

`--model` also accepts a downloaded model directory. Use `--revision` to select a model commit or tag. Once the model and CLIP are cached, use `--local-files-only` to run offline. `--max-new-tokens` sets the explanation length limit (default: 400).

## Evaluation

Use `scripts/evaluate.py` to generate predictions for the three main experiments:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model broken-ice/synthetic-image-forensics --task classification \
  --manifest data/classification.jsonl --output outputs/evaluation/classification

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model broken-ice/synthetic-image-forensics --task localization \
  --manifest data/localization.jsonl --output outputs/evaluation/localization

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model broken-ice/synthetic-image-forensics --task explanation \
  --manifest data/explanation.jsonl --output outputs/evaluation/explanation
```

Use the following scoring commands with the generated predictions and references.

### Classification

Create a label JSONL file. Use `0` for real and `1` for synthetic; sample IDs must match the predictions.

```json
{"sample_id":"example_001","label":1}
```

```bash
python scripts/evaluate_classification.py \
  --predictions outputs/evaluation/classification/predictions.jsonl \
  --references data/classification_labels.jsonl \
  --output outputs/evaluation/classification/metrics.json
```

The evaluator reports accuracy and F1 for raw scores and the default training-set center.

### Localization

Create a JSONL file pairing each predicted mask with its ground-truth mask:

```json
{"sample_id":"example_001","prediction_mask":"predictions/example_001.png","target_mask":"targets/example_001.png","valid_q_seg":true}
```

Paths are relative to this manifest. Use the `mask_path` and `valid_q_seg` values from the localization evaluation predictions, adjusting the mask path to the manifest location. Masks must have matching dimensions at the original image size; nonzero pixels represent foreground.

```bash
python scripts/evaluate_masks.py \
  --manifest data/localization_masks.jsonl \
  --output outputs/evaluation/localization/metrics.json
```

The evaluator reports foreground IoU/F1 and foreground/background mIoU.

### Explanation

Create a reference JSONL file with the same sample IDs and ordering as the predictions:

```json
{"sample_id":"example_001","reference":"The object has inconsistent geometry."}
```

```bash
python scripts/evaluate_explanation.py \
  --predictions outputs/evaluation/explanation/predictions.jsonl \
  --references data/explanation_references.jsonl \
  --output outputs/evaluation/explanation/metrics
```

The evaluator downloads its MiniLM text encoder automatically and reports CSS and ROUGE-L F1. Use a new output directory.

## Training

Training consists of RINE training, C1 joint training and R1 training. Set model, dataset and output paths in `configs/c1.yaml` and `configs/r1.yaml` before running. Training data and prepared feature caches are not included in the model download.

Place `train_combined.jsonl`, `val_combined.jsonl` and `test_combined.jsonl` in the configured manifest directory. Records contain `sample_id`, `forensics_domain` (`real` or `fake`), `class_label` (`0` or `1`) and `image_path` or `image_relpath`. Synthetic-image records also need `explanation` and polygon annotations in `refs`; see `dataset/forensics/unified.py` and `dataset/forensics/synthscars.py` for the annotation structure.

### 1. RINE training

Train the RINE head with a CLIP snapshot and an initial head checkpoint:

```bash
python -m training.rine \
  --clip checkpoints/clip-vit-large-patch14-336 \
  --initial-head checkpoints/rine_initial_head.pt \
  --train-manifest data/manifests/train_combined.jsonl \
  --val-manifest data/manifests/val_combined.jsonl \
  --output outputs/rine
```

### 2. C1 joint training

Set the selected RINE checkpoint path in `configs/c1.yaml`, then jointly train the classification, language and localization components:

```bash
deepspeed --include localhost:0 train.py \
  --config configs/c1.yaml --mode train
```

### 3. R1 training

In `configs/r1.yaml`, set `initialization.model` to the published model ID or a downloaded model directory. R1 reads the frozen segmentation decoder and dense evidence source directly from these weights. Configure the evidence heads, spatial/query/utility caches and DEV records, then run rectifier fitting, utility fitting and joint fitting in order:

```bash
python -m training.spatial rectifier --config configs/r1.yaml
python -m training.spatial utility --config configs/r1.yaml
python -m training.spatial joint --config configs/r1.yaml
```

## Acknowledgments

Built on GLaMM, LLaVA, Segment Anything, RINE and OpenMMLab. See [NOTICE](NOTICE) and [LICENSE](LICENSE) for attribution and licensing.
