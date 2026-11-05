# Data formats

Obtain datasets from their original providers and retain the frozen splits when reproducing the paper. Images, reference descriptions and annotations are not distributed in this code snapshot. Do not rebuild a random split when comparing to the recorded results.

## Inference

All three tasks accept an image-only JSONL manifest:

```json
{"sample_id":"example_001","image_path":"images/example_001.png"}
```

Use a unique ID per image. Relative image paths resolve against the manifest directory. Inference does not read reference explanations, target polygons or masks. Localization explicitly uses the known-Fake G1 condition. Explanation uses autonomous G0.

## Classification labels

```json
{"sample_id":"example_001","label":1}
```

Real is 0; Fake is 1. Prediction and reference ID sets must match exactly. The internal classification test has 2,208 images; the OOD populations are AIGI-Holmes 99,999, GenImage 100,000 and LOKI 2,217.

## Localization references

```json
{"sample_id":"example_001","prediction_mask":"predictions/example_001.png","target_mask":"targets/example_001.png","valid_q_seg":true}
```

Masks must be at the original image size. Nonzero pixels are foreground. Paths resolve against this manifest. Use official per-image union targets: SynthScars polygons, LOKI box unions, X-AIGD annotations and PAL4VST pixel masks. Keep all official empty targets. The four populations contain 1,000 / 229 / 2,419 / 1,441 images respectively.

## Explanation references

```json
{"sample_id":"example_001","reference":"The object has inconsistent geometry."}
```

References must have the same ID order as predictions. The explanation evaluation uses SynthScars Official1000 and LOKI229. LOKI references are the original region descriptions deduplicated in order; do not concatenate the extra global descriptions.

## Training

`UnifiedForensicsDataset` reads `train_combined.jsonl`, `val_combined.jsonl` and `test_combined.jsonl`. Its required fields include `sample_id`, `forensics_domain` (`real`/`fake`), `class_label`, and `image_path` or `image_relpath`. Fake examples also contain an `explanation` and `refs` with authoritative phrases and polygon annotations. See `dataset/forensics/unified.py` and `dataset/forensics/synthscars.py` for the accepted annotation representation. Frozen populations are TRAIN 17,672, VAL 2,212 and TEST 2,208; the Fake subsets used for spatial training are 8,836 and 1,106.
