# Checkpoints

The final system combines separately selected components. It does not replace the P1 decoder with the original GLaMM or C1 decoder.

| Local asset | Role |
|---|---|
| `checkpoints/GLaMM-FullScope/` | GLaMM base model and tokenizer, before the project P1/C1 adaptation |
| `checkpoints/clip-vit-large-patch14-336/` | Shared Hugging Face CLIP vision tower and processor |
| `checkpoints/rine.pt` | Selected frozen RINE head (`rine_state_dict`) |
| `checkpoints/c1.pt` | C1 epoch 5 / optimizer step 2500 module checkpoint |
| `checkpoints/p1_sam_runtime.pt` | Frozen P1 prompt encoder and mask decoder |
| `checkpoints/r1.pt` | Final NoRelative R1, seed 3407, Joint epoch 8 |
| `checkpoints/forensic_source.pt` | Selected dense forensic adapter, epoch 4 |
| `checkpoints/spatial.pt` | Portable bundle assembled from the preceding three spatial assets |

Trained checkpoint SHA-256 identifiers are in `configs/paper/protocol.json`. Inference checks the C1 and RINE identities, and bundle assembly checks the final R1 and forensic source identities. The P1 runtime records the P1 parent checkpoint hash.

## Assemble the spatial bundle

```bash
python scripts/package_spatial.py \
  --r1 checkpoints/r1.pt \
  --p1-runtime checkpoints/p1_sam_runtime.pt \
  --forensic-source checkpoints/forensic_source.pt \
  --gamma 0.01 --output checkpoints/spatial.pt
```

This packs learned tensors without optimization. The utility checkpoint already includes its frozen source heads and temperature buffers. The rectifier uses eight attention heads and locality sigma 0.25.

## P1 decoder provenance

P1 and C1 are separate branches initialized from the same pre-P1 GLaMM state. P1 is trained with the phrase-aligned recipe in `configs/phase3a_p1.yaml`; its selected checkpoint is epoch 7 / step 3500. C1 uses `configs/phase6d3_c1_rine_conditioned_p1.yaml`; the historical `p1` suffix in that filename refers to the training recipe, not initialization from a trained P1 checkpoint.

The prompt encoder starts from GLaMM and receives any P1 checkpoint overrides. All P1 mask-decoder tensors are taken from the selected P1 checkpoint. `training/phase4f_prepare.py::extract_runtime` contains this extraction logic. The resulting prompt/mask path stays frozen through all final R1 stages and inference. C1 supplies the projected language queries.

Checkpoint files and dataset images are not committed to Git. This code snapshot does not yet provide trained-weight download links. Upstream weights retain their original licenses and access requirements.
