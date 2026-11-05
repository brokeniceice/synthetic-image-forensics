# Model loading and packaging

End users load one Hugging Face model repository:

```python
from forensics.runtime import Runtime
runtime = Runtime.from_pretrained("broken-ice/synthetic-image-forensics", device="cuda:0")
```

The same call accepts a downloaded local directory. The inference CLI uses `--model broken-ice/synthetic-image-forensics`; no separate checkpoint assembly is required. The loader uses `huggingface_hub.snapshot_download`, resolves packaged file paths and verifies the selected model identities. It does not execute remote Python code.

## Package layout

```text
config.json          # model format, fixed inference protocol and checksums
forensics.pt         # C1, RINE and selected spatial tensors
base_model/          # exact GLaMM base model/tokenizer snapshot
clip/                # exact CLIP model/processor snapshot
README.md
LICENSE
NOTICE
```

The combined weights preserve every tensor's name, dtype and value. Optimizer state and development metadata are omitted. LoRA is kept as an adapter computation to preserve the BF16 inference arithmetic.

The final model combines C1 epoch 5 / step 2500, the frozen P1 prompt/mask decoder, the selected dense forensic adapter and NoRelative spatial correction at Joint epoch 8. `configs/paper/protocol.json` records source SHA-256 identities; `configs/paper/base_assets.json` records the exact base snapshots.

## P1 decoder provenance

P1 and C1 are separate branches from the same pre-adaptation GLaMM initialization. P1 supplies the decoder, while C1 supplies scores, explanation text and projected language queries. The P1 decoder is not the original GLaMM decoder or the decoder learned by C1.

P1 is selected at epoch 7 / step 3500 using VAL loss. Its prompt encoder starts from GLaMM with any P1 overrides; the mask decoder is taken from P1. Both stay frozen throughout final spatial training and inference. The corresponding recipe is `configs/p1_decoder.yaml`; extraction is implemented in `training/extract_decoder.py`.

## Maintainer export

`scripts/export_model.py` verifies the selected source checkpoints, removes optimizer state, packs project tensors and copies the exact base snapshots into the repository layout above. This is a release-maintenance step, not a requirement for users running inference. The model package must be exported and checked before uploading to Hugging Face.
