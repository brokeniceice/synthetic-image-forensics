# Training and selection

The source includes the training implementations and their selected configurations. Historical stage names are retained inside `training/` so that checkpoint topology and loss definitions remain traceable. Public evaluation has independent entrypoints in `scripts/`.

## Global classification and language branches

1. Train the RINE head on the shared frozen CLIP tower, seed 3407. The recorded recipe uses batch 128, AdamW, learning rate 1e-3, weight decay zero and one epoch. See `training/phase6b6_rine_training.py` and `model/rine_on_c1.py`.
2. Train P1 and C1 as separate branches from the same pre-P1 GLaMM initialization. P1 supplies the selected frozen SAM decoder; C1 supplies authenticity scores, explanations and language queries. P1 is not the parent of C1.
3. Use `configs/phase3a_p1.yaml` and `configs/phase6d3_c1_rine_conditioned_p1.yaml`. The backbone trainer is `training/phase2a_distributed_train.py`. C1 enables `--rine-conditioned-c1 --rine-checkpoint checkpoints/rine.pt`.

Both P1 and C1 use LoRA rank 8, alpha 16 and dropout 0.05 on q/v projections. The recipes run 10 epochs × 500 optimizer steps, learning rate 3e-4, seed 3407, BF16, frozen language/vision backbone, and the recorded trainable heads/embeddings. Loss weights are text 1, classification 1, mask BCE 2 and mask Dice 0.5. The selected C1 is epoch 5 / step 2500; selected P1 is epoch 7 / step 3500. Select by VAL loss, without TEST or external OOD selection.

## Dense evidence and spatial correction

The final inference system needs the selected dense forensic adapter, frozen P1 prompt/mask runtime, frozen evidential source heads/temperatures, C1-projected queries and spatial feature caches. The training helpers expect these prepared inputs; they do not regenerate every historical cache automatically.

The final spatial training has three stages:

1. Rectifier fitting with frozen utility/source/SAM/language paths.
2. Utility fitting with the rectifier frozen.
3. Joint utility/rectifier fitting, keeping the P1 SAM decoder frozen.

The shared rectifier implementation is `training/phase6e3_c1_native_staged.py`. The final utility/Joint loss variant is `training/c1_submission_ablation_train.py`, arm `NoRelative`, seed 3407: segmentation plus ranking, without relative loss. Its selected Joint checkpoint is epoch 8. Geometry, gated embedding arithmetic, BCE/Dice weighting and ranking code are retained in the associated helpers and model files.

`training/phase4f_prepare.py::extract_runtime` builds the frozen P1 decoder runtime. `scripts/package_spatial.py` assembles the selected inference components into one portable bundle.

## Reproduction scope

Inference and metric interfaces are portable and accept explicit files. The reference spatial training supervisors still expect their prepared cache layouts and integrity metadata. The frozen split manifests, trained checkpoints, evidential initialization assets and prepared spatial caches are separate research assets and are not included in Git. Consequently this snapshot is not a complete one-command from-scratch training release. The source and configurations specify the training design, and the checkpoint manifest fixes the exact final model.
