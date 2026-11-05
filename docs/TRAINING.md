# Training

Final inference uses the published Hugging Face package directly. The training code is provided separately for reproducing the training design; frozen training manifests, source-head initialization and prepared spatial caches are additional inputs.

## RINE

`python -m training.rine --initial-head PATH` trains RINE on the shared frozen CLIP tower using the recorded head initialization. The recipe is **torch.optim.Adam**, batch 128, learning rate 1e-3, zero weight decay, one epoch and seed 3407. The CLIP path and TRAIN/VAL manifests are explicit CLI arguments. Unified manifests use Fake=1; the loader translates to the native RINE label convention.

The initialized-head asset is fixed by SHA-256. Unrelated baseline comparisons and external source-repository audits are not required by this trainer.

## P1 decoder and C1

P1 and C1 start from the same pre-adaptation GLaMM initialization as separate training branches. The P1 recipe supplies the selected decoder. C1 additionally trains with the frozen RINE condition.

```bash
deepspeed --include localhost:0 train.py \
  --config configs/p1_decoder.yaml --mode train

deepspeed --include localhost:0 train.py \
  --config configs/c1.yaml --mode train
```

`training/backbone.py` defines tokenizer/model construction, LoRA and optimizer groups; `training/detector.py` preserves the balanced sampler, distributed loss normalization and VAL-loss selection. The C1 recipe enables RINE automatically. An exact pre-adaptation module checkpoint can be supplied with `--initial-checkpoint`; this is distinct from `--resume`.

Both recipes use LoRA rank 8, alpha 16, dropout 0.05, q/v projections, 10 epochs × 500 optimizer steps, learning rate 3e-4, BF16 and seed 3407. Loss weights are text 1, classification 1, mask BCE 2 and mask Dice 0.5. C1 is selected at epoch 5 / step 2500; P1 at epoch 7 / step 3500.

## Spatial correction

```bash
python -m training.spatial rectifier --config configs/spatial.yaml
python -m training.spatial utility --config configs/spatial.yaml
python -m training.spatial joint --config configs/spatial.yaml
```

The final trainer exposes only the selected NoRelative design. Rectifier fitting is followed by utility fitting and then Joint fitting of utility and rectifier. The P1 prompt/mask decoder, dense forensic source and evidential source heads stay frozen. Utility and Joint stages use segmentation plus ranking, without relative loss. The selected Joint checkpoint is epoch 8.

The original sampling order, eligible C1 query population, batch filtering, optimizer settings, geometry, gate interpolation, SAM precision and mask losses are retained. Prepared inputs are supplied through `configs/spatial.yaml`: spatial features, C1 queries, utility features, DEV records, evidence heads and temperatures. The helper modules contain only data loading, final utility/ranking and the selected spatial computations.

The checkpoint preparation utilities in `training/` serve maintainers and training reproduction. End users do not invoke them when loading the released model.
