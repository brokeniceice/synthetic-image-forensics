# Environment

The recorded experiment environment is Linux, Python 3.10, PyTorch 1.13.1+cu117, torchvision 0.14.1+cu117, MMCV-full 1.4.7, and Transformers commit `cae78c46d658a8e496a815c2ee49b9b178fb9c9a`. The model uses BF16; the frozen RINE head runs in FP32. Use a CUDA GPU supporting BF16 and the matching CUDA toolkit for compiled extensions.

```bash
conda create -n image-forensics python=3.10 -y
conda activate image-forensics
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install mmcv-full==1.4.7 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
pip install -r requirements.txt
```

The original ROI base class is retained in `model/roi_extractor.py`; installing MMDetection is not required. Some CUDA/compiler combinations require compiling MMCV locally. Run all project commands from the repository root. No API key is required for the three task evaluators.

The listed versions match the recorded environment. A clean installation on a second machine has not yet been validated; the release checks use the existing experiment environment.
