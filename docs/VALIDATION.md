# Initial release checks

The public evaluators were checked against the frozen predictions and masks used by the experimental assets, without regenerating test predictions.

- **Classification:** all 204,424 internal/OOD examples across Internal2208, GenImage, AIGI-Holmes and LOKI. Centered accuracy and F1 reproduce the frozen unrounded results to 1e-12. Recomputing the mean of the original 17,672 TRAIN margins gives exactly −9.190834884678093.
- **Localization:** all 5,089 final NoRelative masks across SynthScars, LOKI, X-AIGD and PAL4VST were scored against their saved GT masks. Mean/global foreground IoU and F1 and mean/global foreground/background mIoU reproduce the frozen values to 1e-12.
- **Explanation:** all 1,229 original E03 G0 answers were rescored with the pinned local MiniLM asset. ROUGE-L F1 reproduces the frozen values exactly. Mean CSS differs by less than 8e-8 in FP32 and reproduces the paper's percentage values at two decimals. Empty/truncation counts match.
- **Inference implementation:** the extracted G1 generation/replay and native spatial computation match the archived evaluation functions at the Python AST level, apart from portable imports and the explicit bundle check. Spatial utility/gating helpers preserve the original computation. C1/RINE/base-model identities and final spatial checkpoint linkage are checked before inference.
- **Packaging:** the selected spatial checkpoints load strictly into their model classes and produce a portable bundle. Core inference and training modules import successfully from an export containing only the staged release files in the recorded environment. All Python files compile; local machine paths and credential patterns were checked before publication.

A fresh full GPU inference sweep and a clean installation on a separate machine have not been run for this code export. Exact tensor values can depend on the recorded CUDA/software stack and batch ordering. Weights, frozen split manifests and spatial training caches remain separate assets; see the checkpoint and training documentation.
