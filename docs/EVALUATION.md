# Evaluation protocol

The three evaluators correspond to the final experimental asset tables. `configs/paper/protocol.json` records the model identities and unrounded expected results.

| Task | Paper route | Main metrics |
|---|---|---|
| Classification | Canonical fixed CLS question; C1; TRAIN center −9.190834884678093; batch 8 | Accuracy, Fake F1 |
| Localization | P1 decoder + NoRelative R1; known-Fake G1; 400 continuation tokens | Foreground IoU/F1 and foreground/background mIoU |
| Explanation | E03 autonomous G0; explanation question; 512 tokens | CSS, ROUGE-L F1 |

## Classification

Compute `(logit_fake - logit_real)` in the native BF16 tensor, then cast to FP32, matching the archived score exports. Fake means `margin > center`; the raw comparison uses zero. `scripts/fit_center.py` computes an arithmetic TRAIN mean without labels or threshold search. The released inference default is the frozen paper center, not a newly fitted value. Keep the recorded deterministic CLIP preprocessing, batch size and ordered manifest when reproducing scores.

## Localization

Suppress `[REAL]`, `[CLS]` and `[FAKE]` during the continuation; EOS and SEG remain free. If no natural SEG is emitted, remove terminal EOS/padding and append one SEG without regenerating text. Replay the same C1 evidence-conditioned sequence to extract the projected causal SEG predictor. Each query is decoded independently using corrected spatial embeddings and the frozen P1 SAM path. Union per-query logits at original image resolution, then threshold at zero.

Report per-image foreground IoU/F1 means separately from global pixel-pooled IoU/F1. The paper's foreground F1 values (for example 51.31% on SynthScars) use the global aggregation. Foreground/background mIoU averages the two classes. Empty prediction plus empty target scores one when the query route is valid; invalid-query examples contribute zero foreground overlap in the per-image means. Keep full dataset populations and empty ground truths.

## Explanation

Use the same C1 language path as E03, without a GT verdict prefix or control-token suppression. Spatial correction does not change language generation, so no spatial bundle is needed for text evaluation. Preserve generated verdict phrases when cleaning text; remove only structural special tokens.

CSS uses `sentence-transformers/paraphrase-MiniLM-L6-v2` revision `c9a2bfebc254878aee8c3aca9e6844d5bbb102d1`: CPU FP32 encoder, 128 tokens, attention-mask mean pooling, L2 normalization, cosine similarity, batch 32. ROUGE-L uses lowercase ASCII alphanumeric tokens, no stemming, LCS F1. Empty predictions score zero for both metrics and remain in the denominator.

Conditional G1 explanations and the separate AI-review experiment are not the autonomous explanation main table. No AI API scoring service, other-model evaluator or GPU job supervisor is included in the public evaluation interface.
