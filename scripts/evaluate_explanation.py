#!/usr/bin/env python3
"""E03 CSS and ROUGE-L F1 with pinned MiniLM, full population, empty=0."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forensics.text_metrics import clean, rouge_l

REVISION = 'c9a2bfebc254878aee8c3aca9e6844d5bbb102d1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True, help='JSONL with sample_id and reference; order must match predictions')
    p.add_argument('--encoder', default='sentence-transformers/paraphrase-MiniLM-L6-v2', help='Model ID or local snapshot at the pinned revision')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    import numpy as np
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    generated = [json.loads(s) for s in args.predictions.read_text().splitlines() if s.strip()]
    refs = [json.loads(s) for s in args.references.read_text().splitlines() if s.strip()]
    ids = [str(r['sample_id']) for r in generated]
    if not ids or len(ids) != len(set(ids)) or ids != [str(r['sample_id']) for r in refs]:
        raise ValueError('Prediction/reference populations or ordering differ')
    encoder_path = Path(args.encoder)
    if encoder_path.is_dir():
        identity = json.loads((encoder_path / 'identity.json').read_text())
        if identity['revision'] != REVISION:
            raise ValueError('CSS encoder revision differs from E03')
        import hashlib
        for file, record in identity['files'].items():
            if hashlib.sha256((encoder_path/file).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('CSS asset hash mismatch: ' + file)
        if not json.loads((encoder_path/'1_Pooling/config.json').read_text())['pooling_mode_mean_tokens']:
            raise ValueError('CSS pooling mismatch')
        if json.loads((encoder_path/'sentence_bert_config.json').read_text())['max_seq_length'] != 128:
            raise ValueError('CSS truncation mismatch')
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, revision=REVISION, local_files_only=encoder_path.is_dir())
    encoder = AutoModel.from_pretrained(args.encoder, revision=REVISION, local_files_only=encoder_path.is_dir()).eval()
    @torch.no_grad()
    def encode(strings):
        encoded, truncated = [], []
        for start in range(0,len(strings),32):
            batch = strings[start:start+32]
            lengths = [len(tokenizer(x,add_special_tokens=True,truncation=False)['input_ids']) for x in batch]
            x = tokenizer(batch,padding=True,truncation=True,max_length=128,return_tensors='pt')
            h = encoder(**x).last_hidden_state; mask = x['attention_mask'][:,:,None].to(h.dtype)
            pooled = (h*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
            encoded.append(torch.nn.functional.normalize(pooled,p=2,dim=1));truncated.extend(n>128 for n in lengths)
        return torch.cat(encoded), truncated
    answers = [clean(r['generated_text']) for r in generated]
    references = [r['reference'] for r in refs]
    a, ta = encode(answers); b, tb = encode(references)
    css = (a*b).sum(1).clamp(-1,1).tolist()
    scored = [dict(sample_id=sid, css=css[i] if answers[i] else 0.,
        rouge_l_f1=rouge_l(answers[i],references[i]), cleaned_prediction=answers[i],
        prediction_css_truncated=ta[i],reference_css_truncated=tb[i],empty=not bool(answers[i]),
        generation_truncated=bool(generated[i]['truncated'])) for i,sid in enumerate(ids)]
    result = dict(n=len(scored), css=float(np.mean([r['css'] for r in scored])),
        rouge_l_f1=float(np.mean([r['rouge_l_f1'] for r in scored])),empty=sum(r['empty'] for r in scored),
        truncated=sum(r['generation_truncated'] for r in scored),css_prediction_truncated=sum(ta),
        css_reference_truncated=sum(tb),css_revision=REVISION,css_max_tokens=128)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'scores.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored))
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
