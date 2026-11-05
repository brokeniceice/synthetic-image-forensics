"""Exact E03 text cleaning and ROUGE-L; MiniLM pooling is in the CLI."""
import re


def clean(text):
    # Keep substantive verdict words; remove structural markers only.
    text = re.sub(r'</?(?:s|p|im_start|im_end)>|\[(?:CLS|SEG|END|REAL|FAKE)\]|<\|(?:im_start|im_end|endoftext)\|>', ' ', text)
    return ' '.join(text.split())


def tokens(s):
    return re.findall(r'[a-z0-9]+', s.lower())


def rouge_l(a, b):
    a, b = tokens(a), tokens(b)
    if not a or not b:
        return 0.
    previous = [0] * (len(b)+1)
    for x in a:
        current = [0]
        for j, y in enumerate(b):
            current.append(previous[j]+1 if x == y else max(previous[j+1], current[-1]))
        previous = current
    return 2*previous[-1]/(len(a)+len(b))
