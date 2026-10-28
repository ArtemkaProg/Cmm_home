import json
import time

import numpy as np
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model(device="cpu")
print("model loaded")

enc = model.encode("What is the sum of 2 and 3?")
print("encode shape:", enc.shape)

ids = enc.tolist()[0]
print("input ids:", ids)

for i in range(10):
    start = time.time()
    logits = model.get_logits_from_input_ids(ids)
    elapsed = time.time() - start
    nxt = int(np.argmax(logits))
    ids.append(nxt)
    print(f"len(logits)={len(logits)} next={nxt} time={elapsed:.2f}s")

print("decoded:", repr(model.decode(ids)))

with open(model.get_path_to_vocab_file()) as f:
    vocab = json.load(f)
# print("vocab size:", len(vocab))
# print("first items:", list(vocab.items())[:5])

inv = {v: k for k, v in vocab.items()}
pvocab = {}
for k, v in vocab.items():
    if not k.isalnum() and k not in ('ĊĠ') and k not in (".!?,'"):
        continue
    p = k.replace('Ġ', ' ').replace('Ċ', '\n')
    pvocab[v] = p

print('\npvocab:')
print(repr(pvocab[374]))    # ' is'
print(repr(pvocab[220]))    # ' '
print(repr(pvocab[17]))     # '2'

print('\nvocab:')
print(repr(inv[374]))    # ' is'
print(repr(inv[220]))    # ' '
print(repr(inv[17]))     # '2'

print()
ids = model.encode('{"name": "fn_add_numbers"').tolist()[0]
print([pvocab.get(i) for i in ids])
print(ids)

print(f'vocab: {sum(1 for k in vocab if k.startswith("Ġ"))}')
print(f'pvocab: {sum(1 for k in pvocab.values() if 'Ġ' in k)}')

print(f'vocab: {[k for k in vocab if 'Ċ' in k][:5]}')
print(f'pvocab: {[k for k in pvocab.values() if 'Ġ' in k][:5]}')
