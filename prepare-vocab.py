#!/usr/bin/python

from struct import pack

I_TO_TOKEN = {}
sorted = [] # must be already sorted
lines = open("rwkv_vocab_v20230422.txt", "r", encoding="utf-8").readlines()
for l in lines:
    idx = int(l[:l.index(' ')])
    x = eval(l[l.index(' '):l.rindex(' ')])
    x = x.encode("utf-8") if isinstance(x, str) else x
    assert isinstance(x, bytes)
    assert len(x) == int(l[l.rindex(' '):])
    sorted += [x]
    I_TO_TOKEN[idx] = x

out = open("rwkv_vocab.bin", "wb")
out.write(pack("<I", len(I_TO_TOKEN)))
for token_index, token_string in I_TO_TOKEN.items():
    out.write(pack("<I", len(token_string)))
    out.write(token_string)
    out.write(pack("<H", token_index))
