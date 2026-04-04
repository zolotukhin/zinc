import struct

f = open("/root/models/gemma-4-31B-it-Q4_K_M.gguf", "rb")
f.read(4)
struct.unpack("<I", f.read(4))
n_t = struct.unpack("<Q", f.read(8))[0]
n_m = struct.unpack("<Q", f.read(8))[0]

def rs(f):
    l = struct.unpack("<Q", f.read(8))[0]
    return f.read(l).decode("utf-8", errors="replace")

def rv(f, vt=None):
    if vt is None:
        vt = struct.unpack("<I", f.read(4))[0]
    if vt == 8: return rs(f)
    elif vt == 9:
        at = struct.unpack("<I", f.read(4))[0]
        al = struct.unpack("<Q", f.read(8))[0]
        return [rv(f, at) for _ in range(al)]
    else:
        f.read({0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(vt, 1))
        return None

toks = None
for _ in range(n_m):
    k = rs(f)
    v = rv(f)
    if k == "tokenizer.ggml.tokens":
        toks = v
f.close()

spm = chr(0x2581)  # SentencePiece word boundary
gpt2 = chr(0x0120)  # GPT-2 space

for word in ["is", "the", "What", "of", "in"]:
    for prefix, name in [(spm, "SPM"), (gpt2, "GPT2"), (" ", "RAW")]:
        token_text = prefix + word
        found = False
        for i, t in enumerate(toks):
            if t == token_text:
                print(f"{name:5s} {repr(token_text):15s} -> {i}")
                found = True
                break
        if not found:
            print(f"{name:5s} {repr(token_text):15s} -> NOT FOUND")
    print()
