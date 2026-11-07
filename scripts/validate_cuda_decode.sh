#!/usr/bin/env bash
# CUDA decode correctness regression harness for qwen35-9b on the <worker> NVIDIA box.
#
# Validates ZINC's forward_cuda (src/compute/forward_cuda.zig) token-for-token vs
# llama.cpp CUDA across three independent angles, so perf changes (fast matvecs,
# the async stream/event ring, fused kernels) can be checked for correctness
# regressions with one command:
#   (1) single-token argmax over diverse token ids
#   (2) multi-prompt greedy autoregressive generation (exercises pos>0 RoPE,
#       multi-entry attention, SSM conv-ring/recurrent-state carry)
#   (3) full-vocab logit-vector numerical fidelity (Pearson r, RMS, top-k overlap)
#
# Greedy decoding is deterministic, so a CORRECT forward matches llama.cpp argmax
# until a near-tie (logit gap < ~0.2) where ZINC's ~0.06-RMS fp drift can flip the
# pick — that is expected, not a regression. Treat a FULL-match drop or a fidelity
# collapse (r below ~0.99, top-5 overlap < 5) as a real regression.
#
# Run on the box, from the zinc repo, AFTER a build. Provide the target CUDA
# device through ZINC_GPU; keep machine-specific UUIDs in .env or the shell
# environment.
#   ZINC_GPU ZINC_MODEL LLAMA_CPP ZIG  - see defaults below.
set -u
GPU=${ZINC_GPU:-}
if [ -z "$GPU" ]; then
  echo "missing ZINC_GPU; set it to the CUDA_VISIBLE_DEVICES selector for this host" >&2
  exit 2
fi
M=${ZINC_MODEL:-$HOME/workspace/Qwen3.5-9B-Q4_K_M.gguf}
LCPP=${LLAMA_CPP:-$HOME/workspace/llama.cpp}
ZIG=${ZIG:-./scripts/zig.sh}
T=/tmp/zinc_validate
mkdir -p "$T"
export CUDA_VISIBLE_DEVICES=$GPU
LFLAGS="-I $LCPP/include -I $LCPP/ggml/include -L $LCPP/build/bin -lllama -lggml-base -Wl,-rpath,$LCPP/build/bin"

say(){ printf '\n=== %s ===\n' "$1"; }

# --- build llama.cpp reference tools (id->argmax, text->greedy, id->logits) ----
build_refs(){
  cat > "$T/eval.cpp" <<'CPP'
#include "llama.h"
#include <cstdio>
#include <cstdlib>
int main(int c,char**a){const char*mp=a[1];int tok=atoi(a[2]);llama_backend_init();
 llama_model_params m=llama_model_default_params();m.n_gpu_layers=999;
 llama_model*md=llama_model_load_from_file(mp,m);const llama_vocab*v=llama_model_get_vocab(md);
 llama_context_params cp=llama_context_default_params();cp.n_ctx=512;llama_context*ctx=llama_init_from_model(md,cp);
 int nv=llama_vocab_n_tokens(v);llama_batch b=llama_batch_init(1,0,1);
 b.n_tokens=1;b.token[0]=tok;b.pos[0]=0;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;llama_decode(ctx,b);
 const float*lg=llama_get_logits_ith(ctx,0);int bi=0;float bm=lg[0];for(int i=1;i<nv;i++)if(lg[i]>bm){bm=lg[i];bi=i;}
 printf("%d\n",bi);return 0;}
CPP
  cat > "$T/gen.cpp" <<'CPP'
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
static int am(const float*v,int n){int b=0;float m=v[0];for(int i=1;i<n;i++)if(v[i]>m){m=v[i];b=i;}return b;}
int main(int c,char**a){const char*mp=a[1];const char*p=a[2];int ng=c>3?atoi(a[3]):16;llama_backend_init();
 llama_model_params m=llama_model_default_params();m.n_gpu_layers=999;
 llama_model*md=llama_model_load_from_file(mp,m);const llama_vocab*v=llama_model_get_vocab(md);
 llama_context_params cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=512;llama_context*ctx=llama_init_from_model(md,cp);
 int nv=llama_vocab_n_tokens(v);std::vector<llama_token>t(256);
 int n=llama_tokenize(v,p,strlen(p),t.data(),t.size(),true,true);t.resize(n);
 printf("PROMPT_IDS:");for(int i=0;i<n;i++)printf("%s%d",i?",":"",t[i]);printf("\n");
 llama_batch b=llama_batch_init(512,0,1);b.n_tokens=n;
 for(int i=0;i<n;i++){b.token[i]=t[i];b.pos[i]=i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=(i==n-1);}
 llama_decode(ctx,b);int pos=n;const float*lg=llama_get_logits_ith(ctx,n-1);int tk=am(lg,nv);
 printf("GEN_IDS:%d",tk);
 for(int g=1;g<ng;g++){b.n_tokens=1;b.token[0]=tk;b.pos[0]=pos++;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;
  llama_decode(ctx,b);lg=llama_get_logits_ith(ctx,0);tk=am(lg,nv);printf(",%d",tk);}printf("\n");return 0;}
CPP
  cat > "$T/logitdump.cpp" <<'CPP'
#include "llama.h"
#include <cstdio>
#include <cstdlib>
int main(int c,char**a){const char*mp=a[1];int tok=atoi(a[2]);const char*o=a[3];llama_backend_init();
 llama_model_params m=llama_model_default_params();m.n_gpu_layers=999;
 llama_model*md=llama_model_load_from_file(mp,m);const llama_vocab*v=llama_model_get_vocab(md);
 llama_context_params cp=llama_context_default_params();cp.n_ctx=512;llama_context*ctx=llama_init_from_model(md,cp);
 int nv=llama_vocab_n_tokens(v);llama_batch b=llama_batch_init(1,0,1);
 b.n_tokens=1;b.token[0]=tok;b.pos[0]=0;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;llama_decode(ctx,b);
 const float*lg=llama_get_logits_ith(ctx,0);FILE*f=fopen(o,"wb");fwrite(lg,sizeof(float),nv,f);fclose(f);
 printf("%d\n",nv);return 0;}
CPP
  for x in eval gen logitdump; do
    [ -x "$T/$x" ] && continue
    g++ -O2 "$T/$x.cpp" $LFLAGS -o "$T/$x" 2>"$T/$x.build.log" || { echo "BUILD FAIL $x"; cat "$T/$x.build.log"; exit 1; }
  done
}

say "building references + ZINC cuda-dbg"
build_refs
"$ZIG" build cuda-dbg -Dbackend=cuda -Dshaders=false -- 100 "$M" >/dev/null 2>&1 || true
ZBIN=$(ls -t .zig-cache/o/*/cuda-dbg 2>/dev/null | head -1)
[ -x "$ZBIN" ] || { echo "no cuda-dbg binary — build failed"; exit 1; }
echo "zinc cuda-dbg: $ZBIN"
zgen(){ "$ZBIN" gen "$1" "$2" "$M" 2>&1 | awk -F: '/GEN_IDS/{print $2}'; }

fail=0

say "(1) single-token argmax vs llama.cpp"
for t in 0 1 42 100 500 1000 5000 9999 50000; do
  z=$("$ZBIN" gen "$t" 1 "$M" 2>&1 | awk -F: '/GEN_IDS/{print $2}')
  l=$("$T/eval" "$M" "$t" 2>/dev/null)
  [ -n "$z" ] && [ "$z" = "$l" ] && printf "  tok %-6s OK (%s)\n" "$t" "$z" || { printf "  tok %-6s DIFF zinc=%s llama=%s\n" "$t" "$z" "$l"; fail=1; }
done

say "(2) multi-prompt greedy generation (24 tokens) vs llama.cpp"
while IFS= read -r p; do [ -z "$p" ] && continue
  ref=$("$T/gen" "$M" "$p" 24 2>/dev/null)
  pids=$(printf '%s\n' "$ref" | awk -F: '/PROMPT_IDS/{print $2}')
  lg=$(printf '%s\n' "$ref" | awk -F: '/GEN_IDS/{print $2}')
  zg=$(zgen "$pids" 24)
  [ -n "$zg" ] && [ "$lg" = "$zg" ] && printf "  OK   \"%s\"\n" "$p" || { printf "  DIFF \"%s\"\n   llama:%s\n   zinc: %s\n" "$p" "$lg" "$zg"; fail=1; }
done <<'PROMPTS'
The capital of France is
def fibonacci(n):
Once upon a time
Water is made of hydrogen and
import numpy as np
PROMPTS

say "(3) full-vocab logit fidelity (token 100)"
"$ZBIN" logits 100 "$T/zinc_logits.bin" "$M" >/dev/null 2>&1
"$T/logitdump" "$M" 100 "$T/llama_logits.bin" >/dev/null 2>&1
python3 - "$T/zinc_logits.bin" "$T/llama_logits.bin" <<'PY'
import numpy as np, sys
z=np.fromfile(sys.argv[1],dtype=np.float32); l=np.fromfile(sys.argv[2],dtype=np.float32)
n=min(z.size,l.size); z,l=z[:n],l[:n]
r=np.corrcoef(z,l)[0,1]; rms=float(np.sqrt(((z-l)**2).mean()))
t5=len(set(np.argpartition(z,-5)[-5:].tolist())&set(np.argpartition(l,-5)[-5:].tolist()))
ok = (z.argmax()==l.argmax()) and r>0.99 and t5==5
print("  argmax %d vs %d | pearson r=%.5f | rms=%.4f | top5 %d/5 -> %s"%(z.argmax(),l.argmax(),r,rms,t5,"OK" if ok else "REGRESSION"))
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && fail=1

say "RESULT"
[ $fail -eq 0 ] && echo "  ALL CHECKS PASS — forward_cuda is token-correct vs llama.cpp" || echo "  REGRESSION DETECTED — see DIFF lines above"
exit $fail
