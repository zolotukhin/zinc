// GPU test host only; no model is loaded.
// cc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -D__HIP_PLATFORM_AMD__=1 \
//   -Isrc/cuda -I/opt/rocm/include tools/validate_rocm_attention.c \
//   src/rocm/rocm_shim.c -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib \
//   -lamdhip64 -lhiprtc -lhipblas -lm -o /tmp/validate-rocm-attention
// ROCR_VISIBLE_DEVICES=<device> /tmp/validate-rocm-attention
// Checks tiled decode, causal prefill, speculative offsets, slot isolation,
// graph capture/update, guards, and sticky failures against independent double
// precision CPU math. Legal short cases also compare with the old GPU kernels.
#include "cuda_shim.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned hd, heads, kv_heads, seq, scale, sink; } Decode;
typedef struct { unsigned hd, heads, kv_heads, T, scale, sink, base; } Causal;
typedef struct { unsigned hd, heads, kv_heads, ctx, scale, sink; } Slots;
static unsigned cases;
static unsigned quantized;
static unsigned grouped;
static unsigned measure_precision;
static double precision_max_abs, precision_squared;
static size_t precision_values;
static CudaPipe *quantizer, *single_quantizer, *grouped_prefill, *grouped_prefill32;
static CudaPipe *blas_pipes[8];
typedef struct { float scale; signed char values[32]; } KvQ8;
_Static_assert(sizeof(KvQ8)==36,"Host KV block ABI");
typedef struct { unsigned dimensions, base, rows; } KvBatch;
typedef struct { unsigned hd,heads,kv_heads,rows,scale,sink,base,start,keys; } BlasPush;
static uint32_t rng = 0x19ab42e1;
static float random_value(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((int)(rng % 2049) - 1024) / 1024.0f;
}
static void fail(const char *what) {
    fprintf(stderr, "FAIL %s: %s\n", what, cuda_last_error());
    exit(1);
}
static void *allocate(size_t n) {
    void *p = malloc(n);
    if (!p) fail("host allocation");
    return p;
}
static char *source(void) {
    FILE *f = fopen("src/shaders/cuda/kernels.cu", "rb");
    if (!f || fseek(f, 0, SEEK_END)) fail("open kernel source");
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET)) fail("kernel source size");
    char *s = allocate((size_t)n + 1);
    if (fread(s, 1, (size_t)n, f) != (size_t)n) fail("read kernel source");
    fclose(f); s[n] = 0; return s;
}
static CudaBuf *upload(CudaCtx *ctx, const void *p, size_t n) {
    CudaBuf *b = cuda_create_buffer(ctx, n);
    if (!b) fail("device allocation");
    cuda_upload(ctx, b, p, n);
    if (cuda_execution_failed(ctx)) fail("upload");
    return b;
}
// Validate GPU quantization independently before using decoded values as the
// double-precision attention reference. Check guards and a nonzero write offset.
static void quantize(CudaCtx *ctx, CudaBuf **kb, CudaBuf **vb, float *k, float *v,
                     size_t elements, unsigned stride) {
    size_t blocks = elements / 32, prefix = stride / 32, bytes = (blocks + prefix + 1) * sizeof(KvQ8);
    unsigned char *guard = allocate(bytes), *result = allocate(bytes);
    memset(guard, 0xa5, bytes);
    CudaBuf *targets[] = {upload(ctx,guard,bytes),upload(ctx,guard,bytes)};
    CudaBuf *args[] = {*kb, targets[0], *vb, targets[1]};
    KvBatch pc = {stride,1,(unsigned)(elements/stride)};
    unsigned rows_per_grid=pc.rows<4096?pc.rows:4096;
    unsigned grid[] = {(stride+63)/64,rows_per_grid,(pc.rows+rows_per_grid-1)/rows_per_grid}, threads[] = {64,1,1};
    CudaCmd *cmd = cuda_begin_command(ctx);
    unsigned single[] = {stride,stride};
    cuda_dispatch(cmd,pc.rows == 1 ? single_quantizer : quantizer,grid,threads,args,4,
                  pc.rows == 1 ? (void*)single : (void*)&pc,pc.rows == 1 ? sizeof single : sizeof pc,0);
    cuda_commit_and_wait(cmd);
    if (cuda_execution_failed(ctx)) fail("KV quantization");
    for (unsigned side=0; side<2; side++) {
        float *original = side ? v : k;
        cuda_download(ctx,targets[side],result,bytes);
        for (size_t i=0;i<prefix*sizeof(KvQ8);i++) if(result[i]!=0xa5) fail("KV prefix guard");
        for (size_t i=(prefix+blocks)*sizeof(KvQ8);i<bytes;i++) if(result[i]!=0xa5) fail("KV suffix guard");
        KvQ8 *q=(KvQ8*)result+prefix;
        for (size_t b=0;b<blocks;b++) {
            float maximum=0;
            for(unsigned j=0;j<32;j++) maximum=fmaxf(maximum,fabsf(original[b*32+j]));
            if (!isfinite(q[b].scale) || fabsf(q[b].scale-maximum/127.0f)>1e-6f*fmaxf(maximum,1e-20f)) fail("KV scale");
            for(unsigned j=0;j<32;j++) {
                float decoded=q[b].scale*q[b].values[j];
                if (q[b].values[j]==-128 || fabsf(decoded-original[b*32+j])>maximum/254.0f+1e-6f*fmaxf(maximum,1e-20f)) fail("KV rounding bound");
                original[b*32+j]=decoded;
            }
        }
        // Strip the guarded offset for the attention launch. Its row values
        // still came from the production GPU writer, not a host quantizer.
        cuda_upload(ctx,targets[side],q,blocks*sizeof(KvQ8));
    }
    cuda_free_buffer(*kb);cuda_free_buffer(*vb);*kb=targets[0];*vb=targets[1];
    free(guard);free(result);
}
static void close_value(double expected, float actual) {
    if(measure_precision) {
        double error=fabs(expected-actual);
        if(!isfinite(actual)) fail("mixed-precision nonfinite output");
        precision_max_abs=fmax(precision_max_abs,error);precision_squared+=error*error;precision_values++;
        return;
    }
    if (!isfinite(actual) || fabs(expected - actual) > 0.0003 * (1 + fabs(expected))) {
        fprintf(stderr, "expected=%.12g actual=%.9g\n", expected, actual);
        fail("numerical reference");
    }
}
static void cpu_row(const float *q, const float *k, const float *v, const float *out,
                    unsigned hd, unsigned heads, unsigned kv_heads, unsigned seq,
                    double scale, const float *sinks) {
    double *scores = allocate((size_t)seq * sizeof(double));
    for (unsigned h = 0; h < heads; h++) {
        unsigned kh = h / (heads / kv_heads);
        double m = -INFINITY;
        for (unsigned i = 0; i < seq; i++) {
            double dot = 0;
            for (unsigned d = 0; d < hd; d++)
                dot += (double)q[(size_t)h * hd + d] * k[((size_t)i * kv_heads + kh) * hd + d];
            scores[i] = dot * scale;
            m = fmax(m, scores[i]);
        }
        if (!isnan(sinks[h])) m = fmax(m, sinks[h]);
        double sum = isnan(sinks[h]) ? 0 : exp(sinks[h] - m);
        for (unsigned i = 0; i < seq; i++) { scores[i] = exp(scores[i] - m); sum += scores[i]; }
        for (unsigned d = 0; d < hd; d++) {
            double acc = 0;
            for (unsigned i = 0; i < seq; i++)
                acc += scores[i] * v[((size_t)i * kv_heads + kh) * hd + d];
            close_value(acc / sum, out[(size_t)h * hd + d]);
        }
    }
    free(scores);
}
// Explicit mixed-precision contract: IEEE FP16 matrix inputs/probabilities,
// FP32 maxima, denominator and output accumulation. Keep the FP32 guard intact;
// measure the difference from the original FP32 inputs separately.
static void cpu_half_row(const float *q,const float *k,const float *v,const float *out,
                         unsigned hd,unsigned heads,unsigned kv_heads,unsigned seq,
                         double scale,const float *sinks) {
    double scores[1024];double *acc=allocate(hd*sizeof(double));
    for(unsigned h=0;h<heads;h++) {
        unsigned kh=h/(heads/kv_heads);double maximum=-3.4e38,sum=0;
        memset(acc,0,hd*sizeof(double));
        for(unsigned start=0;start<seq;start+=1024) {
            unsigned count=seq-start<1024?seq-start:1024;double m=maximum;
            for(unsigned j=0;j<count;j++) {
                double dot=0;
                for(unsigned d=0;d<hd;d++) dot+=(double)(float)(_Float16)q[(size_t)h*hd+d]*(float)(_Float16)k[((size_t)(start+j)*kv_heads+kh)*hd+d];
                scores[j]=dot*scale;m=fmax(m,scores[j]);
            }
            double rescale=exp(maximum-m);maximum=m;sum*=rescale;
            for(unsigned d=0;d<hd;d++) acc[d]*=rescale;
            for(unsigned j=0;j<count;j++) {
                double weight=exp(scores[j]-maximum);sum+=weight;
                double rounded=(float)(_Float16)weight;
                for(unsigned d=0;d<hd;d++) acc[d]+=rounded*(float)(_Float16)v[((size_t)(start+j)*kv_heads+kh)*hd+d];
            }
        }
        double factor=1;
        if(!isnan(sinks[h])) { double m=fmax(maximum,sinks[h]);factor=exp(maximum-m);sum=sum*factor+exp(sinks[h]-m); }
        for(unsigned d=0;d<hd;d++) close_value(acc[d]*factor/sum,out[(size_t)h*hd+d]);
    }
    free(acc);
}
static void blas_attention(CudaCtx *ctx, CudaCmd *cmd, CudaBuf **bufs, Causal c) {
    unsigned half=grouped==4;CudaPipe **pipes=blas_pipes+(half?4:0);
    BlasPush pc={c.hd,c.heads,c.kv_heads,c.T,c.scale,c.sink,c.base,0,1024};
    size_t rows=(size_t)c.heads*c.T, vectors=rows*c.hd*4, kv=(size_t)c.heads*1024*c.hd*4;
    CudaBuf *b[]={cuda_create_buffer(ctx,half?vectors/2:vectors),cuda_create_buffer(ctx,vectors),
        cuda_create_buffer(ctx,rows*8),cuda_create_buffer(ctx,rows*1024*4),
        cuda_create_buffer(ctx,half?kv/2:kv),cuda_create_buffer(ctx,half?kv/2:kv),
        cuda_create_buffer(ctx,half?rows*1024*2:4)};
    for(unsigned i=0;i<7;i++) if(!b[i]) fail("BLAS scratch allocation");
    unsigned block[]={256,1,1}, grid[]={((unsigned)rows*c.hd+255)/256,1,1};
    CudaBuf *qargs[]={bufs[0],b[0],b[1],b[2]};
    cuda_dispatch(cmd,pipes[0],grid,block,qargs,4,&pc,sizeof pc,0);
    int64_t qs=(int64_t)c.T*c.hd, ks=(int64_t)1024*c.hd, ss=(int64_t)c.T*1024;
    for(;pc.start<c.base+c.T;pc.start+=1024) {
        unsigned kg[]={((unsigned)c.heads*1024*c.hd+255)/256,1,1};
        CudaBuf *kargs[]={bufs[1],bufs[2],b[4],b[5]};
        cuda_dispatch(cmd,pipes[1],kg,block,kargs,4,&pc,sizeof pc,0);
        if(half) cuda_cublas_hgemm_strided_batched(ctx,1,0,1024,c.T,c.hd,b[4],0,c.hd,ks,b[0],0,c.hd,qs,b[3],0,1024,ss,c.heads,0);
        else cuda_cublas_sgemm_strided_batched(ctx,1,0,1024,c.T,c.hd,b[4],0,c.hd,ks,b[0],0,c.hd,qs,b[3],0,1024,ss,c.heads,0);
        unsigned sg[]={c.heads,c.T,1};CudaBuf *sargs[]={b[3],b[1],b[2],b[6]};
        cuda_dispatch(cmd,pipes[2],sg,block,sargs,half?4:3,&pc,sizeof pc,0);
        if(half) cuda_cublas_hgemm_strided_batched(ctx,0,0,c.hd,c.T,1024,b[5],0,c.hd,ks,b[6],0,1024,ss,b[1],0,c.hd,qs,c.heads,1);
        else cuda_cublas_sgemm_strided_batched(ctx,0,0,c.hd,c.T,1024,b[5],0,c.hd,ks,b[3],0,1024,ss,b[1],0,c.hd,qs,c.heads,1);
    }
    CudaBuf *args[]={b[1],b[2],bufs[3],bufs[4]};
    cuda_dispatch(cmd,pipes[3],grid,block,args,4,&pc,sizeof pc,0);
    float ready;cuda_download(ctx,bufs[4],&ready,sizeof ready);
    if(cuda_execution_failed(ctx)) fail("BLAS attention");
    for(unsigned i=0;i<7;i++) cuda_free_buffer(b[i]);
}
// mode: 0 decode, 1 causal prefill, 2 independently positioned/permuted slots.
static void test(CudaCtx *ctx, CudaPipe **pipes, unsigned seq, unsigned hd,
                 unsigned T, unsigned mode, unsigned stress, unsigned graph) {
    const unsigned heads = 4, kv_heads = 2, ns = mode == 2 ? 3 : 1;
    const unsigned sink_offset = 3, stride = hd * kv_heads;
    unsigned positions[3] = {seq - 1, 0, seq > 1025 ? 1024 : seq / 2};
    unsigned slots[3] = {2, 0, 1};
    unsigned rows = mode == 0 ? 1 : mode == 1 ? T : ns;
    size_t qn = (size_t)rows * heads * hd, kn = (size_t)ns * seq * stride;
    float *q = allocate(qn * 4), *k = allocate(kn * 4), *v = allocate(kn * 4);
    float sinks[7], *out = allocate((qn + 8) * 4), *reference = allocate((qn + 8) * 4);
    for (size_t i = 0; i < qn; i++) q[i] = random_value() * (stress ? 40 : 1);
    for (size_t i = 0; i < kn; i++) { k[i] = random_value(); v[i] = random_value(); }
    if (quantized && kn >= 96) {
        for(unsigned j=0;j<32;j++) { k[j]=v[j]=0.0f;k[32+j]*=1e-20f;v[32+j]*=1e-20f;k[64+j]*=1e-3f;v[64+j]*=1e-3f; }
        k[64]=10000.0f;v[64]=-10000.0f;
    }
    for (unsigned i = 0; i < 7; i++) sinks[i] = i % 2 ? NAN : (stress ? 800.0f : -0.25f);
    for (size_t i = 0; i < qn + 8; i++) out[i] = reference[i] = -999.25f;
    CudaBuf *bufs[] = {upload(ctx,q,qn*4), upload(ctx,k,kn*4), upload(ctx,v,kn*4),
        upload(ctx,sinks,sizeof sinks), upload(ctx,out,(qn+8)*4),
        upload(ctx,positions,sizeof positions), upload(ctx,slots,sizeof slots)};
    if (quantized) quantize(ctx,&bufs[1],&bufs[2],k,v,kn,stride);
    Decode dec = {hd,heads,kv_heads,seq,0,sink_offset};
    Causal causal = {hd,heads,kv_heads,T,0,sink_offset,seq-T};
    Slots slot = {hd,heads,kv_heads,seq,0,sink_offset};
    float custom_scale = 0.375f;
    if (stress) { memcpy(&dec.scale,&custom_scale,4); causal.scale=slot.scale=dec.scale; }
    const void *push = mode == 0 ? (void*)&dec : mode == 1 ? (void*)&causal : (void*)&slot;
    size_t push_size = mode == 1 ? sizeof causal : sizeof dec;
    unsigned grid[] = {heads,rows,(hd+255)/256}, block[] = {256,1,1};
    if (grouped && mode == 1) { unsigned tile=grouped==2?32:8;grid[1]=(rows+tile-1)/tile;grid[2]=1; }
    CudaGraph *g = graph ? cuda_graph_create() : NULL;
    unsigned launches = graph ? 2 : 1;
    for (unsigned launch = 0; launch < launches; launch++) {
        if (launch > 0) {
            // Verify changed graph parameters, not just an identical replay.
            custom_scale = 0.625f;
            memcpy(&dec.scale,&custom_scale,4);
            causal.scale=slot.scale=dec.scale;
        }
        if (graph && !cuda_graph_begin(ctx)) fail("graph begin");
        CudaCmd *cmd = cuda_begin_command(ctx);
        if (!cmd) fail("command begin");
        if(grouped>=3 && mode==1) blas_attention(ctx,cmd,bufs,causal);
        else cuda_dispatch(cmd,grouped && mode == 1 ? (grouped==2?grouped_prefill32:grouped_prefill) : pipes[mode],grid,block,bufs,
                           mode == 2 ? 7 : 5,push,push_size,0);
        if (graph) {
            cuda_release_completed(cmd);
            if (!cuda_graph_end_launch(ctx,g)) fail("graph launch/update");
        } else cuda_commit_and_wait(cmd);
        if (cuda_execution_failed(ctx)) fail("tiled launch");
    }
    cuda_download(ctx,bufs[4],out,(qn+8)*4);
    if (cuda_execution_failed(ctx)) fail("download");
    double scale = dec.scale ? custom_scale : 1 / sqrt(hd);
    for (unsigned row = 0; row < rows; row++) {
        // Check every decode/slot row; sample early/middle/last causal queries.
        if (mode == 1 && row != 0 && row != rows/2 && row != rows-1) continue;
        unsigned length = mode == 0 ? seq : mode == 1 ? causal.base + row + 1 : positions[row] + 1;
        size_t off = mode == 2 ? (size_t)slots[row] * seq * stride : 0;
        measure_precision=grouped==4;
        cpu_row(q+(size_t)row*heads*hd,k+off,v+off,out+(size_t)row*heads*hd,
                hd,heads,kv_heads,length,scale,sinks+sink_offset);
        measure_precision=0;
        if(grouped==4) cpu_half_row(q+(size_t)row*heads*hd,k+off,v+off,out+(size_t)row*heads*hd,
                hd,heads,kv_heads,length,scale,sinks+sink_offset);
    }
    for (unsigned i = 0; i < 8; i++) if (out[qn+i] != -999.25f) fail("output guard");
    if (!quantized && seq <= 8192) {
        unsigned old_grid[] = {heads,rows,1};
        unsigned shared = (seq + (mode == 1 ? hd : 0)) * 4;
        CudaCmd *cmd = cuda_begin_command(ctx);
        cuda_dispatch(cmd,pipes[mode+3],old_grid,block,bufs,mode == 2 ? 7 : 5,push,push_size,shared);
        cuda_commit_and_wait(cmd);
        cuda_download(ctx,bufs[4],reference,(qn+8)*4);
        if (cuda_execution_failed(ctx)) fail("original kernel");
        for (size_t i = 0; i < qn; i++) close_value(reference[i],out[i]);
    }
    if (g) cuda_graph_free(g);
    for (unsigned i = 0; i < 7; i++) cuda_free_buffer(bufs[i]);
    free(q); free(k); free(v); free(out); free(reference);
    printf("PASS mode=%u seq=%u hd=%u rows=%u stress=%u graph=%u\n",mode,seq,hd,rows,stress,graph);
    fflush(stdout); cases++;
}
static void failures(void) {
    const char *s = "extern \"C\" __global__ void store(float* x) { x[0]=1.0f; }";
    for (unsigned mode = 0; mode < 3; mode++) {
        CudaCtx *ctx = cuda_init(0);
        if (!ctx) fail("failure-test init");
        CudaPipe *p = cuda_create_pipeline(ctx,s,"store",NULL,0);
        if (!p) fail("failure-test kernel compile");
        float initial = 0;
        CudaBuf *b = upload(ctx,&initial,4);
        if(mode==2) cuda_cublas_sgemm_strided_batched(ctx,0,0,1,1,1,b,0,0,1,b,0,1,1,b,0,1,1,1,0);
        else {
            unsigned grid[] = {1,1,1}, block[] = {mode ? cuda_pipeline_max_threads(p)+1 : 1,1,1};
            CudaCmd *cmd = cuda_begin_command(ctx);
            cuda_dispatch(cmd,p,grid,block,&b,1,NULL,0,mode ? 0 : cuda_max_shared_mem_per_block(ctx)+4);
            cuda_commit_and_wait(cmd);
        }
        if (!cuda_execution_failed(ctx) || cuda_begin_command(ctx) != NULL) fail("execution failure not sticky");
        cuda_free_buffer(b); cuda_free_pipeline(p); cuda_destroy(ctx);
        printf("PASS sticky %s failure\n",mode==2 ? "BLAS" : mode ? "launch" : "shared-memory");
        cases++;
    }
}
int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && strcmp(argv[1],"--q8"))) fail("Use optional --q8");
    quantized = argc == 2;
    char *s = source();
    CudaCtx *ctx = cuda_init(0);
    if (!ctx) fail("GPU init");
    const char *names[] = {quantized ? "attention_tiled_q8" : "attention_tiled",
        quantized ? "attention_causal_tiled_q8" : "attention_causal_tiled","attention_batched_seq_tiled",
        "naive_attention","attention_causal_batched_v2","naive_attention_batched_seq"};
    CudaPipe *pipes[6];
    for (unsigned i = 0; i < 6; i++) {
        pipes[i] = cuda_create_pipeline(ctx,s,names[i],NULL,0);
        if (!pipes[i]) fail("kernel compile");
        if (i < 3 && cuda_pipeline_shared_mem(pipes[i]) > 8192) fail("unbounded shared memory");
        if (i < 3) printf("%s: %u static shared bytes\n", names[i], cuda_pipeline_shared_mem(pipes[i]));
    }
    if (quantized) {
        quantizer=cuda_create_pipeline(ctx,s,"kv_cache_write_batched_q8",NULL,0);
        single_quantizer=cuda_create_pipeline(ctx,s,"kv_cache_write_q8",NULL,0);
        grouped_prefill=cuda_create_pipeline(ctx,s,"attention_causal_q8_prefill",NULL,0);
        grouped_prefill32=cuda_create_pipeline(ctx,s,"attention_causal_q8_prefill32",NULL,0);
        if (!quantizer || !single_quantizer || !grouped_prefill || !grouped_prefill32) fail("quantizer/grouped prefill compile");
        if (cuda_pipeline_shared_mem(grouped_prefill)>cuda_max_shared_mem_per_block(ctx)) fail("prefill shared-memory bound");
        const char *bn[]={"q8_attention_prepare_q","q8_attention_prepare_kv","q8_attention_softmax","q8_attention_finish",
            "q8_attention_prepare_q_f16","q8_attention_prepare_kv_f16","q8_attention_softmax_f16","q8_attention_finish"};
        for(unsigned i=0;i<8;i++) { blas_pipes[i]=cuda_create_pipeline(ctx,s,bn[i],NULL,0);if(!blas_pipes[i]) fail("BLAS kernel compile"); }
    }
    const unsigned lengths[] = {1,1023,1024,1025,4097,8192,8193,16080,16384,16385,32768};
    for (unsigned i = 0; i < sizeof lengths/sizeof lengths[0]; i++)
        test(ctx,pipes,lengths[i],256,1,0,0,0);
    for (unsigned mode = 0; mode < (quantized ? 2u : 3u); mode++) {
        test(ctx,pipes,8191,128,3,mode,1,0);
        test(ctx,pipes,32768,quantized ? 512 : 513,4,mode,0,1);
        test(ctx,pipes,16385,64,3,mode,1,0);
    }
    test(ctx,pipes,1025,128,1025,1,0,0);
    test(ctx,pipes,16080,256,16080,1,0,0);
    test(ctx,pipes,32768,128,4096,1,0,0);
    if (quantized) {
        test(ctx,pipes,65536,256,1,0,0,0);
        test(ctx,pipes,131072,256,3,1,0,1);
        test(ctx,pipes,262144,256,1,0,0,0);
        grouped=1;
        const unsigned tails[]={1,7,8,9,31,32,33,1025,4096};
        for(unsigned i=0;i<sizeof tails/sizeof tails[0];i++) test(ctx,pipes,32768,256,tails[i],1,i%2,i%3==0);
        test(ctx,pipes,262144,128,9,1,1,1);
        test(ctx,pipes,1025,64,1025,1,0,0);
        grouped=2;
        const unsigned tails32[]={1,7,8,9,31,32,33,63,64,65,1025,4096};
        for(unsigned i=0;i<sizeof tails32/sizeof tails32[0];i++) test(ctx,pipes,32768,256,tails32[i],1,i%2,i%3==0);
        test(ctx,pipes,262144,256,33,1,0,1);
        grouped=3;
        const unsigned blen[]={1,1023,1024,1025,8193,32768,262144};
        for(unsigned i=0;i<sizeof blen/sizeof blen[0];i++) test(ctx,pipes,blen[i],256,blen[i]<9?1:9,1,i%2,0);
        test(ctx,pipes,32768,128,257,1,0,0);
        test(ctx,pipes,32768,256,4096,1,0,0);
        grouped=4;
        for(unsigned i=0;i<sizeof blen/sizeof blen[0];i++) test(ctx,pipes,blen[i],256,blen[i]<9?1:9,1,i%2,0);
        test(ctx,pipes,32768,128,257,1,0,0);
        test(ctx,pipes,32768,256,4096,1,0,0);
        cuda_free_pipeline(quantizer);
        cuda_free_pipeline(single_quantizer);
        cuda_free_pipeline(grouped_prefill);
        cuda_free_pipeline(grouped_prefill32);
        for(unsigned i=0;i<8;i++) cuda_free_pipeline(blas_pipes[i]);
    }
    for (unsigned i = 0; i < 6; i++) cuda_free_pipeline(pipes[i]);
    cuda_destroy(ctx); free(s);
    failures();
    printf("PASS %u attention/error cases\n",cases);
    if(precision_values) printf("Mixed-vs-FP32: max_abs=%.9g rmse=%.9g values=%zu\n",precision_max_abs,sqrt(precision_squared/precision_values),precision_values);
    return 0;
}
