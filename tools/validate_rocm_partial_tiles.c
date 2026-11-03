// Remote GPU host only. Validate Q5_1 grouped and Q8_0 dense WMMA against
// native GGML dequantization, including Gemma's 704/2112-column tails.
// Build like validate_rocm_attention.c, adding ggml/include, -lggml-base and
// its library search path. ZINC_REFERENCE_ONLY builds the CPU fixture checks.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef ZINC_REFERENCE_ONLY
#include "cuda_shim.h"
#include "ggml.h"
#endif

typedef struct { unsigned M,K,T,base,expert_stride,dst_tok_stride,route_stride; } Group;
_Static_assert(sizeof(Group)==28,"Grouped Q5_1 push ABI");
static unsigned cases;
static void fail(const char *what) {
    fprintf(stderr,"FAIL %s\n",what); exit(1);
}
static void *allocate(size_t n) {
    void *p=malloc(n); if(!p)fail("host allocation"); return p;
}
static void put_half(unsigned char *p,float value) {
    _Float16 h=(_Float16)value; memcpy(p,&h,2);
}
static float half(const unsigned char *p) {
    _Float16 h; memcpy(&h,p,2); return (float)h;
}
static unsigned code(unsigned e,unsigned row,unsigned block,unsigned k) {
    return (e*11+row*3+block+k*5)%32;
}
static void make_weights(unsigned char *p,unsigned E,unsigned M,unsigned K,unsigned base) {
    const unsigned blocks=K/32;
    memset(p,0xa5,base+(size_t)E*M*blocks*24);
    for(unsigned e=0;e<E;e++)for(unsigned row=0;row<M;row++)for(unsigned b=0;b<blocks;b++) {
        unsigned char *w=p+base+((size_t)e*M+row)*blocks*24+b*24;
        // Make the production tail decisive: an implementation flooring 704
        // to 640 reads only zero weights and must fail this reference.
        const int active=(K%128==0)||b>=K/128*4;
        memset(w,0,24); put_half(w,active?0.5f:0); put_half(w+2,active?-0.25f:0);
        uint32_t high=0;
        for(unsigned k=0;k<32;k++) {
            unsigned q=code(e,row,b,k);
            high|=(q>>4)<<k;
            if(k<16)w[8+k]=(unsigned char)(q&15);
            else w[8+k-16]|=(unsigned char)((q&15)<<4);
        }
        memcpy(w+4,&high,4);
    }
}
static float decode(const unsigned char *w,unsigned k) {
    uint32_t high; memcpy(&high,w+4,4);
    unsigned q=(k<16?w[8+k]&15:w[8+k-16]>>4)|(((high>>k)&1)<<4);
    return half(w)*q+half(w+2);
}
static void make_activations(unsigned char *p,unsigned K,unsigned rows) {
    const unsigned supers=(K+127)/128;
    // Unused groups contain NaNs and nonzero bytes. Kernels must mask them,
    // without relying on benign contents from a previous GPU allocation.
    memset(p,0xff,(size_t)supers*rows*144);
    for(unsigned b=0;b<K/32;b++)for(unsigned r=0;r<rows;r++) {
        unsigned char *tile=p+((size_t)(b/4)*rows+r)*144;
        const unsigned g=b%4; int sum=0;
        put_half(tile+4*g,0.125f);
        for(unsigned k=0;k<32;k++) {
            int q=(int)((r*3+b+k*5)%17)-8;
            tile[16+g*32+k]=(unsigned char)(signed char)q; sum+=q;
        }
        put_half(tile+4*g+2,sum*0.125f);
    }
}
static float activation(const unsigned char *p,unsigned rows,unsigned r,unsigned k) {
    unsigned b=k/32;
    const unsigned char *tile=p+((size_t)(b/4)*rows+r)*144;
    return half(tile+4*(b%4))*(signed char)tile[16+(b%4)*32+k%32];
}
static int q8_code(unsigned row,unsigned block,unsigned k) {
    return (int)((row*3+block+k*5)%31)-15;
}
static void make_q8_weights(unsigned char *p,unsigned M,unsigned K,unsigned base) {
    const unsigned blocks=K/32;
    memset(p,0xa5,base+(size_t)M*blocks*34);
    for(unsigned row=0;row<M;row++)for(unsigned b=0;b<blocks;b++) {
        unsigned char *w=p+base+((size_t)row*blocks+b)*34;
        put_half(w,(K%128==0||b>=K/128*4)?0.5f:0);
        for(unsigned k=0;k<32;k++)w[2+k]=(unsigned char)(signed char)q8_code(row,b,k);
    }
}
static void q8_fixture(unsigned K) {
    const unsigned M=3,rows=5,base=34;
    unsigned char *w=allocate(base+(size_t)M*(K/32)*34);
    unsigned char *x=allocate((size_t)((K+127)/128)*rows*144);
    make_q8_weights(w,M,K,base); make_activations(x,K,rows);
    unsigned nonzero=0;
    for(unsigned row=0;row<M;row++) {
        double dot=0;
        for(unsigned k=0;k<K;k++) {
            const unsigned b=k/32;
            const unsigned char *block=w+base+((size_t)row*(K/32)+b)*34;
            const float expected=(K%128==0||b>=K/128*4)?0.5f*q8_code(row,b,k%32):0;
            if(half(block)*(signed char)block[2+k%32]!=expected)fail("Q8_0 fixture packing");
            dot+=(double)expected*activation(x,rows,1,k);
        }
        nonzero+=dot!=0;
    }
    if(!nonzero)fail("Q8_0 tail fixture has no reference signal");
    free(w); free(x); cases++;
}
static void fixture(unsigned K) {
    const unsigned M=3,E=3,rows=5,base=24;
    unsigned char *w=allocate(base+(size_t)E*M*(K/32)*24);
    unsigned char *x=allocate((size_t)((K+127)/128)*rows*144);
    make_weights(w,E,M,K,base); make_activations(x,K,rows);
    unsigned nonzero=0;
    for(unsigned e=0;e<E;e++)for(unsigned row=0;row<M;row++) {
        const unsigned char *weight=w+base+((size_t)e*M+row)*(K/32)*24;
        double dot=0;
        for(unsigned k=0;k<K;k++) {
            unsigned b=k/32;
            const int active=K%128==0||b>=K/128*4;
            float expected=active?0.5f*code(e,row,b,k%32)-0.25f:0;
            if(decode(weight+b*24,k%32)!=expected)fail("Q5_1 fixture packing");
            dot+=(double)expected*activation(x,rows,1,k);
        }
        nonzero+=dot!=0;
    }
    if(!nonzero)fail("tail fixture has no reference signal");
    free(w); free(x); cases++;
}

#ifndef ZINC_REFERENCE_ONLY
static char *source(void) {
    FILE *f=fopen("src/shaders/cuda/kernels.cu","rb");
    if(!f||fseek(f,0,SEEK_END))fail("kernel source");
    long n=ftell(f); if(n<0||fseek(f,0,SEEK_SET))fail("kernel source size");
    char *s=allocate((size_t)n+1);
    if(fread(s,1,(size_t)n,f)!=(size_t)n)fail("kernel source read");
    fclose(f); s[n]=0; return s;
}
static CudaBuf *upload(CudaCtx *ctx,const void *p,size_t n) {
    CudaBuf *b=cuda_create_buffer(ctx,n); if(!b)fail("device allocation");
    cuda_upload(ctx,b,p,n); if(cuda_execution_failed(ctx))fail("upload"); return b;
}
typedef struct { unsigned M,K,T,a_offset,x_offset,y_offset,acc_mode; } Gemm;
_Static_assert(sizeof(Gemm)==28,"Dense Q8_0 push ABI");
static void q8_case(CudaCtx *ctx,CudaPipe **pipes,unsigned M,unsigned K,unsigned tokens) {
    const unsigned base=34,rows=tokens+1,row_bytes=K/32*34;
    const size_t wb=base+(size_t)M*row_bytes,xb=(size_t)((K+127)/128)*rows*144;
    const size_t values=(size_t)rows*M+19;
    unsigned char *w=allocate(wb),*x=allocate(xb);
    float *decoded=allocate((size_t)M*K*4),*initial=allocate(values*4),*out=allocate(values*4);
    make_q8_weights(w,M,K,base); make_activations(x,K,rows);
    const struct ggml_type_traits *traits=ggml_get_type_traits(GGML_TYPE_Q8_0);
    if(!traits||!traits->to_float)fail("native GGML Q8_0 reference");
    for(unsigned r=0;r<M;r++)traits->to_float(w+base+(size_t)r*row_bytes,decoded+(size_t)r*K,K);
    CudaBuf *a=upload(ctx,w,wb),*b=upload(ctx,x,xb);
    const unsigned tiles[]={112,48,64,64,64,80,16},heights[]={128,128,128,64,32,128,128};
    for(unsigned variant=0;variant<7;variant++) {
        for(size_t i=0;i<values;i++)initial[i]=123456.0f;
        CudaBuf *y=upload(ctx,initial,values*4),*args[]={a,b,b,y};
        // Skip the first token via the real byte-offset ABI. It must stay
        // untouched even when the final token/output-row tile is incomplete.
        Gemm pc={M,K,rows,base,K*4,0,0};
        unsigned grid[]={(M+heights[variant]-1)/heights[variant],(tokens+tiles[variant]-1)/tiles[variant],1};
        unsigned threads[]={heights[variant]*2,1,1};
        CudaCmd *cmd=cuda_begin_command(ctx);
        cuda_dispatch(cmd,pipes[variant],grid,threads,args,4,&pc,sizeof pc,0);
        cuda_commit_and_wait(cmd); if(cuda_execution_failed(ctx))fail("Q8_0 WMMA launch");
        cuda_download(ctx,y,out,values*4);
        for(unsigned t=1;t<rows;t++)for(unsigned r=0;r<M;r++) {
            double expected=0;
            for(unsigned k=0;k<K;k++)expected+=(double)decoded[(size_t)r*K+k]*activation(x,rows,t,k);
            const float actual=out[(size_t)t*M+r];
            if(!isfinite(actual)||fabs(expected-actual)>0.00001) {
                fprintf(stderr,"Q8_0 K=%u M=%u variant=%u expected=%.9g actual=%.9g\n",K,M,variant,expected,actual);
                fail("Q8_0 full native reference (including partial tile)");
            }
        }
        for(unsigned r=0;r<M;r++)if(out[r]!=initial[r])fail("Q8_0 token offset guard");
        for(size_t i=(size_t)rows*M;i<values;i++)if(out[i]!=initial[i])fail("Q8_0 output suffix guard");
        cuda_free_buffer(y); cases++;
    }
    cuda_free_buffer(a); cuda_free_buffer(b); free(w); free(x); free(decoded); free(initial); free(out);
}
static void kernel_case(CudaCtx *ctx,CudaPipe **pipes,unsigned M,unsigned K,unsigned tokens) {
    const unsigned E=3,slots=2,rows=tokens*slots,base=24,stride=M*slots+7;
    const unsigned row_bytes=K/32*24;
    const size_t wb=base+(size_t)E*M*row_bytes,xb=(size_t)((K+127)/128)*rows*144;
    const size_t values=(size_t)tokens*stride+19;
    unsigned char *w=allocate(wb),*x=allocate(xb);
    float *decoded=allocate((size_t)E*M*K*4),*initial=allocate(values*4),*out=allocate(values*4);
    make_weights(w,E,M,K,base); make_activations(x,K,rows);
    const struct ggml_type_traits *traits=ggml_get_type_traits(GGML_TYPE_Q5_1);
    if(!traits||!traits->to_float)fail("native GGML Q5_1 reference");
    for(unsigned e=0;e<E;e++)for(unsigned r=0;r<M;r++)
        traits->to_float(w+base+((size_t)e*M+r)*row_bytes,decoded+((size_t)e*M+r)*K,K);
    CudaBuf *a=upload(ctx,w,wb),*b=upload(ctx,x,xb);
    for(unsigned variant=0;variant<3;variant++) {
        const unsigned tile=variant==2?64:16,max_tiles=(rows+tile-1)/tile+E;
        unsigned *order=allocate((size_t)max_tiles*tile*4),*experts=allocate(max_tiles*4);
        for(unsigned i=0;i<max_tiles*tile;i++)order[i]=UINT32_MAX;
        for(unsigned i=0;i<max_tiles;i++)experts[i]=UINT32_MAX;
        unsigned pos=0;
        for(unsigned e=0;e<E;e++) {
            for(unsigned t=1;t<tokens;t++)for(unsigned slot=0;slot<slots;slot++) {
                if((t*7+slot+1)%E!=e||(t+1==tokens&&slot==1))continue;
                experts[pos/tile]=e; order[pos++]=(t<<16)|slot;
            }
            pos=(pos+tile-1)/tile*tile;
        }
        for(size_t i=0;i<values;i++)initial[i]=123456.0f;
        CudaBuf *o=upload(ctx,order,(size_t)max_tiles*tile*4);
        CudaBuf *e=upload(ctx,experts,max_tiles*4),*y=upload(ctx,initial,values*4);
        CudaBuf *args[]={a,b,o,e,y};
        Group pc={M,K,rows,base,M*row_bytes,stride,slots};
        unsigned grid[]={(M+127)/128,max_tiles,1},threads[]={256,1,1};
        CudaCmd *cmd=cuda_begin_command(ctx);
        cuda_dispatch(cmd,pipes[variant],grid,threads,args,5,&pc,sizeof pc,0);
        cuda_commit_and_wait(cmd); if(cuda_execution_failed(ctx))fail("Q5_1 grouped launch");
        cuda_download(ctx,y,out,values*4);
        for(unsigned t=0;t<tokens;t++)for(unsigned slot=0;slot<slots;slot++)for(unsigned r=0;r<M;r++) {
            const size_t at=(size_t)t*stride+slot*M+r;
            if(t==0||(t+1==tokens&&slot==1)) {
                if(out[at]!=initial[at])fail("invalid/padded route guard");
                continue;
            }
            const unsigned expert=(t*7+slot+1)%E;
            const float *weight=decoded+((size_t)expert*M+r)*K;
            double expected=0;
            for(unsigned k=0;k<K;k++)expected+=(double)weight[k]*activation(x,rows,t*slots+slot,k);
            // Binary-fraction fixtures keep this comparison independent of a
            // loose relative tolerance, even when the old output was zero.
            if(!isfinite(out[at])||fabs(expected-out[at])>0.00001) {
                fprintf(stderr,"K=%u M=%u variant=%u expected=%.9g actual=%.9g\n",K,M,variant,expected,out[at]);
                fail("Q5_1 full native reference (including partial tile)");
            }
        }
        for(unsigned t=0;t<tokens;t++)for(unsigned gap=M*slots;gap<stride;gap++)
            if(out[(size_t)t*stride+gap]!=initial[(size_t)t*stride+gap])fail("token stride guard");
        for(size_t i=(size_t)tokens*stride;i<values;i++)if(out[i]!=initial[i])fail("output suffix guard");
        cuda_free_buffer(o); cuda_free_buffer(e); cuda_free_buffer(y); free(order); free(experts); cases++;
    }
    cuda_free_buffer(a); cuda_free_buffer(b); free(w); free(x); free(decoded); free(initial); free(out);
}
static void quantizer_case(CudaCtx *ctx,CudaPipe *pipe,unsigned K,unsigned rows) {
    const unsigned supers=(K+127)/128;
    const size_t bytes=(size_t)supers*rows*144,guard=(size_t)rows*288+64;
    float *x=allocate((size_t)K*rows*4);
    for(unsigned r=0;r<rows;r++)for(unsigned k=0;k<K;k++)
        x[(size_t)r*K+k]=k%32==0?127.0f:(float)((int)((k+r)%3)-1);
    unsigned char *initial=allocate(bytes+guard),*out=allocate(bytes+guard); memset(initial,0xa5,bytes+guard);
    CudaBuf *a=upload(ctx,x,(size_t)K*rows*4),*b=upload(ctx,initial,bytes+guard),*args[]={a,b};
    // The host folds large row counts over y/z. Exercise that path without
    // relying on a model large enough to allocate 65K prompt tokens.
    const unsigned grid_y=rows>65535?257:rows;
    unsigned pc[]={K,rows},grid[]={(K+255)/256,grid_y,(rows+grid_y-1)/grid_y},threads[]={256,1,1};
    CudaCmd *cmd=cuda_begin_command(ctx); cuda_dispatch(cmd,pipe,grid,threads,args,2,pc,sizeof pc,0);
    cuda_commit_and_wait(cmd); if(cuda_execution_failed(ctx))fail("activation quantizer launch");
    cuda_download(ctx,b,out,bytes+guard);
    for(unsigned h=0;h<supers;h++)for(unsigned r=0;r<rows;r++)for(unsigned g=0;g<4;g++) {
        const unsigned block=h*4+g; const unsigned char *p=out+((size_t)h*rows+r)*144;
        int sum=0;
        for(unsigned k=0;k<32;k++) {
            int expected=block*32+k<K?(int)x[(size_t)r*K+block*32+k]:0;
            if((signed char)p[16+g*32+k]!=expected)fail("quantized values or zero padding");
            sum+=expected;
        }
        if(half(p+g*4)!=(block*32<K?1.0f:0)||half(p+g*4+2)!=sum)fail("quantized scale/sum or zero padding");
    }
    if(memcmp(out+bytes,initial+bytes,guard))fail("quantizer wrote beyond final packed tile");
    cuda_free_buffer(a); cuda_free_buffer(b); free(x); free(initial); free(out); cases++;
}
#endif

int main(int argc,char **argv) {
    const unsigned widths[]={704,32,64,96,128,160,256,512,736,1024};
    for(unsigned i=0;i<sizeof widths/sizeof widths[0];i++)fixture(widths[i]);
    fixture(2112);
    for(unsigned i=0;i<sizeof widths/sizeof widths[0];i++)q8_fixture(widths[i]);
    q8_fixture(2112);
#ifdef ZINC_REFERENCE_ONLY
    (void)argc; (void)argv;
    printf("PASS %u Q5_1/Q8_0 packing and decisive-tail reference fixtures\n",cases);
#else
    const int q8_only=argc==2&&strcmp(argv[1],"--q8-only")==0;
    if(argc>1&&!q8_only)fail("unsupported argument");
    const char *selected=getenv("ZINC_GPU"); char *end=NULL;
    unsigned long device=selected?strtoul(selected,&end,10):0;
    if(selected&&(!*selected||*end||device>INT32_MAX))fail("device selector");
    cases=0; char *s=source(); CudaCtx *ctx=cuda_init((int)device); if(!ctx)fail("GPU init");
    const char *q8_names[]={"gemm_q8_0_wmma_i8","gemm_q8_0_wmma_i8_t48","gemm_q8_0_wmma_i8_t64","gemm_q8_0_wmma_i8_t64_m64","gemm_q8_0_wmma_i8_t64_m32","gemm_q8_0_wmma_i8_t80","gemm_q8_0_wmma_i8_t16"};
    CudaPipe *q8_pipes[7];
    for(unsigned i=0;i<7;i++) { q8_pipes[i]=cuda_create_pipeline(ctx,s,q8_names[i],NULL,0); if(!q8_pipes[i])fail("Q8_0 compile"); }
    for(unsigned k=0;k<10;k++) {
        q8_case(ctx,q8_pipes,17,widths[k],17);
        q8_case(ctx,q8_pipes,129,widths[k],113);
    }
    q8_case(ctx,q8_pipes,17,2112,17); q8_case(ctx,q8_pipes,129,2112,113);
    q8_case(ctx,q8_pipes,2816,2112,17);
    for(unsigned i=0;i<7;i++)cuda_free_pipeline(q8_pipes[i]);
    if(!q8_only) {
    const char *names[]={"gemm_q5_1_experts_grouped_i8_t16_direct","gemm_q5_1_experts_grouped_i8_t16","gemm_q5_1_experts_grouped_i8"};
    CudaPipe *pipes[3];
    for(unsigned i=0;i<3;i++) { pipes[i]=cuda_create_pipeline(ctx,s,names[i],NULL,0); if(!pipes[i])fail("grouped compile"); }
    const unsigned heights[]={17,129},tokens[]={3,17};
    for(unsigned k=0;k<10;k++)for(unsigned m=0;m<2;m++)for(unsigned t=0;t<2;t++)kernel_case(ctx,pipes,heights[m],widths[k],tokens[t]);
    kernel_case(ctx,pipes,2816,704,17);
    CudaPipe *quant=cuda_create_pipeline(ctx,s,"quantize_act_q8_0",NULL,0); if(!quant)fail("quantizer compile");
    for(unsigned k=0;k<10;k++)for(unsigned t=0;t<2;t++)quantizer_case(ctx,quant,widths[k],tokens[t]);
    quantizer_case(ctx,quant,32,65537);
    cuda_free_pipeline(quant); for(unsigned i=0;i<3;i++)cuda_free_pipeline(pipes[i]);
    }
    cuda_destroy(ctx); free(s);
    if(cases!=(q8_only?161u:305u))fail("incomplete numerical suite");
    printf("PASS %u ROCm partial-tile/native-reference/route/quantizer/guard cases\n",cases);
#endif
    return 0;
}
