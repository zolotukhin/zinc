//! CUDA forward-pass debug harness for the qwen35 hybrid-SSM model. Two modes:
//!
//!   zig build cuda-dbg -- <token> [model.gguf]
//!       Per-layer residual-norm dump at pos 0 (used to pinpoint the attention
//!       gate bug: diff vs a reference implementation eval-callback `l_out-N` reference).
//!
//!   zig build cuda-dbg -- gen <id,id,...> <ngen> [model.gguf]
//!       Autoregressive greedy generation from a prompt token-id list. Prefills
//!       the ids (exercising pos>0 RoPE + multi-entry attention + SSM state
//!       carry), then greedily emits ngen tokens. Diff GEN_IDS vs `/tmp/gen`
//!       (the reference implementation greedy) to validate the full decode path beyond pos 0.
//!
//! Read-only w.r.t. the engine — uses only public ForwardCuda methods.
//! @section CUDA Runtime
const std = @import("std");
const device = @import("cuda/device.zig");
const loader = @import("model/loader_cuda.zig");
const forward = @import("compute/forward_cuda.zig");
const forwardgemma = @import("compute/forward_cuda_gemma.zig");
const pipeline = @import("cuda/pipeline.zig");
const buffer = @import("cuda/buffer.zig");
const command = @import("cuda/command.zig");
const scheduler = @import("scheduler/scheduler.zig");

/// Drives either the qwen35/qwen36 (`ForwardCuda`) or gemma4 (`ForwardGemma`)
/// decode engine, selected from the model architecture. Both expose the same
/// `decodeStep`/`readLogits`/`d.vocab` shape, so gen + logit validation is
/// uniform across the whole 5-model catalog.
const Engine = union(enum) {
    qwen: forward.ForwardCuda,
    gemma: forwardgemma.ForwardGemma,

    fn init(allocator: std.mem.Allocator, model: *loader.Model, max_ctx: u32) !Engine {
        if (model.config.architecture == .gemma)
            return .{ .gemma = try forwardgemma.ForwardGemma.init(allocator, model, max_ctx) };
        return .{ .qwen = try forward.ForwardCuda.init(allocator, model, max_ctx) };
    }
    fn deinit(self: *Engine) void {
        switch (self.*) {
            inline else => |*e| e.deinit(),
        }
    }
    fn decodeStep(self: *Engine, token: u32, pos: u32, run_layers: bool) !u32 {
        switch (self.*) {
            inline else => |*e| return e.decodeStep(token, pos, run_layers),
        }
    }
    fn prefillStep(self: *Engine, token: u32, pos: u32) !void {
        switch (self.*) {
            inline else => |*e| try e.prefillStep(token, pos),
        }
    }
    /// Effort 24: batched-GEMM prefill (gemma only). Returns the last token's
    /// argmax. error.Unsupported on the qwen forward so the caller falls back.
    fn prefillBatched(self: *Engine, tokens: []const u32) !u32 {
        switch (self.*) {
            inline else => |*e| {
                if (comptime @hasDecl(@TypeOf(e.*), "prefillBatched")) return e.prefillBatched(tokens);
                return error.Unsupported;
            },
        }
    }
    fn readLogits(self: *Engine, out: []f32) void {
        switch (self.*) {
            inline else => |*e| e.readLogits(out),
        }
    }
    /// Reset the production single-sequence recurrent state between serial
    /// reference sequences. qwen has unindexed SSM recurrent state that would leak
    /// across sequences; gemma is attention-only (position-indexed KV) so its
    /// serial reference is sound without a reset — no-op there.
    fn resetState(self: *Engine) !void {
        switch (self.*) {
            inline else => |*e| {
                if (comptime @hasDecl(@TypeOf(e.*), "resetState")) try e.resetState();
            },
        }
    }
    /// Effort 28 serving: per-sequence slot state alloc/free + batched decode +
    /// per-slot reset, dispatched by arch (gemma `allocSlotKv`/qwen `allocSlotState`;
    /// `resetSlot` clears a reused slot's accumulated SSM state — gemma has none).
    fn allocSlots(self: *Engine, nslots: u32, slot_ctx: u32) !void {
        switch (self.*) {
            inline else => |*e| {
                if (comptime @hasDecl(@TypeOf(e.*), "allocSlotKv")) {
                    try e.allocSlotKv(nslots, slot_ctx);
                } else {
                    try e.allocSlotState(nslots, slot_ctx);
                }
            },
        }
    }
    fn freeSlots(self: *Engine) void {
        switch (self.*) {
            inline else => |*e| {
                if (comptime @hasDecl(@TypeOf(e.*), "freeSlotKv")) e.freeSlotKv() else e.freeSlotState();
            },
        }
    }
    fn decodeBatch(self: *Engine, tokens: []const u32, positions: []const u32, slots: []const u32, out: []u32) !void {
        switch (self.*) {
            inline else => |*e| try e.decodeBatch(tokens, positions, slots, out),
        }
    }
    fn resetSlot(self: *Engine, slot: u32) !void {
        switch (self.*) {
            inline else => |*e| {
                if (comptime @hasDecl(@TypeOf(e.*), "resetSlot")) try e.resetSlot(slot);
            },
        }
    }
    fn vocab(self: *const Engine) u32 {
        switch (self.*) {
            inline else => |*e| return e.d.vocab,
        }
    }
};

const DEFAULT_MODEL = "models/Qwen3.5-9B-Q4_K_M.gguf";

/// Effort 25: batched-GEMM prefill is the DEFAULT for gemma (matches main.zig).
/// True unless ZINC_BATCHED_PREFILL is explicitly off (0/off/false/no). qwen has
/// no prefillBatched (Engine returns error.Unsupported) so it falls back anyway.
fn batchedPrefillDefaultOn() bool {
    const v = std.posix.getenv("ZINC_BATCHED_PREFILL") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

// Synthetic compute kernel for the dispatch sync-vs-async bench: a per-thread
// FMA loop of `iters` so each dispatch costs a tunable, decode-matvec-like amount.
const BENCH_CU =
    \\struct BenchPush { int iters; };
    \\extern "C" __global__ void benchk(float* x, BenchPush pc) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    float a = x[idx];
    \\    for (int i = 0; i < pc.iters; i++) a = a * 1.0000001f + 0.0000001f;
    \\    x[idx] = a;
    \\}
;
const BenchPush = extern struct { iters: i32 };

// Effort-30 int8-MMA feasibility microbench (READ-ONLY, no model). Settles the
// two gating unknowns the awake-session Q4_K-int8 GEMM depends on:
//  (1) does NVRTC on sm_120 COMPILE + correctly execute inline-PTX
//      `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (the fp16 *wmma-intrinsic*
//      lowering miscompiled on sm_120, but inline PTX bypasses intrinsic lowering
//      → if this works, no multi-day nvcc-CUBIN path is needed = huge de-risk);
//  (2) is the Blackwell int8 TC rate actually ~2x fp16 wmma (the entire premise —
//      if not, the int8 lever is dead on arrival regardless of the epilogue).
// `mma_unit` is a single-warp known-value m16n8k32 s8 matmul checked vs a scalar
// host reference (proves both compile AND that the PTX-ISA fragment→register map
// used for the in-register epilogue is correct). `tp_int8`/`tp_f16` issue equal
// counts of 4096-MAC TC calls (int8 m16n8k32 vs fp16 wmma 16x16x16) so calls/s
// ratio == effective MAC/s (TC-rate) ratio under identical occupancy.
const MMA8_CU =
    \\#include <mma.h>
    \\using namespace nvcuda;
    \\// ---- correctness unit: one warp, one m16n8k32 s8 mma, known values --------
    \\extern "C" __global__ void mma_unit(const signed char* A, const signed char* B, int* D) {
    \\    int lane = threadIdx.x & 31;
    \\    int gid = lane >> 2;      // 0..7
    \\    int tig = lane & 3;       // 0..3
    \\    // A is 16x32 row-major; pack 4 consecutive s8 (col-contiguous) per reg.
    \\    int a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    \\    #pragma unroll
    \\    for (int b = 0; b < 4; b++) {
    \\        a0 |= ((int)(unsigned char)A[(gid)   *32 + tig*4      + b]) << (8*b);
    \\        a1 |= ((int)(unsigned char)A[(gid+8) *32 + tig*4      + b]) << (8*b);
    \\        a2 |= ((int)(unsigned char)A[(gid)   *32 + tig*4 + 16 + b]) << (8*b);
    \\        a3 |= ((int)(unsigned char)A[(gid+8) *32 + tig*4 + 16 + b]) << (8*b);
    \\        // B is 32x8 col-major: element (k,n) at B[k + n*32].
    \\        b0 |= ((int)(unsigned char)B[(tig*4      + b) + gid*32]) << (8*b);
    \\        b1 |= ((int)(unsigned char)B[(tig*4 + 16 + b) + gid*32]) << (8*b);
    \\    }
    \\    int c0=0,c1=0,c2=0,c3=0;
    \\    asm volatile(
    \\        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    \\        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    \\        : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
    \\        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    \\    // D is 16x8 row-major; accumulator (row,col) map per PTX ISA.
    \\    D[(gid)  *8 + tig*2    ] = c0;
    \\    D[(gid)  *8 + tig*2 + 1] = c1;
    \\    D[(gid+8)*8 + tig*2    ] = c2;
    \\    D[(gid+8)*8 + tig*2 + 1] = c3;
    \\}
    \\struct TpPush { int iters; };
    \\// ---- int8 throughput: iters independent m16n8k32 s8 mma calls -------------
    \\extern "C" __global__ void tp_int8(int* out, TpPush pc) {
    \\    int a0=0x01020304,a1=0x05060708,a2=0x090a0b0c,a3=0x0d0e0f10;
    \\    int b0=0x11121314,b1=0x15161718;
    \\    int c0=0,c1=0,c2=0,c3=0, d0=0,d1=0,d2=0,d3=0;
    \\    for (int i = 0; i < pc.iters; i++) {
    \\        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    \\            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    \\            : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
    \\            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    \\        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    \\            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    \\            : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
    \\            : "r"(a1),"r"(a2),"r"(a3),"r"(a0),"r"(b1),"r"(b0));
    \\    }
    \\    if (threadIdx.x == 999) out[blockIdx.x] = c0+c1+c2+c3+d0+d1+d2+d3;
    \\}
    \\// ---- fp16 baseline throughput: iters wmma 16x16x16 (same 4096 MAC/call) ---
    \\extern "C" __global__ void tp_f16(int* out, TpPush pc) {
    \\    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
    \\    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
    \\    wmma::fragment<wmma::accumulator,16,16,16,float> cf, df;
    \\    wmma::fill_fragment(af, __float2half(1.0f));
    \\    wmma::fill_fragment(bf, __float2half(1.0f));
    \\    wmma::fill_fragment(cf, 0.0f);
    \\    wmma::fill_fragment(df, 0.0f);
    \\    for (int i = 0; i < pc.iters; i++) {
    \\        wmma::mma_sync(cf, af, bf, cf);
    \\        wmma::mma_sync(df, af, bf, df);
    \\    }
    \\    if (threadIdx.x == 999) out[blockIdx.x] = (int)(cf.x[0] + df.x[0]);
    \\}
;
const TpPush = extern struct { iters: i32 };

// Effort-30 THE KILL-BAR microbench: a FULL Q4_K-int8 tensor-core GEMM vs the
// shipped fp16 `gemm_q4k_tc`, ISOLATED, WITH memory traffic, at gemma shapes
// (M≈K≈4608, T=512). The mma8 feasibility bench already proved (Q1) NVRTC/sm_120
// compiles+executes inline-PTX m16n8k32.s8 correctly and (Q2) int8 TC is ~1.9x
// fp16 register-resident. THIS answers the ONE remaining question mma8 could not:
// does the 1.9x COMPUTE ceiling survive the real weight/activation traffic + the
// Q4_K-asymmetric per-subblock epilogue → is int8 ≥1.3x vs gemm_q4k_tc end-to-end?
//
// Design (grounded in gemm_q4k_tc, kernels.cu:4250): BM=BT=64, BK=32 == one Q4_K
// 32-subblock. Weight nibble stays raw s8 (0..15); activation quantized PER-SUBBLOCK
// (Q8_1-style: sA[t,sb]=max|A|/127, qA=round(A/sA)) so the asymmetric scales fold
// IN-REGISTER with NO store-s32-to-shared (the tax that killed cycle-8's wmma-k16):
//   Y[m,t] = Σ_sb sA[t,sb]·( d_sc[m,sb]·P[m,t,sb] − dm_mn[m,sb]·SA[t,sb] )
// where P = Σ_{k∈sb} nib·qA is the s32 mma accumulator and SA = Σ_{k∈sb} qA.
// The m16n8k32 fragment→(row,col) register map is PTX-ISA-DEFINED (validated bit-
// exact by mma8's mma_unit) so d_sc/dm_mn/sA/SA are applied per accumulator element
// in-register. Correctness = relative error vs gemm_q4k_tc (both quant paths of the
// SAME synthetic weight; int8-activation adds a few % — plausible token-tolerance).
const GEMM8_CU =
    \\#include <mma.h>
    \\using namespace nvcuda;
    \\__device__ __forceinline__ float zinc_half_to_float(unsigned short h) {
    \\    unsigned sign=(unsigned)(h>>15)&1u, exp=(unsigned)(h>>10)&0x1Fu, mant=(unsigned)h&0x3FFu, f;
    \\    if (exp==0u){ if(mant==0u){f=sign<<31;} else { int e=1; while((mant&0x400u)==0u){mant<<=1;e--;} mant&=0x3FFu; f=(sign<<31)|((unsigned)(127-15+e)<<23)|(mant<<13);} }
    \\    else if (exp==0x1Fu){ f=(sign<<31)|(0xFFu<<23)|(mant<<13); }
    \\    else { f=(sign<<31)|((exp-15u+127u)<<23)|(mant<<13); }
    \\    return __int_as_float((int)f);
    \\}
    \\__device__ __forceinline__ void zinc_q4k_scale_min(int j, const unsigned char* q, unsigned char* d, unsigned char* m) {
    \\    if (j<4){ *d=q[j]&63u; *m=q[j+4]&63u; }
    \\    else { *d=(q[j+4]&0xFu)|((q[j-4]>>6)<<4); *m=(q[j+4]>>4)|((q[j]>>6)<<4); }
    \\}
    \\struct GemmPush { unsigned M,K,T,a_offset,x_offset,y_offset,acc_mode,q8_stride; };
    \\// ---- fp16 baseline: VERBATIM gemm_q4k_tc (kernels.cu:4250) ----------------
    \\extern "C" __global__ void gemm_q4k_tc(const unsigned* a_u32, const float* A, float* Y, GemmPush pc) {
    \\    const unsigned BM=64u, BT=64u, BK=32u;
    \\    __shared__ half Ws[BM*BK]; __shared__ half As[BK*BT]; __shared__ float Cs[BT*BM];
    \\    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT;
    \\    unsigned bpr=pc.K>>8, nchunk=pc.K>>5, tid=threadIdx.x, a0=(pc.a_offset>>2);
    \\    const float* Abase=A+(pc.x_offset>>2);
    \\    unsigned warp=tid>>5, fm=warp>>2, ft=warp&3u;
    \\    wmma::fragment<wmma::accumulator,16,16,16,float> c0,c1;
    \\    wmma::fill_fragment(c0,0.0f); wmma::fill_fragment(c1,0.0f);
    \\    for (unsigned c=0;c<nchunk;c++){
    \\        unsigned sbk=c>>3, sb8=c&7u, warp_id=tid>>5, lane=tid&31u;
    \\        #pragma unroll
    \\        for (int u=0;u<8;u++){
    \\            unsigned r=warp_id+(unsigned)u*8u, l=lane, row=m0+r; float wv=0.0f;
    \\            if (row<pc.M){
    \\                unsigned blk=a0+row*bpr*36u+sbk*36u; float d_sc,dm_mn;
    \\                if (lane==0){ unsigned dd=a_u32[blk]; float d=zinc_half_to_float((unsigned short)(dd&0xFFFFu)); float dmin=zinc_half_to_float((unsigned short)(dd>>16)); const unsigned char* scales=(const unsigned char*)(a_u32+blk+1u); unsigned char sc,mn; zinc_q4k_scale_min((int)sb8,scales,&sc,&mn); d_sc=d*(float)sc; dm_mn=dmin*(float)mn; }
    \\            d_sc=__shfl_sync(0xFFFFFFFFu,d_sc,0); dm_mn=__shfl_sync(0xFFFFFFFFu,dm_mn,0);
    \\                const unsigned char* qs=(const unsigned char*)(a_u32+blk+4u); unsigned char qb=qs[(sb8>>1)*32u+l]; unsigned nib=(sb8&1u)==0u?(qb&0xFu):(unsigned)(qb>>4); wv=d_sc*(float)nib-dm_mn;
    \\            }
    \\            Ws[r*BK+l]=__float2half(wv);
    \\        }
    \\        #pragma unroll
    \\        for (int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, t=idx>>5, l=idx&31u, tok=t0+t; As[l*BT+t]=(tok<pc.T)?__float2half(Abase[(size_t)tok*pc.K+c*32u+l]):__float2half(0.0f); }
    \\        __syncthreads();
    \\        #pragma unroll
    \\        for (unsigned ks=0;ks<2;ks++){
    \\            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f,a1f;
    \\            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
    \\            wmma::load_matrix_sync(a0f,&Ws[(fm*16u)*BK+ks*16u],BK);
    \\            wmma::load_matrix_sync(a1f,&Ws[((fm+2u)*16u)*BK+ks*16u],BK);
    \\            wmma::load_matrix_sync(bf,&As[(ks*16u)*BT+ft*16u],BT);
    \\            wmma::mma_sync(c0,a0f,bf,c0); wmma::mma_sync(c1,a1f,bf,c1);
    \\        }
    \\        __syncthreads();
    \\    }
    \\    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+fm*16u],c0,BM,wmma::mem_col_major);
    \\    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+(fm+2u)*16u],c1,BM,wmma::mem_col_major);
    \\    __syncthreads();
    \\    #pragma unroll
    \\    for (int u=0;u<16;u++){ unsigned idx=tid+(unsigned)u*256u, t=idx>>6, m=idx&63u, tok=t0+t, row=m0+m; if(row<pc.M&&tok<pc.T){ unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row; if(pc.acc_mode!=0u) Y[yi]+=Cs[t*BM+m]; else Y[yi]=Cs[t*BM+m]; } }
    \\}
    \\// ---- int8 candidate: raw-nibble s8 * per-subblock-quantized s8 activation ----
    \\extern "C" __global__ void gemm_q4k_int8(const unsigned* a_u32, const float* A, float* Y, GemmPush pc) {
    \\    const unsigned BM=64u, BT=64u, BK=32u;
    \\    __shared__ signed char Wnib[BM*BK];   // m-major raw nibbles (0..15)
    \\    __shared__ signed char Aq[BK*BT];     // k-major quantized activation
    \\    __shared__ float dsc[BM], dmn[BM];    // per-row weight scales, current subblock
    \\    __shared__ float sAs[BT], SAs[BT];    // per-token act scale + qsum, current subblock
    \\    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT;
    \\    unsigned bpr=pc.K>>8, nchunk=pc.K>>5, tid=threadIdx.x, a0=(pc.a_offset>>2);
    \\    const float* Abase=A+(pc.x_offset>>2);
    \\    unsigned warp=tid>>5, lane=tid&31u;
    \\    unsigned mw=warp&3u, grp=warp>>2;     // this warp: m-tile mw, t-tiles grp*4+{0..3}
    \\    unsigned gid=lane>>2, tig=lane&3u;
    \\    float acc[4][4];
    \\    #pragma unroll
    \\    for (int i=0;i<4;i++){ acc[i][0]=0;acc[i][1]=0;acc[i][2]=0;acc[i][3]=0; }
    \\    for (unsigned c=0;c<nchunk;c++){
    \\        unsigned sbk=c>>3, sb8=c&7u;
    \\        // stage weight nibbles (m-major) + per-row d_sc/dm_mn (mirror gemm_q4k_tc)
    \\        #pragma unroll
    \\        for (int u=0;u<8;u++){
    \\            unsigned r=warp+(unsigned)u*8u, l=lane, row=m0+r; float d_sc=0.0f,dm_mn=0.0f; signed char nb=0;
    \\            if (row<pc.M){
    \\                unsigned blk=a0+row*bpr*36u+sbk*36u;
    \\                if (lane==0){ unsigned dd=a_u32[blk]; float d=zinc_half_to_float((unsigned short)(dd&0xFFFFu)); float dmin=zinc_half_to_float((unsigned short)(dd>>16)); const unsigned char* scales=(const unsigned char*)(a_u32+blk+1u); unsigned char sc,mn; zinc_q4k_scale_min((int)sb8,scales,&sc,&mn); d_sc=d*(float)sc; dm_mn=dmin*(float)mn; }
    \\                d_sc=__shfl_sync(0xFFFFFFFFu,d_sc,0); dm_mn=__shfl_sync(0xFFFFFFFFu,dm_mn,0);
    \\                const unsigned char* qs=(const unsigned char*)(a_u32+blk+4u); unsigned char qb=qs[(sb8>>1)*32u+l]; nb=(signed char)((sb8&1u)==0u?(qb&0xFu):(unsigned)(qb>>4));
    \\            }
    \\            Wnib[r*BK+l]=nb;
    \\            if (l==0){ dsc[r]=d_sc; dmn[r]=dm_mn; }
    \\        }
    \\        // stage activation: warp-per-token (warp handles tokens warp,warp+8,...,warp+56)
    \\        #pragma unroll
    \\        for (int j=0;j<8;j++){
    \\            unsigned t=warp+(unsigned)j*8u, tok=t0+t;
    \\            float v=(tok<pc.T)?Abase[(size_t)tok*pc.K+c*32u+lane]:0.0f;
    \\            float amax=fabsf(v);
    \\            #pragma unroll
    \\            for (int o=16;o>0;o>>=1) amax=fmaxf(amax,__shfl_down_sync(0xFFFFFFFFu,amax,o));
    \\            amax=__shfl_sync(0xFFFFFFFFu,amax,0);
    \\            float sA=amax>0.0f?amax/127.0f:1.0f;
    \\            int q=__float2int_rn(v/sA); q=max(-127,min(127,q));
    \\            int qsum=q;
    \\            #pragma unroll
    \\            for (int o=16;o>0;o>>=1) qsum+=__shfl_down_sync(0xFFFFFFFFu,qsum,o);
    \\            Aq[lane*BT+t]=(signed char)q;
    \\            if (lane==0){ sAs[t]=sA; SAs[t]=(float)qsum; }
    \\        }
    \\        __syncthreads();
    \\        // load weight operand for this warp's m-tile (shared across 4 t-tiles)
    \\        int a0r=0,a1r=0,a2r=0,a3r=0;
    \\        #pragma unroll
    \\        for (int b=0;b<4;b++){
    \\            a0r |= ((int)(unsigned char)Wnib[(mw*16u+gid)*BK+tig*4u+(unsigned)b])<<(8*b);
    \\            a1r |= ((int)(unsigned char)Wnib[(mw*16u+gid+8u)*BK+tig*4u+(unsigned)b])<<(8*b);
    \\            a2r |= ((int)(unsigned char)Wnib[(mw*16u+gid)*BK+tig*4u+16u+(unsigned)b])<<(8*b);
    \\            a3r |= ((int)(unsigned char)Wnib[(mw*16u+gid+8u)*BK+tig*4u+16u+(unsigned)b])<<(8*b);
    \\        }
    \\        #pragma unroll
    \\        for (int tt4=0;tt4<4;tt4++){
    \\            unsigned tt=grp*4u+(unsigned)tt4;
    \\            int b0=0,b1=0;
    \\            #pragma unroll
    \\            for (int b=0;b<4;b++){
    \\                b0 |= ((int)(unsigned char)Aq[(tig*4u+(unsigned)b)*BT+tt*8u+gid])<<(8*b);
    \\                b1 |= ((int)(unsigned char)Aq[(tig*4u+16u+(unsigned)b)*BT+tt*8u+gid])<<(8*b);
    \\            }
    \\            int p0=0,p1=0,p2=0,p3=0;
    \\            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    \\                : "+r"(p0),"+r"(p1),"+r"(p2),"+r"(p3)
    \\                : "r"(a0r),"r"(a1r),"r"(a2r),"r"(a3r),"r"(b0),"r"(b1));
    \\            unsigned ma=mw*16u+gid, mb=mw*16u+gid+8u, ta=tt*8u+tig*2u, tb=tt*8u+tig*2u+1u;
    \\            acc[tt4][0]+=sAs[ta]*(dsc[ma]*(float)p0 - dmn[ma]*SAs[ta]);
    \\            acc[tt4][1]+=sAs[tb]*(dsc[ma]*(float)p1 - dmn[ma]*SAs[tb]);
    \\            acc[tt4][2]+=sAs[ta]*(dsc[mb]*(float)p2 - dmn[mb]*SAs[ta]);
    \\            acc[tt4][3]+=sAs[tb]*(dsc[mb]*(float)p3 - dmn[mb]*SAs[tb]);
    \\        }
    \\        __syncthreads();
    \\    }
    \\    // write Y[T,M] token-major
    \\    #pragma unroll
    \\    for (int tt4=0;tt4<4;tt4++){
    \\        unsigned tt=grp*4u+(unsigned)tt4;
    \\        unsigned ma=mw*16u+gid, mb=mw*16u+gid+8u, ta=tt*8u+tig*2u, tb=tt*8u+tig*2u+1u;
    \\        unsigned rma=m0+ma, rmb=m0+mb, tka=t0+ta, tkb=t0+tb;
    \\        if (rma<pc.M&&tka<pc.T) Y[(size_t)tka*pc.M+rma]=acc[tt4][0];
    \\        if (rma<pc.M&&tkb<pc.T) Y[(size_t)tkb*pc.M+rma]=acc[tt4][1];
    \\        if (rmb<pc.M&&tka<pc.T) Y[(size_t)tka*pc.M+rmb]=acc[tt4][2];
    \\        if (rmb<pc.M&&tkb<pc.T) Y[(size_t)tkb*pc.M+rmb]=acc[tt4][3];
    \\    }
    \\}
;
const Gemm8Push = extern struct { M: u32, K: u32, T: u32, a_offset: u32, x_offset: u32, y_offset: u32, acc_mode: u32, q8_stride: u32 };

fn stats(label: []const u8, v: []const f32) void {
    var ss: f64 = 0;
    var mn: f32 = std.math.inf(f32);
    var mx: f32 = -std.math.inf(f32);
    var bad: usize = 0;
    for (v) |x| {
        if (!std.math.isFinite(x)) bad += 1;
        ss += @as(f64, x) * @as(f64, x);
        mn = @min(mn, x);
        mx = @max(mx, x);
    }
    std.debug.print("{s:<9} norm={d:>10.3} min={d:>9.3} max={d:>9.3} nan={d} [0..3]={d:.3} {d:.3} {d:.3}\n", .{ label, std.math.sqrt(ss), mn, mx, bad, v[0], v[1], v[2] });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next();
    const first = args.next() orelse "100";

    if (std.mem.eql(u8, first, "gen")) {
        const ids_arg = args.next() orelse "760,6511,314,9338,369";
        const ngen: u32 = std.fmt.parseInt(u32, args.next() orelse "16", 10) catch 16;
        const model_path = args.next() orelse DEFAULT_MODEL;
        try genMode(allocator, ids_arg, ngen, model_path);
    } else if (std.mem.eql(u8, first, "batch")) {
        // Effort 28 increment 1 (1a): multi-sequence harness. '|' separates
        // sequences, ',' separates prompt token-ids within a sequence.
        const seqs_arg = args.next() orelse "760,6511,314,9338,369|450,3271,310,3444,338";
        const ngen: u32 = std.fmt.parseInt(u32, args.next() orelse "8", 10) catch 8;
        const model_path = args.next() orelse DEFAULT_MODEL;
        try batchMode(allocator, seqs_arg, ngen, model_path);
    } else if (std.mem.eql(u8, first, "sched")) {
        // Effort 28 increment 2: continuous-batching scheduler proof. '|' separates
        // sequences, ',' separates prompt token-ids. nslots < nseq forces slot reuse.
        const seqs_arg = args.next() orelse "760,6511,314,9338,369|450,3271,310,3444,338|1102,323,1023,1024|99,100,101,102,103";
        const ngen: u32 = std.fmt.parseInt(u32, args.next() orelse "8", 10) catch 8;
        const nslots: u32 = std.fmt.parseInt(u32, args.next() orelse "2", 10) catch 2;
        const model_path = args.next() orelse DEFAULT_MODEL;
        try schedMode(allocator, seqs_arg, ngen, nslots, model_path);
    } else if (std.mem.eql(u8, first, "serve")) {
        // Effort 28 increment 3 (3a): concurrent serving engine proof. ONE GPU
        // worker thread drives the admit→prefill→decodeBatch→evict loop; N producer
        // threads enqueue independently and receive their own token stream via a
        // mutex+condvar handoff. Same args as `sched` (nslots < nseq forces reuse).
        const seqs_arg = args.next() orelse "760,6511,314,9338,369|450,3271,310,3444,338|1102,323,1023,1024|99,100,101,102,103";
        const ngen: u32 = std.fmt.parseInt(u32, args.next() orelse "8", 10) catch 8;
        const nslots: u32 = std.fmt.parseInt(u32, args.next() orelse "2", 10) catch 2;
        const model_path = args.next() orelse DEFAULT_MODEL;
        try serveMode(allocator, seqs_arg, ngen, nslots, model_path);
    } else if (std.mem.eql(u8, first, "prof")) {
        const model_path = args.next() orelse DEFAULT_MODEL;
        try profMode(allocator, model_path);
    } else if (std.mem.eql(u8, first, "bench")) {
        const iters: i32 = std.fmt.parseInt(i32, args.next() orelse "2000", 10) catch 2000;
        const n: u32 = std.fmt.parseInt(u32, args.next() orelse "300", 10) catch 300;
        try benchMode(allocator, iters, n);
    } else if (std.mem.eql(u8, first, "mma8")) {
        const iters: u32 = std.fmt.parseInt(u32, args.next() orelse "20000", 10) catch 20000;
        try mma8Mode(allocator, iters);
    } else if (std.mem.eql(u8, first, "gemm8")) {
        const M: u32 = std.fmt.parseInt(u32, args.next() orelse "4608", 10) catch 4608;
        const K: u32 = std.fmt.parseInt(u32, args.next() orelse "4608", 10) catch 4608;
        const T: u32 = std.fmt.parseInt(u32, args.next() orelse "512", 10) catch 512;
        try gemm8Mode(allocator, M, K, T);
    } else if (std.mem.eql(u8, first, "logits")) {
        const token: u32 = std.fmt.parseInt(u32, args.next() orelse "100", 10) catch 100;
        const out_path = args.next() orelse "/tmp/zinc_logits.bin";
        const model_path = args.next() orelse DEFAULT_MODEL;
        try logitsMode(allocator, token, out_path, model_path);
    } else if (std.mem.eql(u8, first, "gdump")) {
        const ids_arg = args.next() orelse "1000";
        const out_path = args.next() orelse "/tmp/zinc_layers.bin";
        const model_path = args.next() orelse DEFAULT_MODEL;
        try gemmaLayerDumpMode(allocator, ids_arg, out_path, model_path);
    } else if (std.mem.eql(u8, first, "glogits")) {
        const ids_arg = args.next() orelse "1000";
        const out_path = args.next() orelse "/tmp/zinc_glogits.bin";
        const model_path = args.next() orelse DEFAULT_MODEL;
        try gemmaLogitsMode(allocator, ids_arg, out_path, model_path);
    } else if (std.mem.eql(u8, first, "tf")) {
        const prompt_arg = args.next() orelse "1000";
        const gen_arg = args.next() orelse "";
        const model_path = args.next() orelse DEFAULT_MODEL;
        try teacherForcedMode(allocator, prompt_arg, gen_arg, model_path);
    } else {
        const token: u32 = std.fmt.parseInt(u32, first, 10) catch 100;
        const model_path = args.next() orelse DEFAULT_MODEL;
        try dumpMode(allocator, token, model_path);
    }
}

/// Autoregressive greedy generation from a comma-separated prompt token list.
fn genMode(allocator: std.mem.Allocator, ids_arg: []const u8, ngen: u32, model_path: []const u8) !void {
    var prompt_buf: [256]u32 = undefined;
    var np: usize = 0;
    var it = std.mem.splitScalar(u8, ids_arg, ',');
    while (it.next()) |s| {
        const trimmed = std.mem.trim(u8, s, " ");
        if (trimmed.len == 0 or np >= prompt_buf.len) continue;
        prompt_buf[np] = try std.fmt.parseInt(u32, trimmed, 10);
        np += 1;
    }
    const prompt = prompt_buf[0..np];

    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try Engine.init(allocator, &model, 512);
    defer fwd.deinit();

    std.debug.print("PROMPT_IDS:", .{});
    for (prompt, 0..) |t, i| std.debug.print("{s}{d}", .{ if (i == 0) "" else ",", t });
    std.debug.print("\n", .{});

    // Prefill: process each prompt token at its position; the argmax after the
    // last prompt token is the first generated token. Prompt-internal tokens
    // need no logits, so ZINC_PREFILL_SKIP=1 runs them via prefillStep (skips
    // the LM head, bit-identical generation) to A/B the head-skip prefill win.
    const pf_skip = std.posix.getenv("ZINC_PREFILL_SKIP") != null;
    // Effort 25: batched-GEMM prefill is the DEFAULT for gemma (qwen falls back
    // via error.Unsupported). ZINC_BATCHED_PREFILL=0/off opts out to per-token.
    // scripts/prefill_catalog.sh forces the opt-out on its baseline arm so the
    // batched-vs-per-token A/B (GEN_IDS must be byte-identical — the gate) still
    // measures the real delta after the default flip.
    const pf_batched = batchedPrefillDefaultOn();
    var pos: u32 = 0;
    var tok: u32 = 0;
    var pf_timer = try std.time.Timer.start();
    var pf_used_batched = false;
    if (pf_batched and prompt.len > 1) {
        if (fwd.prefillBatched(prompt)) |first| {
            tok = first;
            pos = @intCast(prompt.len);
            pf_used_batched = true;
        } else |_| {} // unsupported (qwen) → fall back to the per-token loop
    }
    if (!pf_used_batched) {
        for (prompt, 0..) |t, i| {
            if (pf_skip and i + 1 < prompt.len) {
                try fwd.prefillStep(t, pos);
            } else {
                tok = try fwd.decodeStep(t, pos, true);
            }
            pos += 1;
        }
    }
    const pf_ns = pf_timer.read();
    if (prompt.len > 1) {
        const pf_secs = @as(f64, @floatFromInt(pf_ns)) / 1e9;
        std.debug.print("PREFILL: {d} tokens in {d:.3}s = {d:.2} tok/s (skip={} batched={})\n", .{ prompt.len, pf_secs, @as(f64, @floatFromInt(prompt.len)) / pf_secs, pf_skip, pf_used_batched });
    }

    std.debug.print("GEN_IDS:{d}", .{tok});
    var timer = try std.time.Timer.start(); // steady-state: exclude prefill + first token
    var g: u32 = 1;
    while (g < ngen) : (g += 1) {
        const next = try fwd.decodeStep(tok, pos, true);
        pos += 1;
        std.debug.print(",{d}", .{next});
        tok = next;
    }
    const ns = timer.read();
    std.debug.print("\n", .{});
    if (ngen > 1) {
        const secs = @as(f64, @floatFromInt(ns)) / 1e9;
        const steps: f64 = @floatFromInt(ngen - 1);
        std.debug.print("DECODE: {d} tokens in {d:.3}s = {d:.2} tok/s (correctness-first, sync-per-layer)\n", .{ ngen - 1, secs, steps / secs });
    }
}

/// Effort 28 increment 1, sub-step 1a — multi-sequence batched-serving harness.
///
/// Generates each of B sequences (prompts separated by '|', ids by ',')
/// INDEPENDENTLY through the production single-sequence path (prefillBatched /
/// decodeStep over the shared kv_k cache, which each sequence overwrites from
/// pos 0 — sound because every sequence reads only positions it wrote). Emits
/// `BATCH_SEQ{j}:tok,tok,...` per sequence. This is the SERIAL REFERENCE that
/// the future `decodeBatch` proof (sub-step 1d) must reproduce token-identically.
///
/// It also exercises the NEW slot-based KV plumbing (gemma only): allocate one
/// slot per sequence and run `slotKvSmoke` to validate the slot-offset
/// arithmetic the batched forward (1b/1c) will depend on. Additive — the
/// production decode path is untouched.
fn batchMode(allocator: std.mem.Allocator, seqs_arg: []const u8, ngen: u32, model_path: []const u8) !void {
    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try Engine.init(allocator, &model, 512);
    defer fwd.deinit();

    const pf_batched = batchedPrefillDefaultOn();

    const MAXB = 27; // max concurrent sequences in this harness (btok covers B≤27)
    const MAXP = 256; // max prompt tokens / sequence
    const MAXG = 64; // max generated tokens / sequence
    var prompts: [MAXB][MAXP]u32 = undefined;
    var plens: [MAXB]usize = undefined;
    var serial_out: [MAXB][MAXG]u32 = undefined; // production single-seq reference (decodeStep)
    var nseq: u32 = 0;
    const ng = @min(ngen, @as(u32, MAXG));

    var seq_it = std.mem.splitScalar(u8, seqs_arg, '|');
    while (seq_it.next()) |seq_str| {
        if (nseq >= MAXB) break;
        const seq_trim = std.mem.trim(u8, seq_str, " ");
        if (seq_trim.len == 0) continue;

        var np: usize = 0;
        var it = std.mem.splitScalar(u8, seq_trim, ',');
        while (it.next()) |s| {
            const t = std.mem.trim(u8, s, " ");
            if (t.len == 0 or np >= MAXP) continue;
            prompts[nseq][np] = try std.fmt.parseInt(u32, t, 10);
            np += 1;
        }
        if (np == 0) continue;
        plens[nseq] = np;
        const prompt = prompts[nseq][0..np];

        // SERIAL REFERENCE: generate this sequence on its own via the production
        // single-sequence path (prefillBatched + decodeStep over the shared cache).
        // Reset the recurrent state first so each reference is TRULY single-sequence
        // (qwen's unindexed SSM state would otherwise leak from the prior sequence;
        // no-op for gemma's position-indexed KV).
        try fwd.resetState();
        var pos: u32 = 0;
        var tok: u32 = 0;
        var used_batched = false;
        if (pf_batched and prompt.len > 1) {
            if (fwd.prefillBatched(prompt)) |first| {
                tok = first;
                pos = @intCast(prompt.len);
                used_batched = true;
            } else |_| {}
        }
        if (!used_batched) {
            for (prompt) |t| {
                tok = try fwd.decodeStep(t, pos, true);
                pos += 1;
            }
        }
        serial_out[nseq][0] = tok;
        std.debug.print("BATCH_SEQ{d}:{d}", .{ nseq, tok });
        var gi: u32 = 1;
        while (gi < ng) : (gi += 1) {
            const next = try fwd.decodeStep(tok, pos, true);
            pos += 1;
            serial_out[nseq][gi] = next;
            std.debug.print(",{d}", .{next});
            tok = next;
        }
        std.debug.print("\n", .{});
        nseq += 1;
    }

    // Slot-KV smoke + the sub-step 1b batched-DECODE proof (gemma DENSE only).
    switch (fwd) {
        .gemma => |*g| {
            const slot_ctx: u32 = 512;
            const slots_n = if (nseq == 0) 1 else nseq;
            try g.allocSlotKv(slots_n, slot_ctx);
            defer g.freeSlotKv();
            const ok = g.slotKvSmoke() catch |e| {
                std.debug.print("SLOTKV_SMOKE:ERR {s}\n", .{@errorName(e)});
                return;
            };
            std.debug.print("SLOTKV_SMOKE:{s} (slots={d} slot_ctx={d})\n", .{ if (ok) "ok" else "FAIL", slots_n, slot_ctx });
            if (nseq == 0 or g.d.n_experts > 0) {
                std.debug.print("BATCHDEC:skip ({s})\n", .{if (g.d.n_experts > 0) "MoE — increment 1 is dense gemma" else "no sequences"});
                return;
            }

            // The smoke wrote sentinels into layer-0's K slot — realloc clean KV so
            // every sequence's history starts fresh (each reads only what it wrote).
            try g.allocSlotKv(nseq, slot_ctx);

            // PASS A — SOLO: run each sequence ALONE through decodeBatch (B=1) into
            // its own slot. This is the same numeric path as the batched run, so the
            // batched output must equal it token-for-token (the isolation gate). It
            // is also the "B=1 decodeBatch == gen" sanity vs serial_out.
            var solo_out: [MAXB][MAXG]u32 = undefined;
            var j: u32 = 0;
            while (j < nseq) : (j += 1) {
                const np = plens[j];
                var pos: u32 = 0;
                var tok: u32 = 0;
                var k: usize = 0;
                while (k < np) : (k += 1) { // per-token B=1 prefill into slot j
                    var tk = [_]u32{prompts[j][k]};
                    var ps = [_]u32{pos};
                    var sl = [_]u32{j};
                    var ot = [_]u32{0};
                    try g.decodeBatch(&tk, &ps, &sl, &ot);
                    tok = ot[0];
                    pos += 1;
                }
                solo_out[j][0] = tok;
                var s: u32 = 1;
                while (s < ng) : (s += 1) {
                    var tk = [_]u32{tok};
                    var ps = [_]u32{pos};
                    var sl = [_]u32{j};
                    var ot = [_]u32{0};
                    try g.decodeBatch(&tk, &ps, &sl, &ot);
                    tok = ot[0];
                    pos += 1;
                    solo_out[j][s] = tok;
                }
            }

            // PASS B — BATCHED: reset the slots, prefill each into its slot (B=1),
            // then decode ALL sequences TOGETHER each step (mixed positions, since
            // prompt lengths differ). This exercises per-sequence positions/slots.
            try g.allocSlotKv(nseq, slot_ctx);
            var batched_out: [MAXB][MAXG]u32 = undefined;
            var cur_tok: [MAXB]u32 = undefined;
            var cur_pos: [MAXB]u32 = undefined;
            j = 0;
            while (j < nseq) : (j += 1) {
                const np = plens[j];
                var pos: u32 = 0;
                var tok: u32 = 0;
                var k: usize = 0;
                while (k < np) : (k += 1) {
                    var tk = [_]u32{prompts[j][k]};
                    var ps = [_]u32{pos};
                    var sl = [_]u32{j};
                    var ot = [_]u32{0};
                    try g.decodeBatch(&tk, &ps, &sl, &ot);
                    tok = ot[0];
                    pos += 1;
                }
                batched_out[j][0] = tok;
                cur_tok[j] = tok;
                cur_pos[j] = pos;
            }
            var step: u32 = 1;
            while (step < ng) : (step += 1) {
                var tks: [MAXB]u32 = undefined;
                var pss: [MAXB]u32 = undefined;
                var sls: [MAXB]u32 = undefined;
                var out: [MAXB]u32 = undefined;
                j = 0;
                while (j < nseq) : (j += 1) {
                    tks[j] = cur_tok[j];
                    pss[j] = cur_pos[j];
                    sls[j] = j;
                }
                try g.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                j = 0;
                while (j < nseq) : (j += 1) {
                    batched_out[j][step] = out[j];
                    cur_tok[j] = out[j];
                    cur_pos[j] += 1;
                }
            }

            // Emit + compare. GATE: batched == solo (same numeric path → isolation
            // proof). SANITY: solo == serial (decodeBatch B=1 == production gen).
            var gate_pass = true;
            var sanity_pass = true;
            j = 0;
            while (j < nseq) : (j += 1) {
                std.debug.print("BATCHDEC_SEQ{d}:{d}", .{ j, batched_out[j][0] });
                var s: u32 = 1;
                while (s < ng) : (s += 1) std.debug.print(",{d}", .{batched_out[j][s]});
                var gmatch = true;
                var smatch = true;
                s = 0;
                while (s < ng) : (s += 1) {
                    if (batched_out[j][s] != solo_out[j][s]) gmatch = false;
                    if (solo_out[j][s] != serial_out[j][s]) smatch = false;
                }
                if (!gmatch) gate_pass = false;
                if (!smatch) sanity_pass = false;
                std.debug.print(" [gate={s} sanity={s}]\n", .{ if (gmatch) "MATCH" else "DIFF", if (smatch) "MATCH" else "DIFF" });
            }
            std.debug.print("BATCH_GATE:{s} BATCH_SANITY:{s} (nseq={d} ngen={d})\n", .{ if (gate_pass) "PASS" else "FAIL", if (sanity_pass) "PASS" else "FAIL", nseq, ng });

            // Effort 28 perf A/B: time NG steady-state B=1 decodeBatch steps with
            // the matvec fast path OFF then ON in ONE model load (boost-comparable).
            // Token-identity is already gated above (PASS-A-solo runs B=1) — this
            // just reports the per-stream speedup the fast path buys.
            {
                var which: u32 = 0;
                while (which < 2) : (which += 1) {
                    g.decode_b1_force = (which == 1);
                    try g.allocSlotKv(1, slot_ctx); // fresh slot 0
                    const np = plens[0];
                    var pos: u32 = 0;
                    var tok: u32 = 0;
                    var k: usize = 0;
                    while (k < np) : (k += 1) { // prefill seq0 into slot 0 (B=1)
                        var tk = [_]u32{prompts[0][k]};
                        var ps = [_]u32{pos};
                        var sl = [_]u32{0};
                        var ot = [_]u32{0};
                        try g.decodeBatch(&tk, &ps, &sl, &ot);
                        tok = ot[0];
                        pos += 1;
                    }
                    var w: u32 = 0; // warm a few steps before timing
                    while (w < 3) : (w += 1) {
                        var tk = [_]u32{tok};
                        var ps = [_]u32{pos};
                        var sl = [_]u32{0};
                        var ot = [_]u32{0};
                        try g.decodeBatch(&tk, &ps, &sl, &ot);
                        tok = ot[0];
                        pos += 1;
                    }
                    var timer = try std.time.Timer.start();
                    var s: u32 = 0;
                    while (s < ng) : (s += 1) {
                        var tk = [_]u32{tok};
                        var ps = [_]u32{pos};
                        var sl = [_]u32{0};
                        var ot = [_]u32{0};
                        try g.decodeBatch(&tk, &ps, &sl, &ot);
                        tok = ot[0];
                        pos += 1;
                    }
                    const ns = timer.read();
                    const tps = @as(f64, @floatFromInt(ng)) * 1e9 / @as(f64, @floatFromInt(ns));
                    std.debug.print("B1_TIMING matvec={s}: {d:.2} tok/s ({d} steps)\n", .{ if (which == 1) "ON " else "OFF", tps, ng });
                }
                g.decode_b1_force = null;
            }

            // Effort 28 perf A/B (gemma port of the qwen BTOK_TIMING): time NG
            // steady-state BATCHED decodeBatch steps (B=nseq) with the Q4_K
            // token-batch matvec (`dmmv_q4k_btok`) OFF then ON in ONE model load
            // (boost-comparable). Token-identity is gated by BATCH_GATE above (run
            // with ZINC_BATCH_MROW=1 to exercise btok there). Reports the AGGREGATE
            // decode throughput btok buys at this batch size vs the 64-tile GEMM.
            if (nseq >= 2 and nseq <= 27) {
                var which: u32 = 0;
                while (which < 2) : (which += 1) {
                    g.decode_mrow_force = (which == 1);
                    try g.allocSlotKv(nseq, slot_ctx);
                    var ct: [MAXB]u32 = undefined;
                    var cp: [MAXB]u32 = undefined;
                    j = 0;
                    while (j < nseq) : (j += 1) { // prefill seq j into slot j (B=1)
                        const np = plens[j];
                        var pos: u32 = 0;
                        var tok: u32 = 0;
                        var k: usize = 0;
                        while (k < np) : (k += 1) {
                            var tk = [_]u32{prompts[j][k]};
                            var ps = [_]u32{pos};
                            var sl = [_]u32{@intCast(j)};
                            var ot = [_]u32{0};
                            try g.decodeBatch(&tk, &ps, &sl, &ot);
                            tok = ot[0];
                            pos += 1;
                        }
                        ct[j] = tok;
                        cp[j] = pos;
                    }
                    var w: u32 = 0; // warm a few batched steps before timing
                    while (w < 3) : (w += 1) {
                        var tks: [MAXB]u32 = undefined;
                        var pss: [MAXB]u32 = undefined;
                        var sls: [MAXB]u32 = undefined;
                        var out: [MAXB]u32 = undefined;
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            tks[j] = ct[j];
                            pss[j] = cp[j];
                            sls[j] = @intCast(j);
                        }
                        try g.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            ct[j] = out[j];
                            cp[j] += 1;
                        }
                    }
                    var timer = try std.time.Timer.start();
                    var s: u32 = 0;
                    while (s < ng) : (s += 1) {
                        var tks: [MAXB]u32 = undefined;
                        var pss: [MAXB]u32 = undefined;
                        var sls: [MAXB]u32 = undefined;
                        var out: [MAXB]u32 = undefined;
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            tks[j] = ct[j];
                            pss[j] = cp[j];
                            sls[j] = @intCast(j);
                        }
                        try g.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            ct[j] = out[j];
                            cp[j] += 1;
                        }
                    }
                    const ns = timer.read();
                    const tot = @as(f64, @floatFromInt(ng * nseq));
                    const tps = tot * 1e9 / @as(f64, @floatFromInt(ns));
                    std.debug.print("BTOK_TIMING mrow={s} B={d}: {d:.2} tok/s agg ({d} steps)\n", .{ if (which == 1) "ON " else "OFF", nseq, tps, ng });
                }
                g.decode_mrow_force = null;
            }
        },
        .qwen => |*q| {
            // Inc 4 sub-step 4b: batched DECODE for qwen (hybrid-SSM). First the 4a
            // slot-state smoke (KV + SSM conv + recurrent non-overlap), then the
            // SAME PASS-A-solo / PASS-B-batched proof as gemma against `decodeBatch`.
            // The BATCH_SEQ lines above ARE the qwen serial reference.
            const slot_ctx: u32 = 512;
            const slots_n = @max(@as(u32, 2), if (nseq == 0) @as(u32, 2) else nseq);
            try q.allocSlotState(slots_n, slot_ctx);
            defer q.freeSlotState();
            const ok = q.slotStateSmoke() catch |e| {
                std.debug.print("SLOTSTATE_SMOKE:ERR {s}\n", .{@errorName(e)});
                return;
            };
            std.debug.print("SLOTSTATE_SMOKE:{s} (slots={d} slot_ctx={d})\n", .{ if (ok) "ok" else "FAIL", slots_n, slot_ctx });
            if (nseq == 0) {
                std.debug.print("BATCHDEC:skip (no sequences)\n", .{});
                return;
            }

            // PASS A — SOLO: each sequence ALONE through decodeBatch (B=1) into its
            // own slot. Same numeric path as the batched run → batched must equal it
            // token-for-token (isolation gate); also the B=1==serial sanity.
            try q.allocSlotState(nseq, slot_ctx); // smoke wrote sentinels — reset state
            var solo_out: [MAXB][MAXG]u32 = undefined;
            var j: u32 = 0;
            while (j < nseq) : (j += 1) {
                const np = plens[j];
                var pos: u32 = 0;
                var tok: u32 = 0;
                var k: usize = 0;
                while (k < np) : (k += 1) { // per-token B=1 prefill into slot j
                    var tk = [_]u32{prompts[j][k]};
                    var ps = [_]u32{pos};
                    var sl = [_]u32{j};
                    var ot = [_]u32{0};
                    try q.decodeBatch(&tk, &ps, &sl, &ot);
                    tok = ot[0];
                    pos += 1;
                }
                solo_out[j][0] = tok;
                var s: u32 = 1;
                while (s < ng) : (s += 1) {
                    var tk = [_]u32{tok};
                    var ps = [_]u32{pos};
                    var sl = [_]u32{j};
                    var ot = [_]u32{0};
                    try q.decodeBatch(&tk, &ps, &sl, &ot);
                    tok = ot[0];
                    pos += 1;
                    solo_out[j][s] = tok;
                }
            }

            // PASS B — BATCHED: reset slots, prefill each into its slot (B=1), then
            // decode ALL sequences TOGETHER each step (mixed positions, since prompt
            // lengths differ) — exercises per-sequence positions/slots/SSM state.
            try q.allocSlotState(nseq, slot_ctx);
            var batched_out: [MAXB][MAXG]u32 = undefined;
            var cur_tok: [MAXB]u32 = undefined;
            var cur_pos: [MAXB]u32 = undefined;
            j = 0;
            while (j < nseq) : (j += 1) {
                const np = plens[j];
                var pos: u32 = 0;
                var tok: u32 = 0;
                var k: usize = 0;
                while (k < np) : (k += 1) {
                    var tk = [_]u32{prompts[j][k]};
                    var ps = [_]u32{pos};
                    var sl = [_]u32{j};
                    var ot = [_]u32{0};
                    try q.decodeBatch(&tk, &ps, &sl, &ot);
                    tok = ot[0];
                    pos += 1;
                }
                batched_out[j][0] = tok;
                cur_tok[j] = tok;
                cur_pos[j] = pos;
            }
            var step: u32 = 1;
            while (step < ng) : (step += 1) {
                var tks: [MAXB]u32 = undefined;
                var pss: [MAXB]u32 = undefined;
                var sls: [MAXB]u32 = undefined;
                var out: [MAXB]u32 = undefined;
                j = 0;
                while (j < nseq) : (j += 1) {
                    tks[j] = cur_tok[j];
                    pss[j] = cur_pos[j];
                    sls[j] = j;
                }
                try q.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                j = 0;
                while (j < nseq) : (j += 1) {
                    batched_out[j][step] = out[j];
                    cur_tok[j] = out[j];
                    cur_pos[j] += 1;
                }
            }

            // GATE: batched == solo (isolation). SANITY: solo == serial (B=1 == gen).
            var gate_pass = true;
            var sanity_pass = true;
            j = 0;
            while (j < nseq) : (j += 1) {
                std.debug.print("BATCHDEC_SEQ{d}:{d}", .{ j, batched_out[j][0] });
                var s: u32 = 1;
                while (s < ng) : (s += 1) std.debug.print(",{d}", .{batched_out[j][s]});
                var gmatch = true;
                var smatch = true;
                s = 0;
                while (s < ng) : (s += 1) {
                    if (batched_out[j][s] != solo_out[j][s]) gmatch = false;
                    if (solo_out[j][s] != serial_out[j][s]) smatch = false;
                }
                if (!gmatch) gate_pass = false;
                if (!smatch) sanity_pass = false;
                std.debug.print(" [gate={s} sanity={s}]\n", .{ if (gmatch) "MATCH" else "DIFF", if (smatch) "MATCH" else "DIFF" });
            }
            std.debug.print("BATCH_GATE:{s} BATCH_SANITY:{s} (nseq={d} ngen={d})\n", .{ if (gate_pass) "PASS" else "FAIL", if (sanity_pass) "PASS" else "FAIL", nseq, ng });

            // Effort 28 perf A/B (qwen analog of the gemma B1_TIMING): time NG
            // steady-state B=1 decodeBatch steps with the matvec fast path OFF
            // then ON in ONE model load (boost-comparable). Token-identity is
            // already gated above (PASS-A-solo runs B=1) — this reports the
            // per-stream speedup the fast path buys for qwen.
            {
                var which: u32 = 0;
                while (which < 2) : (which += 1) {
                    q.decode_b1_force = (which == 1);
                    try q.allocSlotState(1, slot_ctx); // fresh slot 0
                    const np = plens[0];
                    var pos: u32 = 0;
                    var tok: u32 = 0;
                    var k: usize = 0;
                    while (k < np) : (k += 1) { // prefill seq0 into slot 0 (B=1)
                        var tk = [_]u32{prompts[0][k]};
                        var ps = [_]u32{pos};
                        var sl = [_]u32{0};
                        var ot = [_]u32{0};
                        try q.decodeBatch(&tk, &ps, &sl, &ot);
                        tok = ot[0];
                        pos += 1;
                    }
                    var w: u32 = 0; // warm a few steps before timing
                    while (w < 3) : (w += 1) {
                        var tk = [_]u32{tok};
                        var ps = [_]u32{pos};
                        var sl = [_]u32{0};
                        var ot = [_]u32{0};
                        try q.decodeBatch(&tk, &ps, &sl, &ot);
                        tok = ot[0];
                        pos += 1;
                    }
                    var timer = try std.time.Timer.start();
                    var s: u32 = 0;
                    while (s < ng) : (s += 1) {
                        var tk = [_]u32{tok};
                        var ps = [_]u32{pos};
                        var sl = [_]u32{0};
                        var ot = [_]u32{0};
                        try q.decodeBatch(&tk, &ps, &sl, &ot);
                        tok = ot[0];
                        pos += 1;
                    }
                    const ns = timer.read();
                    const tps = @as(f64, @floatFromInt(ng)) * 1e9 / @as(f64, @floatFromInt(ns));
                    std.debug.print("B1_TIMING matvec={s}: {d:.2} tok/s ({d} steps)\n", .{ if (which == 1) "ON " else "OFF", tps, ng });
                }
                q.decode_b1_force = null;
            }

            // Effort 28 perf A/B — time NG steady-state BATCHED decodeBatch steps
            // (B=nseq) with the Q4_K token-batch matvec (`dmmv_q4k_btok`) OFF then
            // ON in ONE model load (boost-comparable). Token-identity is gated by
            // BATCH_GATE above (run with ZINC_BATCH_MROW=1 to exercise btok there).
            // Reports the AGGREGATE decode throughput btok buys at this batch size
            // vs the 64-tile batched GEMM.
            if (nseq >= 2 and nseq <= 27) {
                var which: u32 = 0;
                while (which < 2) : (which += 1) {
                    q.decode_mrow_force = (which == 1);
                    try q.allocSlotState(nseq, slot_ctx);
                    var ct: [MAXB]u32 = undefined;
                    var cp: [MAXB]u32 = undefined;
                    j = 0;
                    while (j < nseq) : (j += 1) { // prefill seq j into slot j (B=1)
                        const np = plens[j];
                        var pos: u32 = 0;
                        var tok: u32 = 0;
                        var k: usize = 0;
                        while (k < np) : (k += 1) {
                            var tk = [_]u32{prompts[j][k]};
                            var ps = [_]u32{pos};
                            var sl = [_]u32{@intCast(j)};
                            var ot = [_]u32{0};
                            try q.decodeBatch(&tk, &ps, &sl, &ot);
                            tok = ot[0];
                            pos += 1;
                        }
                        ct[j] = tok;
                        cp[j] = pos;
                    }
                    var w: u32 = 0; // warm a few batched steps before timing
                    while (w < 3) : (w += 1) {
                        var tks: [MAXB]u32 = undefined;
                        var pss: [MAXB]u32 = undefined;
                        var sls: [MAXB]u32 = undefined;
                        var out: [MAXB]u32 = undefined;
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            tks[j] = ct[j];
                            pss[j] = cp[j];
                            sls[j] = @intCast(j);
                        }
                        try q.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            ct[j] = out[j];
                            cp[j] += 1;
                        }
                    }
                    var timer = try std.time.Timer.start();
                    var s: u32 = 0;
                    while (s < ng) : (s += 1) {
                        var tks: [MAXB]u32 = undefined;
                        var pss: [MAXB]u32 = undefined;
                        var sls: [MAXB]u32 = undefined;
                        var out: [MAXB]u32 = undefined;
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            tks[j] = ct[j];
                            pss[j] = cp[j];
                            sls[j] = @intCast(j);
                        }
                        try q.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            ct[j] = out[j];
                            cp[j] += 1;
                        }
                    }
                    const ns = timer.read();
                    const tot = @as(f64, @floatFromInt(ng * nseq));
                    const tps = tot * 1e9 / @as(f64, @floatFromInt(ns));
                    std.debug.print("BTOK_TIMING mrow={s} B={d}: {d:.2} tok/s agg ({d} steps)\n", .{ if (which == 1) "ON " else "OFF", nseq, tps, ng });
                }
                q.decode_mrow_force = null;
            }

            // Effort 28 DENSE launch-collapse perf A/B — time NG steady-state BATCHED
            // decodeBatch steps (B=nseq) with the per-layer batched blocks' commitAndWait
            // RESTORED (collapse OFF) then dropped to async submit + ONE tail drain
            // (collapse ON) in ONE model load (boost-comparable). mrow is held ON for
            // both arms (the default serving path) so the A/B isolates the per-layer
            // sync removal. Token-identity is gated by BATCH_GATE above.
            if (nseq >= 2 and nseq <= 8) {
                var which: u32 = 0;
                while (which < 2) : (which += 1) {
                    q.decode_mrow_force = true;
                    q.decode_collapse_force = (which == 1);
                    try q.allocSlotState(nseq, slot_ctx);
                    var ct: [MAXB]u32 = undefined;
                    var cp: [MAXB]u32 = undefined;
                    j = 0;
                    while (j < nseq) : (j += 1) { // prefill seq j into slot j (B=1)
                        const np = plens[j];
                        var pos: u32 = 0;
                        var tok: u32 = 0;
                        var k: usize = 0;
                        while (k < np) : (k += 1) {
                            var tk = [_]u32{prompts[j][k]};
                            var ps = [_]u32{pos};
                            var sl = [_]u32{@intCast(j)};
                            var ot = [_]u32{0};
                            try q.decodeBatch(&tk, &ps, &sl, &ot);
                            tok = ot[0];
                            pos += 1;
                        }
                        ct[j] = tok;
                        cp[j] = pos;
                    }
                    var w: u32 = 0; // warm a few batched steps before timing
                    while (w < 3) : (w += 1) {
                        var tks: [MAXB]u32 = undefined;
                        var pss: [MAXB]u32 = undefined;
                        var sls: [MAXB]u32 = undefined;
                        var out: [MAXB]u32 = undefined;
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            tks[j] = ct[j];
                            pss[j] = cp[j];
                            sls[j] = @intCast(j);
                        }
                        try q.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            ct[j] = out[j];
                            cp[j] += 1;
                        }
                    }
                    var timer = try std.time.Timer.start();
                    var s: u32 = 0;
                    while (s < ng) : (s += 1) {
                        var tks: [MAXB]u32 = undefined;
                        var pss: [MAXB]u32 = undefined;
                        var sls: [MAXB]u32 = undefined;
                        var out: [MAXB]u32 = undefined;
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            tks[j] = ct[j];
                            pss[j] = cp[j];
                            sls[j] = @intCast(j);
                        }
                        try q.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            ct[j] = out[j];
                            cp[j] += 1;
                        }
                    }
                    const ns = timer.read();
                    const tot = @as(f64, @floatFromInt(ng * nseq));
                    const tps = tot * 1e9 / @as(f64, @floatFromInt(ns));
                    std.debug.print("DCOLLAPSE_TIMING collapse={s} B={d}: {d:.2} tok/s agg ({d} steps)\n", .{ if (which == 1) "ON " else "OFF", nseq, tps, ng });
                }
                q.decode_mrow_force = null;
                q.decode_collapse_force = null;
            }

            // Effort 28 CUDA-graph perf A/B — time NG steady-state BATCHED decodeBatch
            // steps (B=nseq) with the dense batched-decode CUDA-graph replay OFF (the
            // async submit chain) then ON (one captured graph launch per step) in ONE
            // model load (boost-comparable). mrow held ON for both arms (the default
            // serving path) so the A/B isolates the graph capture. Dense OR MoE — MoE
            // is capturable when `moe_graph_capturable` (every layer reads expert ids
            // GPU-side, no host readback) AND mrow on (→ moe_shared_batched +
            // moe_collapse, the no-sync batched routed+shared path). Token-identity is
            // validated by re-running the GATE above under ZINC_BATCH_GRAPH=1 (graph
            // batched == solo == serial). The win needs a CLEAN 5090 window to claim.
            if ((q.d.n_experts == 0 or q.moe_graph_capturable) and nseq >= 2 and nseq <= 8) {
                var which: u32 = 0;
                while (which < 2) : (which += 1) {
                    q.decode_mrow_force = true;
                    q.batch_graph_force = (which == 1);
                    try q.allocSlotState(nseq, slot_ctx);
                    var ct: [MAXB]u32 = undefined;
                    var cp: [MAXB]u32 = undefined;
                    j = 0;
                    while (j < nseq) : (j += 1) { // prefill seq j into slot j (B=1)
                        const np = plens[j];
                        var pos: u32 = 0;
                        var tok: u32 = 0;
                        var k: usize = 0;
                        while (k < np) : (k += 1) {
                            var tk = [_]u32{prompts[j][k]};
                            var ps = [_]u32{pos};
                            var sl = [_]u32{@intCast(j)};
                            var ot = [_]u32{0};
                            try q.decodeBatch(&tk, &ps, &sl, &ot);
                            tok = ot[0];
                            pos += 1;
                        }
                        ct[j] = tok;
                        cp[j] = pos;
                    }
                    var w: u32 = 0; // warm a few batched steps before timing
                    while (w < 3) : (w += 1) {
                        var tks: [MAXB]u32 = undefined;
                        var pss: [MAXB]u32 = undefined;
                        var sls: [MAXB]u32 = undefined;
                        var out: [MAXB]u32 = undefined;
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            tks[j] = ct[j];
                            pss[j] = cp[j];
                            sls[j] = @intCast(j);
                        }
                        try q.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            ct[j] = out[j];
                            cp[j] += 1;
                        }
                    }
                    var timer = try std.time.Timer.start();
                    var s: u32 = 0;
                    while (s < ng) : (s += 1) {
                        var tks: [MAXB]u32 = undefined;
                        var pss: [MAXB]u32 = undefined;
                        var sls: [MAXB]u32 = undefined;
                        var out: [MAXB]u32 = undefined;
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            tks[j] = ct[j];
                            pss[j] = cp[j];
                            sls[j] = @intCast(j);
                        }
                        try q.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            ct[j] = out[j];
                            cp[j] += 1;
                        }
                    }
                    const ns = timer.read();
                    const tot = @as(f64, @floatFromInt(ng * nseq));
                    const tps = tot * 1e9 / @as(f64, @floatFromInt(ns));
                    std.debug.print("GRAPH_TIMING graph={s} B={d}: {d:.2} tok/s agg ({d} steps)\n", .{ if (which == 1) "ON " else "OFF", nseq, tps, ng });
                }
                q.decode_mrow_force = null;
                q.batch_graph_force = null;
            }

            // Effort 28 MoE perf A/B — time NG steady-state BATCHED decodeBatch steps
            // (B=nseq) with the shared-expert batching OFF (per-row matvec, reads each
            // shared weight B×) then ON (one btok matvec each over B rows) in ONE
            // model load (boost-comparable). MoE models only (qwen36-35b-a3b).
            // MEANINGFUL ONLY WITH ZINC_BATCH_MROW=1 — the shared batching rides the
            // `decode_mrow`/btok gate, so without it both arms run the per-row path
            // (no-op A/B). Token-identity is gated by BATCH_GATE above.
            if (q.d.n_experts > 0 and nseq >= 2 and nseq <= 8) {
                var which: u32 = 0;
                while (which < 2) : (which += 1) {
                    q.moe_shared_batched_force = (which == 1);
                    try q.allocSlotState(nseq, slot_ctx);
                    var ct: [MAXB]u32 = undefined;
                    var cp: [MAXB]u32 = undefined;
                    j = 0;
                    while (j < nseq) : (j += 1) { // prefill seq j into slot j (B=1)
                        const np = plens[j];
                        var pos: u32 = 0;
                        var tok: u32 = 0;
                        var k: usize = 0;
                        while (k < np) : (k += 1) {
                            var tk = [_]u32{prompts[j][k]};
                            var ps = [_]u32{pos};
                            var sl = [_]u32{@intCast(j)};
                            var ot = [_]u32{0};
                            try q.decodeBatch(&tk, &ps, &sl, &ot);
                            tok = ot[0];
                            pos += 1;
                        }
                        ct[j] = tok;
                        cp[j] = pos;
                    }
                    var w: u32 = 0; // warm a few batched steps before timing
                    while (w < 3) : (w += 1) {
                        var tks: [MAXB]u32 = undefined;
                        var pss: [MAXB]u32 = undefined;
                        var sls: [MAXB]u32 = undefined;
                        var out: [MAXB]u32 = undefined;
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            tks[j] = ct[j];
                            pss[j] = cp[j];
                            sls[j] = @intCast(j);
                        }
                        try q.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            ct[j] = out[j];
                            cp[j] += 1;
                        }
                    }
                    var timer = try std.time.Timer.start();
                    var s: u32 = 0;
                    while (s < ng) : (s += 1) {
                        var tks: [MAXB]u32 = undefined;
                        var pss: [MAXB]u32 = undefined;
                        var sls: [MAXB]u32 = undefined;
                        var out: [MAXB]u32 = undefined;
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            tks[j] = ct[j];
                            pss[j] = cp[j];
                            sls[j] = @intCast(j);
                        }
                        try q.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            ct[j] = out[j];
                            cp[j] += 1;
                        }
                    }
                    const ns = timer.read();
                    const tot = @as(f64, @floatFromInt(ng * nseq));
                    const tps = tot * 1e9 / @as(f64, @floatFromInt(ns));
                    std.debug.print("MOE_TIMING shared_batched={s} B={d}: {d:.2} tok/s agg ({d} steps)\n", .{ if (which == 1) "ON " else "OFF", nseq, tps, ng });
                }
                q.moe_shared_batched_force = null;
            }

            // Effort 28 MoE launch-collapse A/B — with the batched MoE path forced ON
            // (shared-expert batching + routed experts), time NG steady-state batched
            // steps with the per-row routed sync ON (collapse OFF) then collapsed into
            // ONE layer-tail sync (collapse ON) in ONE model load (boost-comparable).
            // MEANINGFUL ONLY WITH ZINC_BATCH_MROW=1 (the batched MoE path rides the
            // btok gate). Isolates the launch-collapse from the shared-batching win.
            if (q.d.n_experts > 0 and nseq >= 2 and nseq <= 8) {
                q.decode_mrow_force = true;
                q.moe_shared_batched_force = true;
                var which: u32 = 0;
                while (which < 2) : (which += 1) {
                    q.moe_collapse_force = (which == 1);
                    try q.allocSlotState(nseq, slot_ctx);
                    var ct: [MAXB]u32 = undefined;
                    var cp: [MAXB]u32 = undefined;
                    j = 0;
                    while (j < nseq) : (j += 1) { // prefill seq j into slot j (B=1)
                        const np = plens[j];
                        var pos: u32 = 0;
                        var tok: u32 = 0;
                        var k: usize = 0;
                        while (k < np) : (k += 1) {
                            var tk = [_]u32{prompts[j][k]};
                            var ps = [_]u32{pos};
                            var sl = [_]u32{@intCast(j)};
                            var ot = [_]u32{0};
                            try q.decodeBatch(&tk, &ps, &sl, &ot);
                            tok = ot[0];
                            pos += 1;
                        }
                        ct[j] = tok;
                        cp[j] = pos;
                    }
                    var w: u32 = 0; // warm a few batched steps before timing
                    while (w < 3) : (w += 1) {
                        var tks: [MAXB]u32 = undefined;
                        var pss: [MAXB]u32 = undefined;
                        var sls: [MAXB]u32 = undefined;
                        var out: [MAXB]u32 = undefined;
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            tks[j] = ct[j];
                            pss[j] = cp[j];
                            sls[j] = @intCast(j);
                        }
                        try q.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            ct[j] = out[j];
                            cp[j] += 1;
                        }
                    }
                    var timer = try std.time.Timer.start();
                    var s: u32 = 0;
                    while (s < ng) : (s += 1) {
                        var tks: [MAXB]u32 = undefined;
                        var pss: [MAXB]u32 = undefined;
                        var sls: [MAXB]u32 = undefined;
                        var out: [MAXB]u32 = undefined;
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            tks[j] = ct[j];
                            pss[j] = cp[j];
                            sls[j] = @intCast(j);
                        }
                        try q.decodeBatch(tks[0..nseq], pss[0..nseq], sls[0..nseq], out[0..nseq]);
                        j = 0;
                        while (j < nseq) : (j += 1) {
                            ct[j] = out[j];
                            cp[j] += 1;
                        }
                    }
                    const ns = timer.read();
                    const tot = @as(f64, @floatFromInt(ng * nseq));
                    const tps = tot * 1e9 / @as(f64, @floatFromInt(ns));
                    std.debug.print("COLLAPSE_TIMING collapse={s} B={d}: {d:.2} tok/s agg ({d} steps)\n", .{ if (which == 1) "ON " else "OFF", nseq, tps, ng });
                }
                q.moe_collapse_force = null;
                q.moe_shared_batched_force = null;
                q.decode_mrow_force = null;
            }
        },
    }
}

/// Effort 28 increment 2 — continuous-batching SCHEDULER proof (gemma DENSE).
///
/// Drives `Scheduler` (src/scheduler/scheduler.zig) as a real running batch:
/// sequences ARRIVE at staggered ticks, are admitted into a small fixed pool of
/// `nslots` KV slots (nslots < nseq FORCES slot reuse), prefilled into their slot,
/// then DECODED TOGETHER each step at their own per-sequence positions; a sequence
/// that hits its token budget is EVICTED and its slot freed for a waiting arrival.
/// So the batch membership, the per-row co-residents, and the slot a sequence
/// lands in all VARY across the run.
///
/// GATE (`SCHED_GATE`): every sequence's emitted stream must be TOKEN-IDENTICAL to
/// its ISOLATED production run (`serial_out`, via single-sequence prefill+decodeStep).
/// That proves the scheduler introduces no cross-sequence contamination and that a
/// reused slot starts clean — independent of which other sequences share the batch.
/// ADDITIVE: the production decode path + the server mutex are untouched (the server
/// is wired in Increment 3).
fn schedMode(allocator: std.mem.Allocator, seqs_arg: []const u8, ngen: u32, nslots_arg: u32, model_path: []const u8) !void {
    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try Engine.init(allocator, &model, 512);
    defer fwd.deinit();

    if (std.meta.activeTag(fwd) != .gemma) {
        std.debug.print("SCHED:skip (qwen — increment 4)\n", .{});
        return;
    }
    const pf_batched = batchedPrefillDefaultOn();

    const MAXB = 8; // max sequences this harness tracks
    const MAXP = 256; // max prompt tokens / sequence
    const MAXG = 64; // max generated tokens / sequence
    var prompts: [MAXB][MAXP]u32 = undefined;
    var plens: [MAXB]usize = undefined;
    var serial_out: [MAXB][MAXG]u32 = undefined; // isolated production reference
    var serial_len: [MAXB]u32 = undefined; // isolated stream length once EOS-truncated
    var sched_len: [MAXB]u32 = [_]u32{0} ** MAXB; // scheduled stream length at eviction
    var nseq: u32 = 0;
    const ng = @min(ngen, @as(u32, MAXG));

    var seq_it = std.mem.splitScalar(u8, seqs_arg, '|');
    while (seq_it.next()) |seq_str| {
        if (nseq >= MAXB) break;
        const seq_trim = std.mem.trim(u8, seq_str, " ");
        if (seq_trim.len == 0) continue;
        var np: usize = 0;
        var it = std.mem.splitScalar(u8, seq_trim, ',');
        while (it.next()) |s| {
            const t = std.mem.trim(u8, s, " ");
            if (t.len == 0 or np >= MAXP) continue;
            prompts[nseq][np] = try std.fmt.parseInt(u32, t, 10);
            np += 1;
        }
        if (np == 0) continue;
        plens[nseq] = np;

        // ISOLATED REFERENCE: this sequence alone through the production path.
        const prompt = prompts[nseq][0..np];
        var pos: u32 = 0;
        var tok: u32 = 0;
        var used_batched = false;
        if (pf_batched and prompt.len > 1) {
            if (fwd.prefillBatched(prompt)) |firstt| {
                tok = firstt;
                pos = @intCast(prompt.len);
                used_batched = true;
            } else |_| {}
        }
        if (!used_batched) {
            for (prompt) |t| {
                tok = try fwd.decodeStep(t, pos, true);
                pos += 1;
            }
        }
        serial_out[nseq][0] = tok;
        var gi: u32 = 1;
        while (gi < ng) : (gi += 1) {
            const next = try fwd.decodeStep(tok, pos, true);
            pos += 1;
            serial_out[nseq][gi] = next;
            tok = next;
        }
        nseq += 1;
    }
    if (nseq == 0) {
        std.debug.print("SCHED:skip (no sequences)\n", .{});
        return;
    }

    const g = &fwd.gemma;
    if (g.d.n_experts > 0) {
        std.debug.print("SCHED:skip (MoE — increment 1/2 are dense gemma)\n", .{});
        return;
    }

    const nslots = std.math.clamp(nslots_arg, 1, nseq);
    const slot_ctx: u32 = 512;
    try g.allocSlotKv(nslots, slot_ctx);
    defer g.freeSlotKv();

    var sched = try scheduler.Scheduler.init(allocator, nslots);
    defer sched.deinit();

    // Staggered arrivals: sequence j arrives at tick j*STRIDE. Combined with
    // nslots < nseq this yields a ragged batch (mixed positions) + slot reuse.
    const STRIDE: u32 = 2;
    var arrival: [MAXB]u32 = undefined;
    var j: u32 = 0;
    while (j < nseq) : (j += 1) arrival[j] = j * STRIDE;

    // 2b — EOS-driven eviction with VARIABLE per-request gen lengths.
    // Pick an EOS token id and apply it to BOTH the isolated reference and the
    // scheduled run so sequences stop at their OWN (differing) lengths and leave
    // the running batch at different ticks — freeing slots for waiters mid-flight.
    // Default (auto): use the token seq0 emits mid-stream, which makes seq0 (the
    // first arrival) evict early; other seqs stop wherever they hit that token, or
    // run to the `ng` budget. Override with ZINC_SCHED_EOS=<token-id>. maxInt =
    // budget-only (the pre-2b uniform-length behavior).
    var eos: u32 = std.math.maxInt(u32);
    if (std.process.getEnvVarOwned(allocator, "ZINC_SCHED_EOS")) |v| {
        defer allocator.free(v);
        eos = std.fmt.parseInt(u32, std.mem.trim(u8, v, " \n\r\t"), 10) catch std.math.maxInt(u32);
    } else |_| {
        if (ng >= 2) eos = serial_out[0][ng / 2];
    }
    // Truncate each isolated reference at the first EOS occurrence (the stream the
    // model would have produced run alone with this stop token); length = idx+1.
    {
        j = 0;
        while (j < nseq) : (j += 1) {
            var L: u32 = ng;
            var s: u32 = 0;
            while (s < ng) : (s += 1) {
                if (serial_out[j][s] == eos) {
                    L = s + 1;
                    break;
                }
            }
            serial_len[j] = L;
        }
    }

    var sched_out: [MAXB][MAXG]u32 = undefined;
    var completed: u32 = 0;
    const max_ticks = nseq * STRIDE + ng + 16; // safety bound against a stuck loop

    var tick: u32 = 0;
    while (completed < nseq and tick < max_ticks) : (tick += 1) {
        // 1) ARRIVALS for this tick → enqueue (no slot yet).
        j = 0;
        while (j < nseq) : (j += 1) {
            if (arrival[j] == tick) {
                _ = try sched.enqueue(prompts[j][0..plens[j]], .{ .max_tokens = ng });
            }
        }

        // 2) ADMIT waiters into free slots (FIFO) → state .prefilling.
        while ((try sched.admitNext()) != null) {}

        // 3) PREFILL every prefilling slot (per-token B=1 decodeBatch into its slot),
        //    record the first generated token, then promote to .decoding (or complete
        //    immediately if ng==1).
        const to_prefill = sched.pendingPrefill();
        for (to_prefill) |slot_id| {
            const req = &sched.slots[slot_id].?;
            const np = req.prompt_tokens.len;
            var pos: u32 = 0;
            var tok: u32 = 0;
            var k: usize = 0;
            while (k < np) : (k += 1) {
                var tk = [_]u32{req.prompt_tokens[k]};
                var ps = [_]u32{pos};
                var sl = [_]u32{slot_id};
                var ot = [_]u32{0};
                try g.decodeBatch(&tk, &ps, &sl, &ot);
                tok = ot[0];
                pos += 1;
            }
            try req.appendToken(tok);
            try req.transition(.decoding); // .prefilling → .decoding (valid even if it stops below)
            if (req.shouldStop(eos)) {
                try finishSched(&sched, slot_id, &sched_out, &sched_len, ng);
                completed += 1;
            }
        }

        // 4) DECODE one step over the whole running batch (mixed positions/slots).
        const decoders = sched.activeDecoding();
        if (decoders.len > 0) {
            var tks: [MAXB]u32 = undefined;
            var pss: [MAXB]u32 = undefined;
            var sls: [MAXB]u32 = undefined;
            var out: [MAXB]u32 = undefined;
            for (decoders, 0..) |slot_id, i| {
                const req = &sched.slots[slot_id].?;
                const gen_n = req.generated_tokens.items.len;
                tks[i] = req.generated_tokens.items[gen_n - 1];
                // next feed position = prompt_len + (#generated - 1)
                pss[i] = @intCast(req.prompt_tokens.len + gen_n - 1);
                sls[i] = slot_id;
            }
            try g.decodeBatch(tks[0..decoders.len], pss[0..decoders.len], sls[0..decoders.len], out[0..decoders.len]);
            for (decoders, 0..) |slot_id, i| {
                const req = &sched.slots[slot_id].?;
                try req.appendToken(out[i]);
                if (req.shouldStop(eos)) {
                    try finishSched(&sched, slot_id, &sched_out, &sched_len, ng);
                    completed += 1;
                }
            }
        }
    }

    // GATE: every scheduled stream token-identical to its isolated reference,
    // including the SAME EOS-truncated length (variable per request).
    var gate_pass = completed == nseq;
    j = 0;
    while (j < nseq) : (j += 1) {
        const L = serial_len[j];
        var match = sched_len[j] == L;
        var s: u32 = 0;
        while (s < L) : (s += 1) {
            if (sched_out[j][s] != serial_out[j][s]) match = false;
        }
        if (!match) gate_pass = false;
        std.debug.print("SCHED_SEQ{d}(len={d}/{d}):", .{ j, sched_len[j], L });
        s = 0;
        while (s < L) : (s += 1) std.debug.print("{s}{d}", .{ if (s == 0) "" else ",", sched_out[j][s] });
        std.debug.print(" [{s}]\n", .{if (match) "MATCH" else "DIFF"});
    }
    std.debug.print("SCHED_GATE:{s} (nseq={d} nslots={d} ngen={d} eos={d} completed={d} ticks={d})\n", .{ if (gate_pass) "PASS" else "FAIL", nseq, nslots, ng, eos, completed, tick });
}

/// On EOS/budget: copy a finished request's generated stream into `sched_out`
/// (indexed by sequence id-1, the enqueue order), complete it, and free its slot
/// for a waiting arrival.
fn finishSched(sched: *scheduler.Scheduler, slot_id: u32, sched_out: anytype, sched_len: anytype, ng: u32) !void {
    const req = &sched.slots[slot_id].?;
    const seq: usize = @intCast(req.id - 1);
    const items = req.generated_tokens.items;
    sched_len[seq] = @intCast(items.len);
    var s: u32 = 0;
    while (s < ng) : (s += 1) {
        sched_out[seq][s] = if (s < items.len) items[s] else 0;
    }
    try req.transition(.completed);
    sched.release(slot_id);
}

// ── Effort 28 increment 3 (3a): concurrent serving engine ────────────────────
//
// The single-threaded `schedMode` driver proved the continuous-batch loop is
// token-correct. Increment 3 turns it into a *server*: the GPU loop must run on
// its OWN worker thread while many request threads submit work concurrently and
// each receives its own stream. This is the threading model the CUDA HTTP server
// (not yet wired — main.zig:1662) will adopt; 3a proves it WITHOUT the HTTP
// transport so correctness under real thread concurrency is isolated.
//
// Thread-safety model (why this is sound):
//   * ALL GPU work (decodeBatch) runs ONLY on the worker thread. The CUDA shim
//     rebinds the context per call (cuCtxSetCurrent at every entry point), so a
//     single GPU-owning thread needs no extra ceremony.
//   * The ONLY cross-thread mutable state is the scheduler's `pending` FIFO
//     (producers append via enqueue; the worker drains it via admitNext) plus the
//     result/done registry. Both are guarded by ONE mutex. Slot state
//     (prefill/decode/append/release) is worker-only and needs no lock.
//   * Each sequence's tokens depend only on its own slot KV + positions (proven
//     isolated in increment 1), so the nondeterministic admit/interleave ORDER
//     across threads cannot change any sequence's output — exactly what the gate
//     asserts.
const SERVE_MAXB = 8; // max concurrent client threads / sequences
const SERVE_MAXP = 256; // max prompt tokens / sequence
const SERVE_MAXG = 64; // max generated tokens / sequence

/// Shared state between the GPU worker thread and the N producer threads.
const ServeCtx = struct {
    eng: *Engine,
    sched: *scheduler.Scheduler,
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    // Inputs (filled before threads start; prompts must outlive every request —
    // Request borrows the slice, so this storage lives in the parent frame).
    prompts: [SERVE_MAXB][SERVE_MAXP]u32 = undefined,
    plens: [SERVE_MAXB]usize = undefined,
    nseq: u32 = 0,
    ng: u32 = 0,
    eos: u32 = std.math.maxInt(u32),
    // Worker → published stream, keyed by request id-1 (assigned by enqueue).
    pub_out: [SERVE_MAXB][SERVE_MAXG]u32 = undefined,
    pub_len: [SERVE_MAXB]u32 = [_]u32{0} ** SERVE_MAXB,
    pub_done: [SERVE_MAXB]bool = [_]bool{false} ** SERVE_MAXB,
    published: u32 = 0,
    // Client j → received stream + the request id it was assigned (for the gate).
    client_id: [SERVE_MAXB]u64 = [_]u64{0} ** SERVE_MAXB,
    cli_out: [SERVE_MAXB][SERVE_MAXG]u32 = undefined,
    cli_len: [SERVE_MAXB]u32 = [_]u32{0} ** SERVE_MAXB,
};

/// Publish a finished request's stream to its waiter, complete it, free its slot.
/// Worker-thread only; takes the mutex just to flip the done flag + counters.
fn serveFinish(c: *ServeCtx, slot_id: u32) void {
    const req = &c.sched.slots[slot_id].?;
    const items = req.generated_tokens.items;
    const idx: usize = @intCast(req.id - 1);
    const lim = @min(items.len, @as(usize, SERVE_MAXG));
    c.mutex.lock();
    var s: usize = 0;
    while (s < lim) : (s += 1) c.pub_out[idx][s] = items[s];
    c.pub_len[idx] = @intCast(lim);
    c.pub_done[idx] = true;
    c.published += 1;
    c.mutex.unlock();
    c.cond.broadcast(); // wake the client blocked on this request
    req.transition(.completed) catch {};
    c.sched.release(slot_id); // worker-only; frees the slot for a waiter
}

/// The GPU worker: admit waiters, prefill new slots, run ONE batched decode step
/// over all decoders, evict on EOS/budget — until every sequence has finished.
fn serveWorker(c: *ServeCtx) void {
    const eng = c.eng;
    while (true) {
        // Admit pending arrivals into free slots (touches `pending` → under lock).
        c.mutex.lock();
        if (c.published >= c.nseq) {
            c.mutex.unlock();
            break;
        }
        while ((c.sched.admitNext() catch null) != null) {}
        c.mutex.unlock();

        var did_work = false;

        // PREFILL each prefilling slot (per-token B=1 decodeBatch into its slot),
        // record the first generated token, promote to .decoding (slots + GPU are
        // worker-only → no lock). `pendingPrefill` aliases sched.scratch; consume
        // it fully before activeDecoding overwrites it.
        const to_prefill = c.sched.pendingPrefill();
        if (to_prefill.len > 0) did_work = true;
        for (to_prefill) |slot_id| {
            const req = &c.sched.slots[slot_id].?;
            const np = req.prompt_tokens.len;
            var pos: u32 = 0;
            var tok: u32 = 0;
            var k: usize = 0;
            // Clear a reused slot's accumulated SSM state before prefilling the new
            // request from pos=0 (qwen; no-op for gemma) — same as the HTTP engine.
            eng.resetSlot(slot_id) catch return;
            while (k < np) : (k += 1) {
                var tk = [_]u32{req.prompt_tokens[k]};
                var ps = [_]u32{pos};
                var sl = [_]u32{slot_id};
                var ot = [_]u32{0};
                eng.decodeBatch(&tk, &ps, &sl, &ot) catch return;
                tok = ot[0];
                pos += 1;
            }
            req.appendToken(tok) catch return;
            req.transition(.decoding) catch return;
            if (req.shouldStop(c.eos)) serveFinish(c, slot_id);
        }

        // DECODE one step over the whole running batch (mixed positions/slots).
        const decoders = c.sched.activeDecoding();
        if (decoders.len > 0) {
            did_work = true;
            var tks: [SERVE_MAXB]u32 = undefined;
            var pss: [SERVE_MAXB]u32 = undefined;
            var sls: [SERVE_MAXB]u32 = undefined;
            var out: [SERVE_MAXB]u32 = undefined;
            for (decoders, 0..) |slot_id, i| {
                const req = &c.sched.slots[slot_id].?;
                const gen_n = req.generated_tokens.items.len;
                tks[i] = req.generated_tokens.items[gen_n - 1];
                pss[i] = @intCast(req.prompt_tokens.len + gen_n - 1);
                sls[i] = slot_id;
            }
            eng.decodeBatch(tks[0..decoders.len], pss[0..decoders.len], sls[0..decoders.len], out[0..decoders.len]) catch return;
            for (decoders, 0..) |slot_id, i| {
                const req = &c.sched.slots[slot_id].?;
                req.appendToken(out[i]) catch return;
                if (req.shouldStop(c.eos)) serveFinish(c, slot_id);
            }
        }

        // Nothing ready (waiting on a slow producer to enqueue) → yield briefly.
        if (!did_work) std.Thread.sleep(100 * std.time.ns_per_us);
    }
}

/// A producer thread: enqueue one request, block until the worker publishes its
/// stream, then copy it out for the gate. Mirrors what an HTTP handler will do
/// (enqueue + SSE-stream its own tokens) minus the transport.
fn serveClient(c: *ServeCtx, j: u32) void {
    c.mutex.lock();
    const id = c.sched.enqueue(c.prompts[j][0..c.plens[j]], .{ .max_tokens = c.ng }) catch {
        c.mutex.unlock();
        return;
    };
    c.client_id[j] = id;
    const idx: usize = @intCast(id - 1);
    while (!c.pub_done[idx]) c.cond.wait(&c.mutex);
    const lim: usize = c.pub_len[idx];
    var s: usize = 0;
    while (s < lim) : (s += 1) c.cli_out[j][s] = c.pub_out[idx][s];
    c.cli_len[j] = c.pub_len[idx];
    c.mutex.unlock();
}

/// Effort 28 increment 3, sub-step 3a — concurrent serving engine proof.
///
/// Computes an ISOLATED single-sequence reference for each prompt (production
/// decodeStep over the shared cache), then runs ALL sequences concurrently
/// through ONE GPU worker thread fed by N producer threads, and asserts each
/// client's received stream is token-identical to its isolated reference. This
/// proves the server's threading model (one GPU owner, many producers, per-request
/// delivery, thread-safe enqueue + slot reuse) is correct under real concurrency.
/// Additive — production paths untouched; the worker reuses the SAME Scheduler API
/// + decodeBatch the future HTTP server will call.
fn serveMode(allocator: std.mem.Allocator, seqs_arg: []const u8, ngen: u32, nslots_arg: u32, model_path: []const u8) !void {
    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try Engine.init(allocator, &model, 512);
    defer fwd.deinit();

    // Effort 28 increment 4: this harness now drives EITHER gemma4 dense OR the
    // qwen35/36 hybrid-SSM (+MoE) forward — both expose decodeBatch + slot state via
    // the Engine union dispatch, so the threading/slot-reuse proof is arch-uniform.
    const pf_batched = batchedPrefillDefaultOn();

    const ng = @min(ngen, @as(u32, SERVE_MAXG));
    const ctx = try allocator.create(ServeCtx);
    defer allocator.destroy(ctx);
    ctx.* = .{ .eng = undefined, .sched = undefined, .ng = ng };

    var serial_out: [SERVE_MAXB][SERVE_MAXG]u32 = undefined; // isolated reference
    var serial_len: [SERVE_MAXB]u32 = undefined; // EOS-truncated reference length
    var nseq: u32 = 0;

    var seq_it = std.mem.splitScalar(u8, seqs_arg, '|');
    while (seq_it.next()) |seq_str| {
        if (nseq >= SERVE_MAXB) break;
        const seq_trim = std.mem.trim(u8, seq_str, " ");
        if (seq_trim.len == 0) continue;
        var np: usize = 0;
        var it = std.mem.splitScalar(u8, seq_trim, ',');
        while (it.next()) |s| {
            const t = std.mem.trim(u8, s, " ");
            if (t.len == 0 or np >= SERVE_MAXP) continue;
            ctx.prompts[nseq][np] = try std.fmt.parseInt(u32, t, 10);
            np += 1;
        }
        if (np == 0) continue;
        ctx.plens[nseq] = np;

        // ISOLATED REFERENCE: this sequence alone through the production path.
        // Reset the production single-seq recurrent state first so qwen's unindexed
        // SSM state does not leak from the previous reference sequence (no-op gemma).
        try fwd.resetState();
        const prompt = ctx.prompts[nseq][0..np];
        var pos: u32 = 0;
        var tok: u32 = 0;
        var used_batched = false;
        if (pf_batched and prompt.len > 1) {
            if (fwd.prefillBatched(prompt)) |firstt| {
                tok = firstt;
                pos = @intCast(prompt.len);
                used_batched = true;
            } else |_| {}
        }
        if (!used_batched) {
            for (prompt) |t| {
                tok = try fwd.decodeStep(t, pos, true);
                pos += 1;
            }
        }
        serial_out[nseq][0] = tok;
        var gi: u32 = 1;
        while (gi < ng) : (gi += 1) {
            const next = try fwd.decodeStep(tok, pos, true);
            pos += 1;
            serial_out[nseq][gi] = next;
            tok = next;
        }
        nseq += 1;
    }
    if (nseq == 0) {
        std.debug.print("SERVE:skip (no sequences)\n", .{});
        return;
    }

    // EOS for variable per-request lengths (mirrors schedMode): mid-flight eviction
    // exercises the slot-reuse race under concurrency. Env override or auto.
    var eos: u32 = std.math.maxInt(u32);
    if (std.process.getEnvVarOwned(allocator, "ZINC_SCHED_EOS")) |v| {
        defer allocator.free(v);
        eos = std.fmt.parseInt(u32, std.mem.trim(u8, v, " \n\r\t"), 10) catch std.math.maxInt(u32);
    } else |_| {
        if (ng >= 2) eos = serial_out[0][ng / 2];
    }
    ctx.eos = eos;
    {
        var j: u32 = 0;
        while (j < nseq) : (j += 1) {
            var L: u32 = ng;
            var s: u32 = 0;
            while (s < ng) : (s += 1) {
                if (serial_out[j][s] == eos) {
                    L = s + 1;
                    break;
                }
            }
            serial_len[j] = L;
        }
    }

    const nslots = std.math.clamp(nslots_arg, 1, nseq);
    const slot_ctx: u32 = 512;
    try fwd.allocSlots(nslots, slot_ctx);
    defer fwd.freeSlots();

    var sched = try scheduler.Scheduler.init(allocator, nslots);
    defer sched.deinit();

    ctx.eng = &fwd;
    ctx.sched = &sched;
    ctx.nseq = nseq;

    // Spawn the GPU worker, then N producers that all hit the engine concurrently.
    const worker = try std.Thread.spawn(.{}, serveWorker, .{ctx});
    var clients: [SERVE_MAXB]std.Thread = undefined;
    var spawned: u32 = 0;
    while (spawned < nseq) : (spawned += 1) {
        clients[spawned] = try std.Thread.spawn(.{}, serveClient, .{ ctx, spawned });
    }
    var ci: u32 = 0;
    while (ci < nseq) : (ci += 1) clients[ci].join();
    worker.join();

    // GATE: each client's received stream token-identical to its isolated
    // reference, including the SAME EOS-truncated length.
    var gate_pass = ctx.published == nseq;
    var j: u32 = 0;
    while (j < nseq) : (j += 1) {
        const L = serial_len[j];
        var match = ctx.cli_len[j] == L;
        var s: u32 = 0;
        while (s < L) : (s += 1) {
            if (ctx.cli_out[j][s] != serial_out[j][s]) match = false;
        }
        if (!match) gate_pass = false;
        std.debug.print("SERVE_SEQ{d}(id={d} len={d}/{d}):", .{ j, ctx.client_id[j], ctx.cli_len[j], L });
        s = 0;
        while (s < L) : (s += 1) std.debug.print("{s}{d}", .{ if (s == 0) "" else ",", ctx.cli_out[j][s] });
        std.debug.print(" [{s}]\n", .{if (match) "MATCH" else "DIFF"});
    }
    std.debug.print("SERVE_GATE:{s} (nseq={d} nslots={d} ngen={d} eos={d} published={d})\n", .{ if (gate_pass) "PASS" else "FAIL", nseq, nslots, ng, eos, ctx.published });
}

/// Dispatch sync-vs-async microbench: the same kernel launched N times under the
/// current decode pattern (commitAndWait each → CPU blocks per dispatch) vs the
/// async-ring pattern (commitAsync all, one drain → GPU runs back-to-back). The
/// ratio quantifies the async `CUstream`/`CUevent` ring's prize on this WSL2 box:
/// both the removed per-dispatch sync round-trip AND the GPU staying loaded
/// (which holds boost — see the 525 vs 2520 MHz finding). Read-only, no model.
fn benchMode(allocator: std.mem.Allocator, iters: i32, n: u32) !void {
    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    const ctx = dev.ctx;

    const src = try allocator.dupeZ(u8, BENCH_CU);
    defer allocator.free(src);
    var pipe = try pipeline.createPipeline(ctx, src.ptr, "benchk");
    defer pipeline.freePipeline(&pipe);

    const grid = [3]u32{ 2048, 1, 1 };
    const block = [3]u32{ 256, 1, 1 };
    const nthreads: usize = 2048 * 256;
    var buf = try buffer.createBuffer(ctx, nthreads * @sizeOf(f32));
    defer buffer.freeBuffer(&buf);
    const push = BenchPush{ .iters = iters };

    // warmup (also lets the GPU boost before timing)
    var w: u32 = 0;
    while (w < 30) : (w += 1) {
        var cmd = try command.beginCommand(ctx);
        cmd.dispatch(&pipe, grid, block, &.{&buf}, &push, @sizeOf(BenchPush), 0);
        cmd.commitAndWait();
    }

    // SYNC: commitAndWait after every dispatch (the current decodeStep pattern).
    var t = try std.time.Timer.start();
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        var cmd = try command.beginCommand(ctx);
        cmd.dispatch(&pipe, grid, block, &.{&buf}, &push, @sizeOf(BenchPush), 0);
        cmd.commitAndWait();
    }
    const sync_ns = t.read();

    // ASYNC: commitAsync all (pipelined on one stream), then drain in order.
    const cmds = try allocator.alloc(command.CudaCommand, n);
    defer allocator.free(cmds);
    t.reset();
    i = 0;
    while (i < n) : (i += 1) {
        cmds[i] = try command.beginCommand(ctx);
        cmds[i].dispatch(&pipe, grid, block, &.{&buf}, &push, @sizeOf(BenchPush), 0);
        cmds[i].commitAsync();
    }
    i = 0;
    while (i < n) : (i += 1) cmds[i].wait();
    const async_ns = t.read();

    const nf: f64 = @floatFromInt(n);
    const sync_ms = @as(f64, @floatFromInt(sync_ns)) / 1e6;
    const async_ms = @as(f64, @floatFromInt(async_ns)) / 1e6;
    std.debug.print("=== dispatch sync-vs-async bench (N={d}, grid=2048x256, iters={d}) ===\n", .{ n, iters });
    std.debug.print("sync  (commitAndWait each) : {d:>8.2} ms  {d:.4} ms/disp  {d:>8.0} disp/s\n", .{ sync_ms, sync_ms / nf, nf / (sync_ms / 1000.0) });
    std.debug.print("async (commitAsync + drain): {d:>8.2} ms  {d:.4} ms/disp  {d:>8.0} disp/s\n", .{ async_ms, async_ms / nf, nf / (async_ms / 1000.0) });
    std.debug.print("async speedup: {d:.2}x   (per-dispatch saving: {d:.4} ms — the sync round-trip + boost-starvation the ring removes)\n", .{ sync_ms / async_ms, (sync_ms - async_ms) / nf });
}

/// Effort-30 int8-MMA feasibility microbench (see MMA8_CU). Read-only, no model.
fn mma8Mode(allocator: std.mem.Allocator, iters: u32) !void {
    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    const ctx = dev.ctx;

    const src = try allocator.dupeZ(u8, MMA8_CU);
    defer allocator.free(src);

    std.debug.print("=== Effort-30 int8 mma.sync.m16n8k32 feasibility microbench ===\n", .{});

    // (1) COMPILE + CORRECTNESS: does NVRTC/sm_120 accept inline-PTX s8 mma?
    var pu = pipeline.createPipeline(ctx, src.ptr, "mma_unit") catch |e| {
        std.debug.print("Q1 NVRTC-COMPILE inline-PTX m16n8k32.s8: FAILED ({}) => needs nvcc-CUBIN path\n", .{e});
        return;
    };
    defer pipeline.freePipeline(&pu);
    std.debug.print("Q1 NVRTC-COMPILE inline-PTX m16n8k32.s8: OK (no nvcc-CUBIN needed)\n", .{});

    // synthetic A[16x32] row-major, B[32x8] col-major (deterministic patterns).
    var a_host: [16 * 32]i8 = undefined;
    var b_host: [32 * 8]i8 = undefined;
    for (0..16) |m| for (0..32) |k| {
        a_host[m * 32 + k] = @intCast(@as(i32, @intCast((m * 3 + k * 2) % 13)) - 6);
    };
    for (0..32) |k| for (0..8) |n| {
        b_host[k + n * 32] = @intCast(@as(i32, @intCast((k + n * 5) % 11)) - 5);
    };
    var a_buf = try buffer.createBuffer(ctx, a_host.len);
    defer buffer.freeBuffer(&a_buf);
    var b_buf = try buffer.createBuffer(ctx, b_host.len);
    defer buffer.freeBuffer(&b_buf);
    var d_buf = try buffer.createBuffer(ctx, 16 * 8 * @sizeOf(i32));
    defer buffer.freeBuffer(&d_buf);
    buffer.upload(ctx, &a_buf, std.mem.asBytes(&a_host));
    buffer.upload(ctx, &b_buf, std.mem.asBytes(&b_host));
    {
        var cmd = try command.beginCommand(ctx);
        cmd.dispatch(&pu, .{ 1, 1, 1 }, .{ 32, 1, 1 }, &.{ &a_buf, &b_buf, &d_buf }, null, 0, 0);
        cmd.commitAndWait();
    }
    var d_host: [16 * 8]i32 = undefined;
    buffer.download(ctx, &d_buf, std.mem.sliceAsBytes(d_host[0..]));
    var bad: usize = 0;
    for (0..16) |m| for (0..8) |n| {
        var acc: i32 = 0;
        for (0..32) |k| acc += @as(i32, a_host[m * 32 + k]) * @as(i32, b_host[k + n * 32]);
        if (d_host[m * 8 + n] != acc) bad += 1;
    };
    if (bad == 0) {
        std.debug.print("Q1b MMA CORRECTNESS vs scalar ref: PASS (128/128 elems, fragment map correct)\n", .{});
    } else {
        std.debug.print("Q1b MMA CORRECTNESS: FAIL ({d}/128 wrong) => layout/instr bug, ratio below is UNTRUSTWORTHY\n", .{bad});
    }

    // (2) THROUGHPUT: int8 m16n8k32 vs fp16 wmma 16x16x16 (both 4096 MAC/call).
    var p8 = try pipeline.createPipeline(ctx, src.ptr, "tp_int8");
    defer pipeline.freePipeline(&p8);
    var pf = try pipeline.createPipeline(ctx, src.ptr, "tp_f16");
    defer pipeline.freePipeline(&pf);

    const grid = [3]u32{ 1056, 1, 1 }; // 132 SM * 8 blocks
    const block = [3]u32{ 256, 1, 1 }; // 8 warps/block
    var out_buf = try buffer.createBuffer(ctx, grid[0] * @sizeOf(i32));
    defer buffer.freeBuffer(&out_buf);
    const push = TpPush{ .iters = @intCast(iters) };

    const run = struct {
        fn go(c: anytype, pipe: *pipeline.CudaPipeline, g: [3]u32, bl: [3]u32, ob: *buffer.CudaBuffer, p: *const TpPush, warm: bool) !f64 {
            const reps: u32 = if (warm) 3 else 5;
            var r: u32 = 0;
            if (warm) {
                while (r < reps) : (r += 1) {
                    var cm = try command.beginCommand(c);
                    cm.dispatch(pipe, g, bl, &.{ob}, p, @sizeOf(TpPush), 0);
                    cm.commitAndWait();
                }
                return 0;
            }
            var t = try std.time.Timer.start();
            r = 0;
            while (r < reps) : (r += 1) {
                var cm = try command.beginCommand(c);
                cm.dispatch(pipe, g, bl, &.{ob}, p, @sizeOf(TpPush), 0);
                cm.commitAndWait();
            }
            return @as(f64, @floatFromInt(t.read())) / 1e6 / @as(f64, @floatFromInt(reps));
        }
    };
    _ = try run.go(ctx, &p8, grid, block, &out_buf, &push, true);
    _ = try run.go(ctx, &pf, grid, block, &out_buf, &push, true);
    // interleaved to average out boost drift
    var i8_ms: f64 = 0;
    var f16_ms: f64 = 0;
    var rr: u32 = 0;
    while (rr < 4) : (rr += 1) {
        i8_ms += try run.go(ctx, &p8, grid, block, &out_buf, &push, false);
        f16_ms += try run.go(ctx, &pf, grid, block, &out_buf, &push, false);
    }
    i8_ms /= 4;
    f16_ms /= 4;
    const nwarps: f64 = @floatFromInt(grid[0] * (block[0] / 32));
    const macs: f64 = nwarps * @as(f64, @floatFromInt(iters)) * 2.0 * 4096.0;
    const i8_tops = macs / (i8_ms / 1e3) / 1e12;
    const f16_tops = macs / (f16_ms / 1e3) / 1e12;
    std.debug.print("Q2 THROUGHPUT (grid=1056x256, iters={d}, 2 chains):\n", .{iters});
    std.debug.print("   int8 m16n8k32 : {d:>7.2} ms  {d:>7.1} TMAC/s\n", .{ i8_ms, i8_tops });
    std.debug.print("   fp16 wmma16^3 : {d:>7.2} ms  {d:>7.1} TMAC/s\n", .{ f16_ms, f16_tops });
    std.debug.print("   int8/fp16 TC-rate ratio = {d:.2}x  (premise wants ~2.0x; <1.3x end-to-end => int8 lever DEAD)\n", .{i8_tops / f16_tops});
}

/// Effort-30 THE KILL-BAR: full Q4_K-int8 GEMM vs fp16 gemm_q4k_tc, WITH memory,
/// at gemma shapes. Read-only (synthetic weight + activation, no model). Answers
/// the one question mma8 can't: does the 1.9x compute ceiling survive real traffic
/// + the asymmetric epilogue? PASS bar = int8 ≥1.3x vs gemm_q4k_tc ISOLATED.
fn gemm8Mode(allocator: std.mem.Allocator, M: u32, K: u32, T: u32) !void {
    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    const ctx = dev.ctx;
    const src = try allocator.dupeZ(u8, GEMM8_CU);
    defer allocator.free(src);

    std.debug.print("=== Effort-30 Q4_K int8 GEMM kill-bar (M={d} K={d} T={d}) ===\n", .{ M, K, T });
    if (K % 256 != 0 or M % 64 != 0 or T % 64 != 0) {
        std.debug.print("shapes must be K%256==0, M%64==0, T%64==0\n", .{});
        return;
    }

    // synthetic Q4_K weight [M,K]: bpr superblocks/row * 36 u32. Deterministic.
    const bpr = K >> 8;
    const wu32: usize = @as(usize, M) * bpr * 36;
    const w_host = try allocator.alloc(u32, wu32);
    defer allocator.free(w_host);
    // d=0.03, dmin=0.015 as f16 bits; scales/nibbles pseudo-random but valid.
    const d_bits: u16 = @bitCast(@as(f16, 0.03));
    const dmin_bits: u16 = @bitCast(@as(f16, 0.015));
    const d_dmin: u32 = @as(u32, d_bits) | (@as(u32, dmin_bits) << 16);
    var seed: u32 = 0x1234567;
    const rnd = struct {
        fn next(s: *u32) u32 {
            s.* = s.* *% 1664525 +% 1013904223;
            return s.*;
        }
    };
    {
        var i: usize = 0;
        while (i < wu32) : (i += 36) {
            w_host[i] = d_dmin;
            var j: usize = 1;
            while (j < 36) : (j += 1) w_host[i + j] = rnd.next(&seed);
        }
    }
    // synthetic activation A[T,K] f32 in a modest range.
    const a_host = try allocator.alloc(f32, @as(usize, T) * K);
    defer allocator.free(a_host);
    for (a_host, 0..) |*x, i| {
        const r = rnd.next(&seed);
        x.* = (@as(f32, @floatFromInt(r % 2001)) - 1000.0) / 1000.0; // [-1,1]
        _ = i;
    }

    var w_buf = try buffer.createBuffer(ctx, wu32 * @sizeOf(u32));
    defer buffer.freeBuffer(&w_buf);
    var a_buf = try buffer.createBuffer(ctx, a_host.len * @sizeOf(f32));
    defer buffer.freeBuffer(&a_buf);
    var y_fp16 = try buffer.createBuffer(ctx, @as(usize, T) * M * @sizeOf(f32));
    defer buffer.freeBuffer(&y_fp16);
    var y_int8 = try buffer.createBuffer(ctx, @as(usize, T) * M * @sizeOf(f32));
    defer buffer.freeBuffer(&y_int8);
    buffer.upload(ctx, &w_buf, std.mem.sliceAsBytes(w_host));
    buffer.upload(ctx, &a_buf, std.mem.sliceAsBytes(a_host));

    var p_fp16 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_tc");
    defer pipeline.freePipeline(&p_fp16);
    var p_int8 = pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_int8") catch |e| {
        std.debug.print("int8 GEMM COMPILE FAILED ({}) => needs nvcc-CUBIN path after all\n", .{e});
        return;
    };
    defer pipeline.freePipeline(&p_int8);

    const push = Gemm8Push{ .M = M, .K = K, .T = T, .a_offset = 0, .x_offset = 0, .y_offset = 0, .acc_mode = 0, .q8_stride = 0 };
    const grid = [3]u32{ M / 64, T / 64, 1 };
    const block = [3]u32{ 256, 1, 1 };

    const run = struct {
        fn go(c: anytype, pipe: *pipeline.CudaPipeline, g: [3]u32, bl: [3]u32, yb: *buffer.CudaBuffer, wb: *buffer.CudaBuffer, ab: *buffer.CudaBuffer, p: *const Gemm8Push, reps: u32) !f64 {
            var t = try std.time.Timer.start();
            var r: u32 = 0;
            while (r < reps) : (r += 1) {
                var cm = try command.beginCommand(c);
                cm.dispatch(pipe, g, bl, &.{ wb, ab, yb }, p, @sizeOf(Gemm8Push), 0);
                cm.commitAndWait();
            }
            return @as(f64, @floatFromInt(t.read())) / 1e6 / @as(f64, @floatFromInt(reps));
        }
    };
    // warm + correctness pass
    _ = try run.go(ctx, &p_fp16, grid, block, &y_fp16, &w_buf, &a_buf, &push, 2);
    _ = try run.go(ctx, &p_int8, grid, block, &y_int8, &w_buf, &a_buf, &push, 2);

    // CORRECTNESS: relative error int8 vs fp16 gemm_q4k_tc
    const yf = try allocator.alloc(f32, @as(usize, T) * M);
    defer allocator.free(yf);
    const yi = try allocator.alloc(f32, @as(usize, T) * M);
    defer allocator.free(yi);
    buffer.download(ctx, &y_fp16, std.mem.sliceAsBytes(yf));
    buffer.download(ctx, &y_int8, std.mem.sliceAsBytes(yi));
    var max_rel: f64 = 0;
    var sum_abs_err: f64 = 0;
    var sum_abs_ref: f64 = 0;
    var nfin: usize = 0;
    for (yf, 0..) |ref, i| {
        const got = yi[i];
        if (!std.math.isFinite(got) or !std.math.isFinite(ref)) continue;
        nfin += 1;
        const e = @abs(@as(f64, got) - @as(f64, ref));
        sum_abs_err += e;
        sum_abs_ref += @abs(@as(f64, ref));
        const den = @abs(@as(f64, ref)) + 1e-3;
        const rel = e / den;
        if (rel > max_rel) max_rel = rel;
    }
    const mean_rel = if (sum_abs_ref > 0) sum_abs_err / sum_abs_ref else 0;
    std.debug.print("CORRECTNESS int8 vs gemm_q4k_tc: mean_rel(L1)={d:.4}  max_rel={d:.4}  finite={d}/{d}\n", .{ mean_rel, max_rel, nfin, yf.len });
    if (mean_rel > 0.15) {
        std.debug.print("  => mean rel error too high; int8 kernel LIKELY BUGGY, timing UNTRUSTWORTHY\n", .{});
    }

    // TIMING: interleaved ABBA, drop nothing (already warmed), 6 rounds
    var fp16_ms: f64 = 0;
    var int8_ms: f64 = 0;
    var rr: u32 = 0;
    while (rr < 6) : (rr += 1) {
        fp16_ms += try run.go(ctx, &p_fp16, grid, block, &y_fp16, &w_buf, &a_buf, &push, 3);
        int8_ms += try run.go(ctx, &p_int8, grid, block, &y_int8, &w_buf, &a_buf, &push, 3);
    }
    fp16_ms /= 6;
    int8_ms /= 6;
    const speedup = fp16_ms / int8_ms;
    std.debug.print("TIMING (avg of 6 interleaved rounds, 3 reps each):\n", .{});
    std.debug.print("   fp16 gemm_q4k_tc : {d:>7.3} ms\n", .{fp16_ms});
    std.debug.print("   int8 gemm_q4k    : {d:>7.3} ms\n", .{int8_ms});
    std.debug.print("   int8 speedup = {d:.3}x  (KILL-BAR: >=1.30x => WIRE; <1.30x => ABANDON int8)\n", .{speedup});
}

/// Decode-bottleneck profile: splits per-token time into embed+tail vs the
/// 32-layer stack (via the run_layers flag) to size the sync-per-layer overhead
/// that the async CUstream/CUevent ring would remove. Read-only.
fn profMode(allocator: std.mem.Allocator, model_path: []const u8) !void {
    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try forward.ForwardCuda.init(allocator, &model, 512);
    defer fwd.deinit();

    const K: u32 = 40;
    // warmup
    var i: u32 = 0;
    while (i < 5) : (i += 1) _ = try fwd.decodeStep(100, i, true);

    var t = try std.time.Timer.start();
    i = 0;
    while (i < K) : (i += 1) _ = try fwd.decodeStep(100, i, false); // embed + tail only
    const et_ns = t.read();

    t.reset();
    i = 0;
    while (i < K) : (i += 1) _ = try fwd.decodeStep(100, i, true); // full forward
    const full_ns = t.read();

    const et_ms = @as(f64, @floatFromInt(et_ns)) / 1e6 / @as(f64, @floatFromInt(K));
    const full_ms = @as(f64, @floatFromInt(full_ns)) / 1e6 / @as(f64, @floatFromInt(K));
    const layers_ms = full_ms - et_ms;
    // ~65 commitAndWait/token: 32 layers x (mixer + ffn) + 1 tail.
    const commits: f64 = 65;
    std.debug.print("=== decode profile (4090, {d} iters) ===\n", .{K});
    std.debug.print("embed+tail : {d:.3} ms/token\n", .{et_ms});
    std.debug.print("32 layers  : {d:.3} ms/token\n", .{layers_ms});
    std.debug.print("full       : {d:.3} ms/token  ({d:.2} tok/s)\n", .{ full_ms, 1000.0 / full_ms });
    std.debug.print("~per-commit: {d:.3} ms  ({d:.0} sync round-trips/token)\n", .{ full_ms / commits, commits });
    std.debug.print("(async ring batches these into ~1 submit/token — the headroom to the 97 t/s bar)\n", .{});
}

/// Dump the full vocab logit vector for `token` at pos 0 to a raw-f32 file,
/// for numerical-fidelity comparison vs a reference implementation logit dump.
fn logitsMode(allocator: std.mem.Allocator, token: u32, out_path: []const u8, model_path: []const u8) !void {
    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try Engine.init(allocator, &model, 512);
    defer fwd.deinit();

    const vocab = fwd.vocab();
    const buf = try allocator.alloc(f32, vocab);
    defer allocator.free(buf);
    _ = try fwd.decodeStep(token, 0, true);
    fwd.readLogits(buf);

    const f = try std.fs.cwd().createFile(out_path, .{});
    defer f.close();
    try f.writeAll(std.mem.sliceAsBytes(buf));

    var bi: usize = 0;
    var bm = buf[0];
    for (buf, 0..) |v, i| if (v > bm) {
        bm = v;
        bi = i;
    };
    std.debug.print("wrote {d} logits to {s}; argmax={d} ({d:.4})\n", .{ vocab, out_path, bi, bm });
}

/// Teacher-forced next-token agreement vs a reference continuation. Feeds the
/// TRUE tokens (prompt ++ gen) and, at each generated position, checks whether
/// ZINC's argmax equals the reference's actual next token — so a single near-tie
/// flip costs one match instead of permanently desyncing the free-running greedy
/// compare. This is the standard token-correctness metric and is robust to the
/// q8_1-activation near-ties that separate ZINC's (correct) f32 forward from the
/// llama-CUDA reference. Prints "TF_MATCH:k/N".
fn teacherForcedMode(allocator: std.mem.Allocator, prompt_arg: []const u8, gen_arg: []const u8, model_path: []const u8) !void {
    var buf: [512]u32 = undefined;
    var np: usize = 0;
    inline for (.{ prompt_arg, gen_arg }) |arg| {
        var it = std.mem.splitScalar(u8, arg, ',');
        while (it.next()) |s| {
            const trimmed = std.mem.trim(u8, s, " ");
            if (trimmed.len == 0 or np >= buf.len) continue;
            buf[np] = std.fmt.parseInt(u32, trimmed, 10) catch continue;
            np += 1;
        }
    }
    // Count prompt tokens to know where the generated region begins.
    var plen: usize = 0;
    {
        var it = std.mem.splitScalar(u8, prompt_arg, ',');
        while (it.next()) |s| {
            if (std.mem.trim(u8, s, " ").len != 0) plen += 1;
        }
    }
    const seq = buf[0..np];
    if (np == 0 or plen == 0 or plen >= np) return error.BadSequence;

    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try Engine.init(allocator, &model, 512);
    defer fwd.deinit();

    // Teacher-forced: feed the TRUE token at every position; the argmax after
    // feeding seq[i] is the prediction for seq[i+1]. Score the gen region.
    var pos: u32 = 0;
    var match: u32 = 0;
    var total: u32 = 0;
    var pred: u32 = 0;
    while (pos < np) : (pos += 1) {
        pred = try fwd.decodeStep(seq[pos], pos, true);
        const next = pos + 1;
        if (next < np and next >= plen) {
            total += 1;
            if (pred == seq[next]) match += 1;
        }
    }
    std.debug.print("TF_MATCH:{d}/{d}\n", .{ match, total });
}

/// Prefill a prompt id list, then dump the full vocab logit vector predicting
/// the NEXT token (i.e. logits after the last prompt token) to a raw-f32 file.
/// Lets a pos>0 logit-fidelity comparison vs the reference implementation pinpoint whether a greedy
/// divergence is a real bug or a near-tie fp flip.
fn gemmaLogitsMode(allocator: std.mem.Allocator, ids_arg: []const u8, out_path: []const u8, model_path: []const u8) !void {
    var prompt_buf: [256]u32 = undefined;
    var np: usize = 0;
    var it = std.mem.splitScalar(u8, ids_arg, ',');
    while (it.next()) |s| {
        const trimmed = std.mem.trim(u8, s, " ");
        if (trimmed.len == 0 or np >= prompt_buf.len) continue;
        prompt_buf[np] = try std.fmt.parseInt(u32, trimmed, 10);
        np += 1;
    }
    const prompt = prompt_buf[0..np];
    if (np == 0) return error.EmptyPrompt;

    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try Engine.init(allocator, &model, 512);
    defer fwd.deinit();

    var pos: u32 = 0;
    var tok: u32 = 0;
    for (prompt) |t| {
        tok = try fwd.decodeStep(t, pos, true);
        pos += 1;
    }
    const vocab = fwd.vocab();
    const buf = try allocator.alloc(f32, vocab);
    defer allocator.free(buf);
    fwd.readLogits(buf);

    const f = try std.fs.cwd().createFile(out_path, .{});
    defer f.close();
    try f.writeAll(std.mem.sliceAsBytes(buf));
    std.debug.print("prefilled {d} tokens; argmax next = {d}; wrote {d} logits to {s}\n", .{ np, tok, vocab, out_path });
}

/// Per-layer residual-norm dump at pos 0 (single token).
fn dumpMode(allocator: std.mem.Allocator, token: u32, model_path: []const u8) !void {
    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    if (model.config.architecture == .gemma) return gemmaDumpMode(allocator, &model, token);
    var fwd = try forward.ForwardCuda.init(allocator, &model, 512);
    defer fwd.deinit();

    const n = fwd.d.n_embd;
    const interval = fwd.d.full_attn_interval;
    const buf = try allocator.alloc(f32, @max(n, fwd.d.vocab));
    defer allocator.free(buf);

    std.debug.print("=== CUDA per-layer dump, token {d}, pos 0, {d} layers (interval {d}) ===\n", .{ token, fwd.d.n_layers, interval });

    _ = try fwd.decodeStep(token, 0, false);
    fwd.readHidden(buf[0..n]);
    stats("embed", buf[0..n]);

    var L: u32 = 0;
    while (L < fwd.d.n_layers) : (L += 1) {
        const is_attn = ((L + 1) % interval) == 0;
        if (is_attn) try fwd.attentionLayerPub(L, 0) else try fwd.ssmLayerPub(L);
        fwd.readHidden(buf[0..n]);
        var lbl: [24]u8 = undefined;
        stats(try std.fmt.bufPrint(&lbl, "L{d:0>2}-{s}", .{ L, if (is_attn) "att" else "ssm" }), buf[0..n]);
        try fwd.ffnBlockPub(L);
        fwd.readHidden(buf[0..n]);
        stats(try std.fmt.bufPrint(&lbl, "L{d:0>2}-ffn", .{L}), buf[0..n]);
    }
}

/// gemma4 per-layer residual-norm dump at pos 0 (single token). Dumps the
/// residual stream after attention, after FFN, and after the per-layer output
/// scale (== the reference implementation `l_out-N`), so the post-outscale norm can be diffed
/// against a gemma4 eval-callback reference to find the first divergent layer.
fn gemmaDumpMode(allocator: std.mem.Allocator, model: *loader.Model, token: u32) !void {
    var fwd = try forwardgemma.ForwardGemma.init(allocator, model, 512);
    defer fwd.deinit();

    const n = fwd.d.n_embd;
    const buf = try allocator.alloc(f32, @max(n, fwd.d.vocab));
    defer allocator.free(buf);

    std.debug.print("=== CUDA gemma4 per-layer dump, token {d}, pos 0, {d} layers ===\n", .{ token, fwd.d.n_layers });

    _ = try fwd.decodeStep(token, 0, false);
    fwd.readHidden(buf[0..n]);
    stats("embed", buf[0..n]);

    var L: u32 = 0;
    while (L < fwd.d.n_layers) : (L += 1) {
        var lbl: [24]u8 = undefined;
        const tag: []const u8 = if (fwd.geom[L].is_swa) "swa" else "FUL";
        try fwd.attentionLayerPub(L, 0);
        fwd.readHidden(buf[0..n]);
        stats(try std.fmt.bufPrint(&lbl, "L{d:0>2}-att-{s}", .{ L, tag }), buf[0..n]);
        try fwd.ffnLayerPub(L);
        fwd.readHidden(buf[0..n]);
        stats(try std.fmt.bufPrint(&lbl, "L{d:0>2}-ffn", .{L}), buf[0..n]);
        try fwd.layerOutScalePub(L);
        fwd.readHidden(buf[0..n]);
        stats(try std.fmt.bufPrint(&lbl, "L{d:0>2}-out", .{L}), buf[0..n]);
    }
}

/// gemma4 per-layer residual-VECTOR dump at the LAST position of a prompt id
/// list. Prefills ids[0..n-1] (full forward, populating KV at pos 0..n-2), then
/// at the final position steps through layers, writing the post-output-scale
/// residual (== the reference implementation `l_out-N`) of every layer to a flat f32 binary
/// [n_layers * n_embd]. Pairs with a reference implementation eval-callback dumping l_out's last
/// column; cosine/maxdiff per layer pinpoints the first POSITION-dependent
/// (rope/KV) divergence that the pos-0 norm dump cannot see.
fn gemmaLayerDumpMode(allocator: std.mem.Allocator, ids_arg: []const u8, out_path: []const u8, model_path: []const u8) !void {
    var prompt_buf: [256]u32 = undefined;
    var np: usize = 0;
    var it = std.mem.splitScalar(u8, ids_arg, ',');
    while (it.next()) |s| {
        const trimmed = std.mem.trim(u8, s, " ");
        if (trimmed.len == 0 or np >= prompt_buf.len) continue;
        prompt_buf[np] = try std.fmt.parseInt(u32, trimmed, 10);
        np += 1;
    }
    const prompt = prompt_buf[0..np];
    if (np == 0) return error.EmptyPrompt;

    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();
    var fwd = try forwardgemma.ForwardGemma.init(allocator, &model, 512);
    defer fwd.deinit();

    const n = fwd.d.n_embd;
    const nl = fwd.d.n_layers;
    const buf = try allocator.alloc(f32, @max(n, fwd.d.vocab));
    defer allocator.free(buf);
    const all = try allocator.alloc(f32, nl * n);
    defer allocator.free(all);

    // Prefill every prompt token except the last (full forward, KV carry).
    var pos: u32 = 0;
    while (pos + 1 < np) : (pos += 1) _ = try fwd.decodeStep(prompt[pos], pos, true);
    const last_pos: u32 = @intCast(np - 1);

    std.debug.print("=== gemma4 layer-vector dump @ pos {d}, {d} layers, prompt {s} ===\n", .{ last_pos, nl, ids_arg });
    _ = try fwd.decodeStep(prompt[last_pos], last_pos, false); // embed only
    var L: u32 = 0;
    while (L < nl) : (L += 1) {
        try fwd.attentionLayerPub(L, last_pos);
        try fwd.ffnLayerPub(L);
        try fwd.layerOutScalePub(L);
        fwd.readHidden(all[L * n ..][0..n]);
        var lbl: [24]u8 = undefined;
        const tag: []const u8 = if (fwd.geom[L].is_swa) "swa" else "FUL";
        stats(try std.fmt.bufPrint(&lbl, "L{d:0>2}-{s}", .{ L, tag }), all[L * n ..][0..n]);
    }
    const f = try std.fs.cwd().createFile(out_path, .{});
    defer f.close();
    try f.writeAll(std.mem.sliceAsBytes(all));
    std.debug.print("wrote {d} layer vectors ({d} floats) to {s}\n", .{ nl, nl * n, out_path });
}
