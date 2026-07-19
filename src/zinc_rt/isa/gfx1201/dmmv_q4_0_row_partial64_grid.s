.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.text
.globl zinc_rt_dmmv_q4_0_row_partial64_grid
.type zinc_rt_dmmv_q4_0_row_partial64_grid,@function

// Grid-over-rows Q4_0 partial-sum DMMV. One wave64 workgroup computes one
// output row; lanes 0..31 split the K dimension and write 64 partial slots
// (lanes 32..63 write zero). The host reduces the fixed 64-float partial row.
//
// ABI:
//   s[0:1] = input f32 vector pointer (length cols)
//   s[2:3] = output f32 partials pointer (rows * 64 floats)
//   s[4:5] = Q4_0 weight base pointer (row-major; may be VRAM-resident)
//   s6     = cols (multiple of 32)
//   s7     = total_rows
//   ttmp9  = workgroup_id_x (row id on gfx11/gfx12)
//   v0     = workitem_id_x (lane 0..63)
zinc_rt_dmmv_q4_0_row_partial64_grid:
    s_mov_b32 s8, ttmp9                 // s8 = global row
    s_cmp_ge_u32 s8, s7
    s_cbranch_scc1 done

    v_mov_b32_e32 v8, v0                // v8 = lane id
    v_mov_b32_e32 v1, 0                 // v1 = partial accumulator
    v_and_b32_e32 v21, 31, v8           // lane modulo 32, input column within block
    v_and_b32_e32 v22, 15, v8           // lane modulo 16, packed nibble byte
    s_lshr_b32 s10, s6, 5               // s10 = num_blocks = cols / 32
    s_mul_i32 s12, s8, s10              // row * num_blocks
    s_mul_i32 s12, s12, 18              // row byte base
    s_mov_b32 s11, 0                    // block index

block_loop:
    s_cmp_ge_u32 s11, s10
    s_cbranch_scc1 store_partial

    s_mul_i32 s13, s11, 18
    s_add_u32 s13, s13, s12             // absolute block byte offset
    v_mov_b32_e32 v10, s13
    global_load_ushort v2, v10, s[4:5]  // f16 block scale

    s_mul_i32 s14, s11, 32
    s_lshl_b32 s17, s14, 2
    v_lshlrev_b32_e32 v12, 2, v21
    v_add_nc_u32_e32 v12, s17, v12
    global_load_b32 v6, v12, s[0:1]     // input[block*32 + lane%32]

    s_add_u32 s16, s13, 2
    v_add_nc_u32_e32 v11, s16, v22
    global_load_ubyte v3, v11, s[4:5]   // packed nibble byte

    s_waitcnt vmcnt(0)
    v_cvt_f32_f16_e32 v2, v2
    v_and_b32_e32 v4, 0x0f, v3
    v_lshrrev_b32_e32 v5, 4, v3
    v_mov_b32_e32 v30, 16
    v_cmp_lt_u32_e32 v21, v30
    v_cndmask_b32_e32 v4, v5, v4
    v_add_nc_u32_e32 v4, -8, v4
    v_cvt_f32_i32_e32 v4, v4
    v_mul_f32_e32 v4, v2, v4
    v_fmac_f32_e32 v1, v4, v6

    s_add_u32 s11, s11, 1
    s_branch block_loop

store_partial:
    v_mov_b32_e32 v30, 32
    v_cmp_lt_u32_e32 v8, v30
    v_cndmask_b32_e32 v1, 0, v1
    s_lshl_b32 s19, s8, 8               // row * 64 partials * sizeof(f32)
    v_lshlrev_b32_e32 v14, 2, v8
    v_add_nc_u32_e32 v14, s19, v14
    global_store_b32 v14, v1, s[2:3]

done:
    s_nop 0
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
