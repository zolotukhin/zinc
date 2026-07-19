.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.text
.globl zinc_rt_argmax_f32_range
.type zinc_rt_argmax_f32_range,@function

// ABI:
//   s[0:1] = f32 score input pointer
//   s[2:3] = output pointer: u32 selected_token, f32 selected_score
//   s4     = rows
//   s5     = start_row
//
// One workitem scans a compact score range and stores the absolute selected row.
// Used after a resident DMMV grid in the same IB so full-LM-head validation can
// read back one token/score pair instead of the whole logits row range.
zinc_rt_argmax_f32_range:
    v_mov_b32_e32 v0, 0
    v_mov_b32_e32 v3, 0
    s_mov_b32 s9, 1
    global_load_b32 v1, v0, s[0:1]
    s_waitcnt vmcnt(0)

loop:
    s_cmp_ge_u32 s9, s4
    s_cbranch_scc1 done

    s_lshl_b32 s10, s9, 2
    v_mov_b32_e32 v0, s10
    global_load_b32 v2, v0, s[0:1]
    s_waitcnt vmcnt(0)

    v_mov_b32_e32 v4, s9
    v_cmp_gt_f32_e32 v2, v1
    v_cndmask_b32_e32 v1, v1, v2
    v_cndmask_b32_e32 v3, v3, v4

    s_add_u32 s9, s9, 1
    s_branch loop

done:
    v_add_u32_e32 v3, s5, v3
    v_mov_b32_e32 v0, 0
    global_store_b32 v0, v3, s[2:3]
    v_mov_b32_e32 v0, 4
    global_store_b32 v0, v1, s[2:3]

    s_nop 0
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
