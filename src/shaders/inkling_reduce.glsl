// Each invocation receives the same reduction. The cross-subgroup path also
// works for repeated reductions in the same workgroup.
shared float subgroup_totals[64];

float reduce_sum(float v) {
    v = subgroupAdd(v);
    if (gl_NumSubgroups == 1u) return v;
    if (subgroupElect()) subgroup_totals[gl_SubgroupID] = v;
    barrier();
    v = 0.0;
    for (uint i = 0u; i < gl_NumSubgroups; ++i) v += subgroup_totals[i];
    barrier();
    return v;
}

float reduce_max(float v) {
    v = subgroupMax(v);
    if (gl_NumSubgroups == 1u) return v;
    if (subgroupElect()) subgroup_totals[gl_SubgroupID] = v;
    barrier();
    v = subgroup_totals[0];
    for (uint i = 1u; i < gl_NumSubgroups; ++i) v = max(v, subgroup_totals[i]);
    barrier();
    return v;
}
