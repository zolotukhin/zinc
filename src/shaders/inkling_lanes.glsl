// The subgroup invocation order can differ from the workgroup invocation
// order. SPIR-V 1.6 guarantees full subgroups when their width divides this
// one-dimensional workgroup, so the shuffle paths can index by subgroup.
bool can_shuffle(uint width) {
    return gl_SubgroupSize >= width && gl_SubgroupSize <= gl_WorkGroupSize.x;
}

uint shuffle_lane() {
    return gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
}
