#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct _23
{
    uint _m0;
    float _m1;
};

struct _49
{
    float _m0[1];
};

struct _141
{
    float _m0[1];
};

struct _148
{
    float _m0[1];
};

kernel void main0(constant _23& _25 [[buffer(0)]], device _49& _51 [[buffer(1)]], device _141& _143 [[buffer(2)]], device _148& _150 [[buffer(3)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupSize [[thread_execution_width]])
{
    threadgroup spvUnsafeArray<float, 64> _76;
    uint _31 = gl_WorkGroupID.x * _25._m0;
    float _167;
    _167 = 0.0;
    for (uint _166 = gl_LocalInvocationID.x; _166 < _25._m0; )
    {
        uint _54 = _31 + _166;
        _167 += (_51._m0[_54] * _51._m0[_54]);
        _166 += 64u;
        continue;
    }
    float _68 = simd_sum(_167);
    float _170;
    if (gl_SubgroupSize < 64u)
    {
        _76[gl_LocalInvocationID.x] = _68;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (gl_LocalInvocationID.x == 0u)
        {
            uint _94 = (gl_SubgroupSize + 63u) / gl_SubgroupSize;
            float _169;
            _169 = 0.0;
            for (uint _168 = 0u; _168 < _94; )
            {
                _169 += _76[_168 * gl_SubgroupSize];
                _168++;
                continue;
            }
            _76[0] = _169;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _170 = _76[0];
    }
    else
    {
        _170 = _68;
    }
    float _128 = rsqrt((_170 / float(_25._m0)) + _25._m1);
    for (uint _171 = gl_LocalInvocationID.x; _171 < _25._m0; )
    {
        uint _146 = _31 + _171;
        _143._m0[_146] = _150._m0[_171] * (_51._m0[_146] * _128);
        _171 += 64u;
        continue;
    }
}

