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

struct _22
{
    uint _m0;
    uint _m1;
    uint _m2;
    uint _m3;
    uint _m4;
};

struct _49
{
    float _m0[1];
};

struct _166
{
    float _m0[1];
};

struct _176
{
    float _m0[1];
};

struct _193
{
    float _m0[1];
};

kernel void main0(constant _22& _24 [[buffer(0)]], device _49& _51 [[buffer(1)]], device _166& _168 [[buffer(2)]], device _176& _178 [[buffer(3)]], device _193& _195 [[buffer(4)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint gl_SubgroupSize [[thread_execution_width]])
{
    threadgroup spvUnsafeArray<float, 64> _77;
    uint _30 = gl_WorkGroupID.x * _24._m2;
    float _207;
    _207 = 0.0;
    for (uint _206 = gl_LocalInvocationID.x; _206 < _24._m2; )
    {
        uint _55 = _30 + _206;
        _207 += (_51._m0[_55] * _51._m0[_55]);
        _206 += 64u;
        continue;
    }
    float _69 = simd_sum(_207);
    float _210;
    if (gl_SubgroupSize < 64u)
    {
        _77[gl_LocalInvocationID.x] = _69;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (gl_LocalInvocationID.x == 0u)
        {
            uint _95 = (gl_SubgroupSize + 63u) / gl_SubgroupSize;
            float _209;
            _209 = 0.0;
            for (uint _208 = 0u; _208 < _95; )
            {
                _209 += _77[_208 * gl_SubgroupSize];
                _208++;
                continue;
            }
            _77[0] = _209;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _210 = _77[0];
    }
    else
    {
        _210 = _69;
    }
    float _127 = rsqrt((_210 / float(_24._m2)) + 9.9999999747524270787835121154785e-07);
    for (uint _211 = gl_LocalInvocationID.x; _211 < _24._m2; _211 += 64u)
    {
        uint _142 = _30 + _211;
        uint _212;
        if (_24._m4 != 0u)
        {
            _212 = _142;
        }
        else
        {
            _212 = _211 % _24._m3;
        }
        _195._m0[_142] = ((_51._m0[_142] * _127) * _168._m0[_212]) * (_178._m0[_142] / (1.0 + exp(-_178._m0[_142])));
    }
}

