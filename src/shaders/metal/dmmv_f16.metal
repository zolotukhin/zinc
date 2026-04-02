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

struct _29
{
    uint _m0;
    uint _m1;
    uint _m2;
    uint _m3;
    uint _m4;
};

struct _64
{
    float _m0[1];
};

struct _83
{
    half _m0[1];
};

struct _185
{
    float _m0[1];
};

kernel void main0(constant _29& _31 [[buffer(0)]], device _64& _66 [[buffer(1)]], device _83& _85 [[buffer(2)]], device _185& _187 [[buffer(3)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint gl_NumSubgroups [[simdgroups_per_threadgroup]], uint gl_SubgroupID [[simdgroup_index_in_threadgroup]])
{
    threadgroup spvUnsafeArray<float, 2> _144;
    threadgroup spvUnsafeArray<float, 2> _150;
    uint _19 = gl_WorkGroupID.x * 2u;
    uint _23 = _19 + 1u;
    uint _37 = _31._m2 / 2u;
    uint _43 = _31._m3 / 4u;
    float _208;
    float _209;
    _209 = 0.0;
    _208 = 0.0;
    float _221;
    float _222;
    for (uint _207 = gl_LocalInvocationID.x; _207 < _31._m1; _209 = _222, _208 = _221, _207 += 64u)
    {
        uint _70 = _43 + _207;
        if (_19 < _31._m0)
        {
            _221 = _208 + (float(_85._m0[(_37 + (_19 * _31._m1)) + _207]) * _66._m0[_70]);
        }
        else
        {
            _221 = _208;
        }
        if (_23 < _31._m0)
        {
            _222 = _209 + (float(_85._m0[(_37 + (_23 * _31._m1)) + _207]) * _66._m0[_70]);
        }
        else
        {
            _222 = _209;
        }
    }
    float _131 = simd_sum(_208);
    float _133 = simd_sum(_209);
    float _210;
    float _214;
    if (gl_NumSubgroups > 1u)
    {
        bool _139 = simd_is_first();
        if (_139)
        {
            _144[gl_SubgroupID] = _131;
            _150[gl_SubgroupID] = _133;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _211;
        float _215;
        if (gl_LocalInvocationID.x == 0u)
        {
            _215 = _150[0] + _150[1];
            _211 = _144[0] + _144[1];
        }
        else
        {
            _215 = _133;
            _211 = _131;
        }
        _214 = _215;
        _210 = _211;
    }
    else
    {
        _214 = _133;
        _210 = _131;
    }
    if (gl_LocalInvocationID.x == 0u)
    {
        uint _177 = _31._m4 / 4u;
        if (_19 < _31._m0)
        {
            _187._m0[_177 + _19] = _210;
        }
        if (_23 < _31._m0)
        {
            _187._m0[_177 + _23] = _214;
        }
    }
}

