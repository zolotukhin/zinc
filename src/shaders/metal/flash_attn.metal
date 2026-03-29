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

struct _139
{
    uint _m0[1];
};

struct _176
{
    float _m0[1];
};

struct _186
{
    float _m0[1];
};

struct _348
{
    float _m0[1];
};

struct _402
{
    float _m0[1];
};

kernel void main0(constant _22& _24 [[buffer(0)]], device _139& _141 [[buffer(1)]], device _176& _178 [[buffer(2)]], device _186& _188 [[buffer(3)]], device _348& _350 [[buffer(4)]], device _402& _404 [[buffer(5)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]])
{
    threadgroup spvUnsafeArray<float, 256> _57;
    threadgroup float _69;
    threadgroup float _71;
    threadgroup spvUnsafeArray<float, 256> _199;
    uint _34 = gl_WorkGroupID.x / (_24._m1 / _24._m2);
    uint _40 = gl_WorkGroupID.x * _24._m0;
    for (uint _417 = gl_LocalInvocationID.x; _417 < _24._m0; )
    {
        _57[_417] = 0.0;
        _417 += 64u;
        continue;
    }
    bool _66 = gl_LocalInvocationID.x == 0u;
    if (_66)
    {
        _69 = -1000000015047466219876688855040.0;
        _71 = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _79 = rsqrt(float(_24._m0));
    uint _418;
    _418 = 0u;
    uint _105;
    for (; _418 < _24._m3; _418 = _105)
    {
        _105 = _418 + 256u;
        uint _112 = min(_105, _24._m3) - _418;
        for (uint _420 = gl_LocalInvocationID.x; _420 < _112; _420 += 64u)
        {
            uint _126 = _418 + _420;
            uint _163 = _24._m0 * ((((_141._m0[_126 / _24._m4] * _24._m4) + (_126 % _24._m4)) * _24._m2) + _34);
            float _439;
            _439 = 0.0;
            for (uint _437 = 0u; _437 < _24._m0; )
            {
                _439 += (_178._m0[_40 + _437] * _188._m0[_163 + _437]);
                _437++;
                continue;
            }
            _199[_420] = _439 * _79;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _422;
        _422 = -1000000015047466219876688855040.0;
        for (uint _421 = gl_LocalInvocationID.x; _421 < _112; )
        {
            _422 = fast::max(_422, _199[_421]);
            _421 += 64u;
            continue;
        }
        float _228 = simd_max(_422);
        float _230 = _69;
        float _232 = fast::max(_230, _228);
        float _424;
        _424 = 0.0;
        for (uint _423 = gl_LocalInvocationID.x; _423 < _112; )
        {
            float _247 = _199[_423];
            float _250 = exp(_247 - _232);
            _199[_423] = _250;
            _424 += _250;
            _423 += 64u;
            continue;
        }
        float _261 = simd_sum(_424);
        float _263 = _69;
        float _266 = exp(_263 - _232);
        for (uint _425 = gl_LocalInvocationID.x; _425 < _24._m0; )
        {
            _57[_425] *= _266;
            _425 += 64u;
            continue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _426;
        _426 = gl_LocalInvocationID.x;
        for (; _426 < _24._m0; _426 += 64u)
        {
            float _435;
            _435 = 0.0;
            for (uint _433 = 0u; _433 < _112; )
            {
                uint _311 = _418 + _433;
                _435 += (_199[_433] * _350._m0[(_24._m0 * ((((_141._m0[_311 / _24._m4] * _24._m4) + (_311 % _24._m4)) * _24._m2) + _34)) + _426]);
                _433++;
                continue;
            }
            _57[_426] += _435;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (_66)
        {
            _71 = (_71 * _266) + _261;
            _69 = _232;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float _384 = 1.0 / _71;
    for (uint _419 = gl_LocalInvocationID.x; _419 < _24._m0; )
    {
        _404._m0[_40 + _419] = _57[_419] * _384;
        _419 += 64u;
        continue;
    }
}

