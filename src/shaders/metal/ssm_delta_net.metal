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

struct _21
{
    uint _m0;
    uint _m1;
    uint _m2;
    uint _m3;
    uint _m4;
    uint _m5;
    uint _m6;
    uint _m7;
    uint _m8;
};

struct _105
{
    float _m0[1];
};

struct _327
{
    float _m0[1];
};

struct _347
{
    half _m0[1];
};

struct _359
{
    float _m0[1];
};

struct _387
{
    half _m0[1];
};

struct _396
{
    float _m0[1];
};

struct _414
{
    float _m0[1];
};

struct _471
{
    float _m0[1];
};

struct _562
{
    float _m0[1];
};

kernel void main0(constant _21& _23 [[buffer(0)]], device _105& _107 [[buffer(1)]], device _327& _329 [[buffer(2)]], device void* spvBufferAliasSet0Binding1 [[buffer(3)]], device void* spvBufferAliasSet0Binding4 [[buffer(4)]], device _414& _416 [[buffer(5)]], device _471& _473 [[buffer(6)]], device _562& _564 [[buffer(7)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint gl_SubgroupSize [[thread_execution_width]])
{
    device auto& _349 = *(device _347*)spvBufferAliasSet0Binding1;
    device auto& _389 = *(device _387*)spvBufferAliasSet0Binding4;
    device auto& _361 = *(device _359*)spvBufferAliasSet0Binding1;
    device auto& _398 = *(device _396*)spvBufferAliasSet0Binding4;
    threadgroup spvUnsafeArray<float, 128> _102;
    threadgroup spvUnsafeArray<float, 128> _157;
    threadgroup spvUnsafeArray<float, 128> _196;
    threadgroup float _409;
    threadgroup float _412;
    threadgroup float _sg_reduce[4];
    do
    {
        if (gl_WorkGroupID.x >= _23._m1)
        {
            break;
        }
        uint _41 = _23._m3 * _23._m4;
        uint _583;
        if (_23._m4 == _23._m1)
        {
            _583 = gl_WorkGroupID.x;
        }
        else
        {
            _583 = gl_WorkGroupID.x % _23._m4;
        }
        uint _62 = _583 * _23._m3;
        uint _69 = _41 + _62;
        uint _78 = gl_WorkGroupID.x * _23._m2;
        uint _79 = (2u * _41) + _78;
        uint _584 = gl_LocalInvocationID.x;
        for (;;)
        {
            bool _90 = _584 < _23._m3;
            bool _97;
            if (_90)
            {
                _97 = _584 < _23._m2;
            }
            else
            {
                _97 = _90;
            }
            if (_97)
            {
                _102[_584] = _107._m0[_62 + _584];
                _584 += 64u;
                continue;
            }
            else
            {
                break;
            }
        }
        uint _124 = gl_LocalInvocationID.x + _23._m3;
        for (uint _585 = _124; _585 < _23._m2; )
        {
            _102[_585] = 0.0;
            _585 += 64u;
            continue;
        }
        uint _586 = gl_LocalInvocationID.x;
        for (;;)
        {
            bool _149 = _586 < _23._m3;
            bool _156;
            if (_149)
            {
                _156 = _586 < _23._m2;
            }
            else
            {
                _156 = _149;
            }
            if (_156)
            {
                _157[_586] = _107._m0[_69 + _586];
                _586 += 64u;
                continue;
            }
            else
            {
                break;
            }
        }
        for (uint _587 = _124; _587 < _23._m2; )
        {
            _157[_587] = 0.0;
            _587 += 64u;
            continue;
        }
        for (uint _588 = gl_LocalInvocationID.x; _588 < _23._m2; )
        {
            _196[_588] = _107._m0[_79 + _588];
            _588 += 64u;
            continue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _212 = min(_23._m3, _23._m2);
        float _590;
        _590 = 0.0;
        for (uint _589 = gl_LocalInvocationID.x; _589 < _212; )
        {
            _590 += (_102[_589] * _102[_589]);
            _589 += 64u;
            continue;
        }
        float _238 = simd_sum(_590);
        if (gl_SubgroupSize < 64u)
        {
            _sg_reduce[gl_LocalInvocationID.x / gl_SubgroupSize] = _238;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationID.x == 0u)
            {
                uint _n_sg = (64u + gl_SubgroupSize - 1u) / gl_SubgroupSize;
                float _total = 0.0;
                for (uint _si = 0u; _si < _n_sg; _si++)
                    _total += _sg_reduce[_si];
                _sg_reduce[0] = _total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            _238 = _sg_reduce[0];
        }
        float _250 = rsqrt(fast::max(_238, 9.9999999600419720025001879548654e-13)) / sqrt(float(_23._m3));
        for (uint _591 = gl_LocalInvocationID.x; _591 < _212; )
        {
            _102[_591] *= _250;
            _591 += 64u;
            continue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _593;
        _593 = 0.0;
        for (uint _592 = gl_LocalInvocationID.x; _592 < _212; )
        {
            _593 += (_157[_592] * _157[_592]);
            _592 += 64u;
            continue;
        }
        float _298 = simd_sum(_593);
        if (gl_SubgroupSize < 64u)
        {
            _sg_reduce[gl_LocalInvocationID.x / gl_SubgroupSize] = _298;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gl_LocalInvocationID.x == 0u)
            {
                uint _n_sg_k = (64u + gl_SubgroupSize - 1u) / gl_SubgroupSize;
                float _total_k = 0.0;
                for (uint _si = 0u; _si < _n_sg_k; _si++)
                    _total_k += _sg_reduce[_si];
                _sg_reduce[0] = _total_k;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            _298 = _sg_reduce[0];
        }
        float _302 = rsqrt(fast::max(_298, 9.9999999600419720025001879548654e-13));
        for (uint _594 = gl_LocalInvocationID.x; _594 < _212; )
        {
            _157[_594] *= _302;
            _594 += 64u;
            continue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (gl_LocalInvocationID.x == 0u)
        {
            float _595;
            if (_23._m7 != 0u)
            {
                float _596;
                if (_23._m6 != 0u)
                {
                    _596 = _329._m0[gl_WorkGroupID.x] + float(_349._m0[gl_WorkGroupID.x]);
                }
                else
                {
                    _596 = _329._m0[gl_WorkGroupID.x] + _361._m0[gl_WorkGroupID.x];
                }
                _595 = _596;
            }
            else
            {
                _595 = _329._m0[gl_WorkGroupID.x];
            }
            float _372 = log(1.0 + exp(_595));
            float _598;
            if (_23._m8 != 0u)
            {
                float _597;
                if (_23._m5 != 0u)
                {
                    _597 = float(_389._m0[gl_WorkGroupID.x]);
                }
                else
                {
                    _597 = _398._m0[gl_WorkGroupID.x];
                }
                _598 = _372 * _597;
            }
            else
            {
                _598 = -_372;
            }
            _409 = exp(_598);
            _412 = 1.0 / (1.0 + exp(-_416._m0[gl_WorkGroupID.x]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _441 = min(_23._m2, _23._m3);
        for (uint _599 = gl_LocalInvocationID.x; _599 < _23._m2; _599 += 64u)
        {
            uint _613 = _78 + _599;
            uint _459 = _23._m2 * _613;
            for (uint _600 = 0u; _600 < _23._m2; )
            {
                uint _476 = _459 + _600;
                _473._m0[_476] *= _409;
                _600++;
                continue;
            }
            float _604;
            _604 = 0.0;
            for (uint _601 = 0u; _601 < _441; )
            {
                _604 += (_473._m0[_459 + _601] * _157[_601]);
                _601++;
                continue;
            }
            float _514 = _412 * (_196[_599] - _604);
            for (uint _605 = 0u; _605 < _441; )
            {
                uint _526 = _459 + _605;
                _473._m0[_526] += (_157[_605] * _514);
                _605++;
                continue;
            }
            float _609;
            _609 = 0.0;
            for (uint _606 = 0u; _606 < _441; )
            {
                _609 += (_473._m0[_459 + _606] * _102[_606]);
                _606++;
                continue;
            }
            _564._m0[_613] = _609;
        }
        break;
    } while(false);
}
