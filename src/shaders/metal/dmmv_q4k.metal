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

struct _36
{
    uchar _m0[1];
};

#ifndef SPIRV_CROSS_CONSTANT_ID_1
#define SPIRV_CROSS_CONSTANT_ID_1 4096u
#endif
constant uint _226 = SPIRV_CROSS_CONSTANT_ID_1;

struct _266
{
    uint _m0;
    uint _m1;
    uint _m2;
    uint _m3;
    uint _m4;
};

struct _292
{
    float _m0[1];
};

struct _406
{
    float _m0[1];
};

kernel void main0(device _36& _38 [[buffer(0)]], constant _266& _268 [[buffer(1)]], device _292& _294 [[buffer(2)]], device _406& _408 [[buffer(3)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    threadgroup spvUnsafeArray<float, _226> _229;
    do
    {
        uint _273 = _268._m1 / 256u;
        uint _278 = _268._m3 / 4u;
        for (uint _735 = gl_LocalInvocationID.x; _735 < _268._m1; )
        {
            _229[_735] = _294._m0[_278 + _735];
            _735 += 64u;
            continue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _312 = (gl_WorkGroupID.x * 64u) + gl_LocalInvocationID.x;
        if (_312 >= _268._m0)
        {
            break;
        }
        float _737;
        _737 = 0.0;
        float _739;
        for (uint _736 = 0u; _736 < _273; _737 = _739, _736++)
        {
            uint _333 = _736 * 256u;
            uint _346 = (_268._m2 + ((_312 * _273) * 144u)) + (_736 * 144u);
            float2 _376 = float2(as_type<half2>(uint(_38._m0[_346]) | (uint(_38._m0[_346 + 1u]) << uint(8))));
            float _377 = _376.x;
            float2 _380 = float2(as_type<half2>(uint(_38._m0[_346 + 2u]) | (uint(_38._m0[_346 + 3u]) << uint(8))));
            float _381 = _380.x;
            uint _384 = _346 + 16u;
            uint _386 = _346 + 4u;
            _739 = _737;
            float _751;
            for (uint _738 = 0u; _738 < 4u; _739 = _751, _738++)
            {
                bool _565;
                uint _465 = _738 * 2u;
                uint _468 = _465 + 1u;
                float _741;
                do
                {
                    _565 = _465 < 4u;
                    if (_565)
                    {
                        _741 = float(uint(_38._m0[_386 + _465]) & 63u);
                        break;
                    }
                    else
                    {
                        uint _578 = _386 + _465;
                        _741 = float((uint(_38._m0[_578 + 4u]) & 15u) | (((uint(_38._m0[_578 - 4u]) >> uint(6)) & 3u) << uint(4)));
                        break;
                    }
                    break; // unreachable workaround
                } while(false);
                float _742;
                do
                {
                    if (_565)
                    {
                        _742 = float(uint(_38._m0[(_386 + _465) + 4u]) & 63u);
                        break;
                    }
                    else
                    {
                        uint _623 = _386 + _465;
                        _742 = float(((uint(_38._m0[_623 + 4u]) >> uint(4)) & 15u) | (((uint(_38._m0[_623]) >> uint(6)) & 3u) << uint(4)));
                        break;
                    }
                    break; // unreachable workaround
                } while(false);
                bool _654;
                float _477 = _377 * _741;
                float _480 = _381 * _742;
                float _743;
                do
                {
                    _654 = _468 < 4u;
                    if (_654)
                    {
                        _743 = float(uint(_38._m0[_386 + _468]) & 63u);
                        break;
                    }
                    else
                    {
                        uint _667 = _386 + _468;
                        _743 = float((uint(_38._m0[_667 + 4u]) & 15u) | (((uint(_38._m0[_667 - 4u]) >> uint(6)) & 3u) << uint(4)));
                        break;
                    }
                    break; // unreachable workaround
                } while(false);
                float _744;
                do
                {
                    if (_654)
                    {
                        _744 = float(uint(_38._m0[(_386 + _468) + 4u]) & 63u);
                        break;
                    }
                    else
                    {
                        uint _712 = _386 + _468;
                        _744 = float(((uint(_38._m0[_712 + 4u]) >> uint(4)) & 15u) | (((uint(_38._m0[_712]) >> uint(6)) & 3u) << uint(4)));
                        break;
                    }
                    break; // unreachable workaround
                } while(false);
                float _489 = _377 * _743;
                float _492 = _381 * _744;
                uint _496 = _333 + (_738 * 64u);
                uint _500 = _333 + (_468 * 32u);
                uint _504 = _384 + (_738 * 32u);
                _751 = _739;
                for (uint _749 = 0u; _749 < 32u; )
                {
                    uint _515 = uint(_38._m0[_504 + _749]);
                    _751 = (_751 + (((_477 * float(_515 & 15u)) - _480) * _229[_496 + _749])) + (((_489 * float((_515 >> uint(4)) & 15u)) - _492) * _229[_500 + _749]);
                    _749++;
                    continue;
                }
            }
        }
        _408._m0[(_268._m4 / 4u) + _312] = _737;
        break;
    } while(false);
}

