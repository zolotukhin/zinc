#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _17
{
    uint _m0;
    uint _m1;
    uint _m2;
    uint _m3;
    uint _m4;
};

struct _67
{
    float _m0[1];
};

struct _77
{
    float _m0[1];
};

struct _91
{
    float _m0[1];
};

kernel void main0(constant _17& _19 [[buffer(0)]], device _67& _69 [[buffer(1)]], device _77& _79 [[buffer(2)]], device _91& _93 [[buffer(3)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= _19._m0)
        {
            break;
        }
        uint _41 = (_19._m2 / 4u) + (gl_GlobalInvocationID.x * _19._m1);
        uint _46 = _19._m3 / 4u;
        float _109;
        _109 = 0.0;
        for (uint _108 = 0u; _108 < _19._m1; )
        {
            _109 += (_69._m0[_41 + _108] * _79._m0[_46 + _108]);
            _108++;
            continue;
        }
        _93._m0[(_19._m4 / 4u) + gl_GlobalInvocationID.x] = _109;
        break;
    } while(false);
}

