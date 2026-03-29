#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _17
{
    uint _m0;
};

struct _31
{
    float _m0[1];
};

struct _36
{
    float _m0[1];
};

struct _44
{
    float _m0[1];
};

kernel void main0(constant _17& _19 [[buffer(0)]], device _31& _33 [[buffer(1)]], device _36& _38 [[buffer(2)]], device _44& _46 [[buffer(3)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    if (gl_GlobalInvocationID.x < _19._m0)
    {
        _33._m0[gl_GlobalInvocationID.x] = _38._m0[gl_GlobalInvocationID.x] + _46._m0[gl_GlobalInvocationID.x];
    }
}

