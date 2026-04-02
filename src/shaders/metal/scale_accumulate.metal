#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _17
{
    uint _m0;
    uint _m1;
};

struct _37
{
    float _m0[1];
};

struct _43
{
    float _m0[1];
};

kernel void main0(constant _17& _19 [[buffer(0)]], device _37& _39 [[buffer(1)]], device _43& _45 [[buffer(2)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    if (gl_GlobalInvocationID.x < _19._m0)
    {
        _39._m0[gl_GlobalInvocationID.x] += (as_type<float>(_19._m1) * _45._m0[gl_GlobalInvocationID.x]);
    }
}

