#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _17
{
    uint _m0;
};

struct _34
{
    float _m0[1];
};

struct _46
{
    float _m0[1];
};

struct _51
{
    float _m0[1];
};

kernel void main0(constant _17& _19 [[buffer(0)]], device _34& _36 [[buffer(1)]], device _46& _48 [[buffer(2)]], device _51& _53 [[buffer(3)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    if (gl_GlobalInvocationID.x < _19._m0)
    {
        _48._m0[gl_GlobalInvocationID.x] = _53._m0[gl_GlobalInvocationID.x] * (1.0 / (1.0 + exp(-_36._m0[gl_GlobalInvocationID.x])));
    }
}

