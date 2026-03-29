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

struct _50
{
    float _m0[1];
};

struct _56
{
    float _m0[1];
};

kernel void main0(constant _17& _19 [[buffer(0)]], device _34& _36 [[buffer(1)]], device _50& _52 [[buffer(2)]], device _56& _58 [[buffer(3)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= _19._m0)
        {
            break;
        }
        _52._m0[gl_GlobalInvocationID.x] = (_36._m0[gl_GlobalInvocationID.x] / (1.0 + exp(-_36._m0[gl_GlobalInvocationID.x]))) * _58._m0[gl_GlobalInvocationID.x];
        break;
    } while(false);
}

