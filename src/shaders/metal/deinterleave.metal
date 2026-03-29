#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _17
{
    uint _m0;
    uint _m1;
};

struct _69
{
    float _m0[1];
};

struct _74
{
    float _m0[1];
};

struct _83
{
    float _m0[1];
};

kernel void main0(constant _17& _19 [[buffer(0)]], device _69& _71 [[buffer(1)]], device _74& _76 [[buffer(2)]], device _83& _85 [[buffer(3)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= (_19._m0 * _19._m1))
        {
            break;
        }
        uint _55 = 2u * (((gl_GlobalInvocationID.x / _19._m0) * _19._m0) + (gl_GlobalInvocationID.x % _19._m0));
        _71._m0[gl_GlobalInvocationID.x] = _76._m0[_55];
        _85._m0[gl_GlobalInvocationID.x] = _76._m0[_55 + 1u];
        break;
    } while(false);
}

