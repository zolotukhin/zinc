#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _23
{
    uint _m0;
    uint _m1;
    uint _m2;
    uint _m3;
    uint _m4;
};

struct _57
{
    float _m0[1];
};

struct _102
{
    float _m0[1];
};

kernel void main0(constant _23& _25 [[buffer(0)]], device _57& _59 [[buffer(1)]], device _102& _104 [[buffer(2)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    float _31 = as_type<float>(_25._m4);
    uint _37 = _25._m1 / 2u;
    uint _43 = gl_WorkGroupID.x * _25._m0;
    for (uint _159 = gl_LocalInvocationID.x; _159 < _37; )
    {
        uint _62 = _43 + _159;
        float _65 = _59._m0[_62];
        uint _71 = _62 + _37;
        float _73 = _59._m0[_71];
        float _94 = float(_25._m3) * (1.0 / powr(_31, float(2u * _159) / float(_25._m1)));
        float _97 = cos(_94);
        float _100 = sin(_94);
        _104._m0[_62] = (_65 * _97) - (_73 * _100);
        _104._m0[_71] = (_65 * _100) + (_73 * _97);
        _159 += 64u;
        continue;
    }
    uint _136 = _25._m1 + gl_LocalInvocationID.x;
    for (uint _160 = _136; _160 < _25._m0; )
    {
        uint _148 = _43 + _160;
        _104._m0[_148] = _59._m0[_148];
        _160 += 64u;
        continue;
    }
}
