#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _17
{
    uint _m0;
    uint _m1;
    uint _m2;
};

struct _66
{
    half _m0[1];
};

struct _76
{
    float _m0[1];
};

struct _90
{
    float _m0[1];
};

struct _103
{
    float _m0[1];
};

struct _124
{
    float _m0[1];
};

kernel void main0(constant _17& _19 [[buffer(0)]], device void* spvBufferAliasSet0Binding1 [[buffer(1)]], device _90& _92 [[buffer(2)]], device _103& _105 [[buffer(3)]], device _124& _126 [[buffer(4)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    device auto& _68 = *(device _66*)spvBufferAliasSet0Binding1;
    device auto& _78 = *(device _76*)spvBufferAliasSet0Binding1;
    do
    {
        if (gl_GlobalInvocationID.x >= _19._m0)
        {
            break;
        }
        uint _35 = _19._m1 - 1u;
        float _184;
        _184 = 0.0;
        float _113;
        for (uint _183 = 0u; _183 < _19._m1; _184 = _113, _183++)
        {
            uint _56 = (gl_GlobalInvocationID.x * _19._m1) + _183;
            float _188;
            if (_19._m2 != 0u)
            {
                _188 = float(_68._m0[_56]);
            }
            else
            {
                _188 = _78._m0[_56];
            }
            float _189;
            if (_183 < _35)
            {
                _189 = _92._m0[(_183 * _19._m0) + gl_GlobalInvocationID.x];
            }
            else
            {
                _189 = _105._m0[gl_GlobalInvocationID.x];
            }
            _113 = _184 + (_188 * _189);
        }
        _126._m0[gl_GlobalInvocationID.x] = _184 * (1.0 / (1.0 + exp(-_184)));
        if (_35 > 1u)
        {
            for (uint _185 = 0u; _185 < (_19._m1 - 2u); )
            {
                _92._m0[(_185 * _19._m0) + gl_GlobalInvocationID.x] = _92._m0[((_185 + 1u) * _19._m0) + gl_GlobalInvocationID.x];
                _185++;
                continue;
            }
        }
        _92._m0[((_19._m1 - 2u) * _19._m0) + gl_GlobalInvocationID.x] = _105._m0[gl_GlobalInvocationID.x];
        break;
    } while(false);
}

