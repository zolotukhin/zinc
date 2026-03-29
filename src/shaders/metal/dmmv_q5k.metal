#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _28
{
    uint _m0;
    uint _m1;
    uint _m2;
    uint _m3;
    uint _m4;
};

struct _89
{
    uchar _m0[1];
};

struct _367
{
    float _m0[1];
};

struct _403
{
    float _m0[1];
};

kernel void main0(constant _28& _30 [[buffer(0)]], device _89& _91 [[buffer(1)]], device _367& _369 [[buffer(2)]], device _403& _405 [[buffer(3)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= _30._m0)
        {
            break;
        }
        uint _46 = _30._m1 / 256u;
        uint _56 = _30._m2 + ((gl_GlobalInvocationID.x * _46) * 176u);
        uint _62 = _30._m3 / 4u;
        float _429;
        _429 = 0.0;
        uint _442;
        float _450;
        for (uint _428 = 0u, _443 = 0u; _428 < _46; _429 = _450, _428++, _443 = _442)
        {
            uint _85 = _56 + (_428 * 176u);
            float2 _421 = float2(as_type<half2>(uint(_91._m0[_85]) | (uint(_91._m0[_85 + 1u]) << uint(8))));
            float _422 = _421.x;
            float2 _426 = float2(as_type<half2>(uint(_91._m0[_85 + 2u]) | (uint(_91._m0[_85 + 3u]) << uint(8))));
            float _427 = _426.x;
            _442 = _443;
            _450 = _429;
            uint _397;
            float _447;
            for (uint _430 = 0u; _430 < 4u; _442 = _397, _430++, _450 = _447)
            {
                uint _139 = _430 * 2u;
                uint _143 = _139 + 1u;
                uint _433;
                uint _435;
                if (_139 < 4u)
                {
                    uint _152 = (_85 + 4u) + _139;
                    _435 = uint(_91._m0[_152 + 4u]) & 63u;
                    _433 = uint(_91._m0[_152]) & 63u;
                }
                else
                {
                    uint _172 = (_85 + 4u) + _139;
                    uint _176 = uint(_91._m0[_172 + 4u]);
                    _435 = (_176 >> uint(4)) | ((uint(_91._m0[_172]) >> uint(6)) << uint(4));
                    _433 = (_176 & 15u) | ((uint(_91._m0[_172 - 4u]) >> uint(6)) << uint(4));
                }
                uint _436;
                uint _437;
                if (_143 < 4u)
                {
                    uint _218 = (_85 + 4u) + _143;
                    _437 = uint(_91._m0[_218 + 4u]) & 63u;
                    _436 = uint(_91._m0[_218]) & 63u;
                }
                else
                {
                    uint _237 = (_85 + 4u) + _143;
                    uint _241 = uint(_91._m0[_237 + 4u]);
                    _437 = (_241 >> uint(4)) | ((uint(_91._m0[_237]) >> uint(6)) << uint(4));
                    _436 = (_241 & 15u) | ((uint(_91._m0[_237 - 4u]) >> uint(6)) << uint(4));
                }
                float _277 = _422 * float(_433);
                float _282 = _427 * float(_435);
                float _287 = _422 * float(_436);
                float _292 = _427 * float(_437);
                uint _300 = (_85 + 48u) + (_430 * 32u);
                uint _308 = _62 + _442;
                _447 = _450;
                for (uint _444 = 0u; _444 < 32u; )
                {
                    uint _323 = uint(_91._m0[_300 + _444]);
                    uint _338 = uint(_91._m0[(_85 + 16u) + _444]);
                    _447 = (_447 + (((_277 * float((_323 & 15u) | (((_338 >> _139) & 1u) << uint(4)))) - _282) * _369._m0[_308 + _444])) + (((_287 * float((_323 >> uint(4)) | (((_338 >> _143) & 1u) << uint(4)))) - _292) * _369._m0[(_308 + 32u) + _444]);
                    _444++;
                    continue;
                }
                _397 = _442 + 64u;
            }
        }
        _405._m0[(_30._m4 / 4u) + gl_GlobalInvocationID.x] = _429;
        break;
    } while(false);
}

