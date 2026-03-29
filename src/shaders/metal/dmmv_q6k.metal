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

struct _301
{
    float _m0[1];
};

struct _379
{
    float _m0[1];
};

kernel void main0(constant _28& _30 [[buffer(0)]], device _89& _91 [[buffer(1)]], device _301& _303 [[buffer(2)]], device _379& _381 [[buffer(3)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= _30._m0)
        {
            break;
        }
        uint _46 = _30._m1 / 256u;
        uint _56 = _30._m2 + ((gl_GlobalInvocationID.x * _46) * 210u);
        uint _62 = _30._m3 / 4u;
        float _401;
        _401 = 0.0;
        uint _406;
        float _410;
        for (uint _400 = 0u, _407 = 0u; _400 < _46; _401 = _410, _400++, _407 = _406)
        {
            uint _85 = _56 + (_400 * 210u);
            float2 _398 = float2(as_type<half2>(uint(_91._m0[_85 + 208u]) | (uint(_91._m0[_85 + 209u]) << uint(8))));
            float _399 = _398.x;
            _406 = _407;
            _410 = _401;
            uint _373;
            float _409;
            for (uint _402 = 0u; _402 < 2u; _406 = _373, _402++, _410 = _409)
            {
                uint _126 = _85 + (_402 * 64u);
                uint _134 = (_85 + 128u) + (_402 * 32u);
                uint _142 = (_85 + 192u) + (_402 * 8u);
                _409 = _410;
                for (uint _404 = 0u; _404 < 32u; )
                {
                    uint _158 = _126 + _404;
                    uint _161 = uint(_91._m0[_158]);
                    uint _169 = uint(_91._m0[_158 + 32u]);
                    uint _176 = uint(_91._m0[_134 + _404]);
                    uint _215 = _142 + (_404 / 16u);
                    uint _218 = uint(_91._m0[_215]);
                    uint _238 = uint(_91._m0[_215 + 2u]);
                    uint _258 = uint(_91._m0[_215 + 4u]);
                    uint _279 = uint(_91._m0[_215 + 6u]);
                    uint _308 = (_62 + _406) + _404;
                    _409 = (((_409 + (((_399 * (float(int(_218)) - ((_218 >= 128u) ? 256.0 : 0.0))) * (float((_161 & 15u) | ((_176 & 3u) << uint(4))) - 32.0)) * _303._m0[_308])) + (((_399 * (float(int(_238)) - ((_238 >= 128u) ? 256.0 : 0.0))) * (float((_169 & 15u) | (((_176 >> uint(2)) & 3u) << uint(4))) - 32.0)) * _303._m0[_308 + 32u])) + (((_399 * (float(int(_258)) - ((_258 >= 128u) ? 256.0 : 0.0))) * (float((_161 >> uint(4)) | (((_176 >> uint(4)) & 3u) << uint(4))) - 32.0)) * _303._m0[_308 + 64u])) + (((_399 * (float(int(_279)) - ((_279 >= 128u) ? 256.0 : 0.0))) * (float((_169 >> uint(4)) | (((_176 >> uint(6)) & 3u) << uint(4))) - 32.0)) * _303._m0[_308 + 96u]);
                    _404++;
                    continue;
                }
                _373 = _406 + 128u;
            }
        }
        _381._m0[(_30._m4 / 4u) + gl_GlobalInvocationID.x] = _401;
        break;
    } while(false);
}

