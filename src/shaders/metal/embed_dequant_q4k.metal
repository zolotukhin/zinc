#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _26
{
    uchar _m0[1];
};

struct _124
{
    uint _m0[1];
};

struct _131
{
    uint _m0;
    uint _m1;
};

struct _276
{
    float _m0[1];
};

kernel void main0(device _26& _28 [[buffer(0)]], device _124& _126 [[buffer(1)]], constant _131& _133 [[buffer(2)]], device _276& _278 [[buffer(3)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]])
{
    do
    {
        uint _138 = _133._m0 / 256u;
        uint _142 = gl_LocalInvocationID.x / 8u;
        uint _145 = gl_LocalInvocationID.x % 8u;
        if (_142 >= _138)
        {
            break;
        }
        bool _318;
        uint _166 = (_133._m1 + ((_126._m0[0] * _138) * 144u)) + (_142 * 144u);
        uint _207 = _166 + 4u;
        float _399;
        do
        {
            _318 = _145 < 4u;
            if (_318)
            {
                _399 = float(uint(_28._m0[_207 + _145]) & 63u);
                break;
            }
            else
            {
                uint _331 = _207 + _145;
                _399 = float((uint(_28._m0[_331 + 4u]) & 15u) | (((uint(_28._m0[_331 - 4u]) >> uint(6)) & 3u) << uint(4)));
                break;
            }
            break; // unreachable workaround
        } while(false);
        float _400;
        do
        {
            if (_318)
            {
                _400 = float(uint(_28._m0[(_207 + _145) + 4u]) & 63u);
                break;
            }
            else
            {
                uint _376 = _207 + _145;
                _400 = float(((uint(_28._m0[_376 + 4u]) >> uint(4)) & 15u) | (((uint(_28._m0[_376]) >> uint(6)) & 3u) << uint(4)));
                break;
            }
            break; // unreachable workaround
        } while(false);
        float _223 = float2(as_type<half2>(uint(_28._m0[_166]) | (uint(_28._m0[_166 + 1u]) << uint(8)))).x * _399;
        float _227 = float2(as_type<half2>(uint(_28._m0[_166 + 2u]) | (uint(_28._m0[_166 + 3u]) << uint(8)))).x * _400;
        uint _233 = _145 * 32u;
        uint _234 = (_142 * 256u) + _233;
        uint _238 = _166 + 16u;
        for (uint _401 = 0u; _401 < 32u; _401++)
        {
            uint _403;
            if (_318)
            {
                _403 = uint(_28._m0[(_238 + _233) + _401]) & 15u;
            }
            else
            {
                _403 = (uint(_28._m0[(_238 + ((_145 - 4u) * 32u)) + _401]) >> uint(4)) & 15u;
            }
            _278._m0[_234 + _401] = (_223 * float(_403)) - _227;
        }
        break;
    } while(false);
}

