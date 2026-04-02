#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct _29
{
    uint _m0;
    uint _m1;
    uint _m2;
    uint _m3;
    uint _m4;
};

struct _87
{
    uchar _m0[1];
};

struct _131
{
    float _m0[1];
};

struct _283
{
    float _m0[1];
};

kernel void main0(constant _29& _31 [[buffer(0)]], device _87& _89 [[buffer(1)]], device _131& _133 [[buffer(2)]], device _283& _285 [[buffer(3)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint gl_NumSubgroups [[simdgroups_per_threadgroup]], uint gl_SubgroupID [[simdgroup_index_in_threadgroup]])
{
    threadgroup spvUnsafeArray<float, 2> _242;
    threadgroup spvUnsafeArray<float, 2> _248;
    uint _19 = gl_WorkGroupID.x * 2u;
    uint _23 = _19 + 1u;
    uint _38 = _31._m1 / 32u;
    float _306;
    float _307;
    _307 = 0.0;
    _306 = 0.0;
    float _328;
    float _330;
    for (uint _305 = gl_LocalInvocationID.x; _305 < _38; _307 = _330, _306 = _328, _305 += 64u)
    {
        uint _63 = (_31._m3 / 4u) + (_305 * 32u);
        if (_19 < _31._m0)
        {
            uint _83 = (_31._m2 + ((_19 * _38) * 34u)) + (_305 * 34u);
            uint _110 = _83 + 2u;
            float _316;
            _316 = 0.0;
            for (uint _315 = 0u; _315 < 32u; )
            {
                _316 += (float(char(_89._m0[_110 + _315])) * _133._m0[_63 + _315]);
                _315++;
                continue;
            }
            _328 = _306 + (float2(as_type<half2>(uint(_89._m0[_83]) | (uint(_89._m0[_83 + 1u]) << uint(8)))).x * _316);
        }
        else
        {
            _328 = _306;
        }
        if (_23 < _31._m0)
        {
            uint _169 = (_31._m2 + ((_23 * _38) * 34u)) + (_305 * 34u);
            uint _188 = _169 + 2u;
            float _321;
            _321 = 0.0;
            for (uint _320 = 0u; _320 < 32u; )
            {
                _321 += (float(char(_89._m0[_188 + _320])) * _133._m0[_63 + _320]);
                _320++;
                continue;
            }
            _330 = _307 + (float2(as_type<half2>(uint(_89._m0[_169]) | (uint(_89._m0[_169 + 1u]) << uint(8)))).x * _321);
        }
        else
        {
            _330 = _307;
        }
    }
    float _229 = simd_sum(_306);
    float _231 = simd_sum(_307);
    float _308;
    float _312;
    if (gl_NumSubgroups > 1u)
    {
        bool _237 = simd_is_first();
        if (_237)
        {
            _242[gl_SubgroupID] = _229;
            _248[gl_SubgroupID] = _231;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _309;
        float _313;
        if (gl_LocalInvocationID.x == 0u)
        {
            _313 = _248[0] + _248[1];
            _309 = _242[0] + _242[1];
        }
        else
        {
            _313 = _231;
            _309 = _229;
        }
        _312 = _313;
        _308 = _309;
    }
    else
    {
        _312 = _231;
        _308 = _229;
    }
    if (gl_LocalInvocationID.x == 0u)
    {
        uint _275 = _31._m4 / 4u;
        if (_19 < _31._m0)
        {
            _285._m0[_275 + _19] = _308;
        }
        if (_23 < _31._m0)
        {
            _285._m0[_275 + _23] = _312;
        }
    }
}

