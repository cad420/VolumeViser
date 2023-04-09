#undef UTIL_ENABLE_OPENGL
#undef UTIL_ENABLE_OPENGL

#include <Core/Renderer.hpp>
#include <Algorithm/LevelOfDetailPolicy.hpp>
#include <unordered_set>
#include <cuda_runtime.h>
#include "Common.hpp"
#include "../Common/helper_math.h"

#include "PBR.hpp"

#define Transform(t) t.x, t.y, t.z


// from exposure render

VISER_BEGIN


using BlockUID = GridVolume::BlockUID;

static Int3 DefaultVTexShape{1024, 1024, 1024};

__device__ __constant__ float3		gAaBbMin;
__device__ __constant__ float3		gAaBbMax;
__device__ __constant__ float3		gInvAaBbMin;
__device__ __constant__ float3		gInvAaBbMax;
__device__ __constant__ float		gIntensityMin;
__device__ __constant__ float		gIntensityMax;
__device__ __constant__ float		gIntensityRange;
__device__ __constant__ float		gIntensityInvRange;
__device__ __constant__ float		gStepSize;
__device__ __constant__ float		gStepSizeShadow;
__device__ __constant__ float		gDensityScale;
__device__ __constant__ float		gGradientDelta;
__device__ __constant__ float		gInvGradientDelta;
__device__ __constant__ float3		gGradientDeltaX;
__device__ __constant__ float3		gGradientDeltaY;
__device__ __constant__ float3		gGradientDeltaZ;
__device__ __constant__ int			gFilmWidth;
__device__ __constant__ int			gFilmHeight;
__device__ __constant__ int			gFilmNoPixels;
__device__ __constant__ int			gFilterWidth;
__device__ __constant__ float		gFilterWeights[10];
__device__ __constant__ float		gExposure;
__device__ __constant__ float		gInvExposure;
__device__ __constant__ float		gGamma;
__device__ __constant__ float		gInvGamma;
__device__ __constant__ float		gDenoiseEnabled;
__device__ __constant__ float		gDenoiseWindowRadius;
__device__ __constant__ float		gDenoiseInvWindowArea;
__device__ __constant__ float		gDenoiseNoise;
__device__ __constant__ float		gDenoiseWeightThreshold;
__device__ __constant__ float		gDenoiseLerpThreshold;
__device__ __constant__ float		gDenoiseLerpC;
__device__ __constant__ float		gNoIterations;
__device__ __constant__ float		gInvNoIterations;


texture<short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexDensity;

texture<short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexGradientMagnitude;

texture<float, cudaTextureType1D, cudaReadModeElementType>			gTexOpacity;
texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexDiffuse;
texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexSpecular;
texture<float, cudaTextureType1D, cudaReadModeElementType>			gTexRoughness;
texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexEmission;

namespace{
    using namespace cuda;

    using HashTableItem = GPUPageTableMgr::PageTableItem;
    static constexpr int HashTableSize = 1024;
    static constexpr int MaxLodLevels = LevelOfDist::MaxLevelCount;
    static constexpr float MachineEpsilon = std::numeric_limits<float>::epsilon() * 0.5f;

    CUB_GPU float gamma(int n) {
        return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
    }

    struct HashTable{
        HashTableItem hash_table[HashTableSize];
    };

    CUB_CPU_GPU inline void swap(float& a, float& b)
    {
        float t = a; a = b; b = t;
    }

    struct CUDAVolumeParams
    {
        AABB bound;
        float3 voxel_dim;
        float3 voxel_space;
        uint32_t block_length;
        uint32_t padding;
        uint32_t block_size;
    };

    struct CUDARenderParams{
        float lod_policy[MaxLodLevels];
        float ray_step;
        float max_ray_dist;//中心的最远距离
        float3 inv_tex_shape;
        bool use_2d_tf;
        int2 mpi_node_offset;
    };

    struct CUDAPBParams{
        int shadow_ray_lod;

    };

    struct CUDAPerFrameParams{
        float3 cam_pos;
        float fov;
        float3 cam_dir;
        int frame_width;
        float3 cam_right;
        int frame_height;
        float3 cam_up;
        float frame_w_over_h;
        int debug_mode;
    };

    struct CUDAPageTable{
        CUDABufferView1D<HashTableItem> table;
    };

    struct CUDAFrameBuffer{
        CUDABufferView2D<uint32_t> color;
        CUDABufferView2D<float> depth;
    };

    struct RayCastResult{
        float4 color;
        float depth;
    };
    struct VirtualSamplingResult{
        uint32_t flag;
        float scalar;
    };



//    CUB_CPU_GPU inline float3 UniformSampleSphere(const float2& U)
//    {
//        float z = 1.f - 2.f * U.x;
//        float r = sqrtf(max(0.f, 1.f - z*z));
//        float phi = 2.f * PI_F * U.y;
//        float x = r * cosf(phi);
//        float y = r * sinf(phi);
//        return make_float3(x, y, z);
//    }


    CUB_CPU_GPU inline float SphericalTheta(const float3& Wl)
    {
        return acosf(clamp(Wl.y, -1.f, 1.f));
    }

    CUB_CPU_GPU inline float SphericalPhi(const float3& Wl)
    {
        float p = atan2f(Wl.z, Wl.x);
        return (p < 0.f) ? p + 2.f * PI_F : p;
    }

    CUB_CPU_GPU bool IsBlack(float3 c)
	{
		if (c.x != 0 || c.y != 0 || c.z != 0 ) return false;
		return true;
	}


    CUB_CPU_GPU inline float3 FromRGB(float r, float g, float b)
	{
		static float CoeffX[3] = { 0.4124f, 0.3576f, 0.1805f };
		static float CoeffY[3] = { 0.2126f, 0.7152f, 0.0722f };
		static float CoeffZ[3] = { 0.0193f, 0.1192f, 0.9505f };

		float XYZ[3];

		XYZ[0] =	CoeffX[0] * r +
					CoeffX[1] * g +
					CoeffX[2] * b;

		XYZ[1] =	CoeffY[0] * r +
					CoeffY[1] * g +
					CoeffY[2] * b;

		XYZ[2] =	CoeffZ[0] * r +
					CoeffZ[1] * g +
					CoeffZ[2] * b;

		return make_float3(XYZ[0], XYZ[1], XYZ[2]);
	}


    CUB_CPU_GPU inline float3 FromRGB(float3 c){
        FromRGB(c.x, c.y, c.z);
    }


    class CRNG{

        public:
            CUB_CPU_GPU CRNG(unsigned int* pSeed0, unsigned int* pSeed1)
            {
                m_pSeed0 = pSeed0;
                m_pSeed1 = pSeed1;
            }

            CUB_CPU_GPU float Get1()
            {
                *m_pSeed0 = 36969 * ((*m_pSeed0) & 65535) + ((*m_pSeed0) >> 16);
                *m_pSeed1 = 18000 * ((*m_pSeed1) & 65535) + ((*m_pSeed1) >> 16);

                unsigned int ires = ((*m_pSeed0) << 16) + (*m_pSeed1);

                union
                {
                    float f;
                    unsigned int ui;
                } res;

                res.ui = (ires & 0x007fffff) | 0x40000000;

                return (res.f - 2.f) / 2.f;
            }

            CUB_CPU_GPU float2 Get2()
            {
                return make_float2(Get1(), Get1());
            }

            CUB_CPU_GPU float3 Get3()
            {
                return make_float3(Get1(), Get1(), Get1());
            }

        private:
            unsigned int*	m_pSeed0;
            unsigned int*	m_pSeed1;
    };

    class CLightSample
    {
    public:
        float2 m_Pos;
        float m_Component;

        CUB_CPU_GPU CLightSample()
        {
            m_Pos	 	= make_float2(0.0f);
            m_Component	= 0.0f;
        }

        CUB_CPU_GPU CLightSample& CLightSample::operator=(const CLightSample& Other)
        {
            m_Pos	 	= Other.m_Pos;
            m_Component = Other.m_Component;

            return *this;
        }

        CUB_GPU void LargeStep(CRNG& Rnd)
        {
            m_Pos		= Rnd.Get2();
            m_Component	= Rnd.Get1();
        }
    };

    class CBrdfSample
    {
    public:
        float	m_Component;
        float2	m_Dir;

        CUB_CPU_GPU CBrdfSample(void)
        {
            m_Component = 0.0f;
            m_Dir 		= make_float2(0.0f);
        }

        CUB_CPU_GPU CBrdfSample(const float& Component, const float2& Dir)
        {
            m_Component = Component;
            m_Dir 		= Dir;
        }

        CUB_CPU_GPU CBrdfSample& CBrdfSample::operator=(const CBrdfSample& Other)
        {
            m_Component = Other.m_Component;
            m_Dir 		= Other.m_Dir;

            return *this;
        }

        CUB_GPU void LargeStep(CRNG& Rnd)
        {
            m_Component	= Rnd.Get1();
            m_Dir		= Rnd.Get2();
        }
    };

    class CLightingSample
    {
    public:
        CBrdfSample		m_BsdfSample;
        CLightSample 	m_LightSample;
        float			m_LightNum;

        CUB_CPU_GPU CLightingSample(void)
        {
            m_LightNum = 0.0f;
        }

        CUB_CPU_GPU CLightingSample& CLightingSample::operator=(const CLightingSample& Other)
        {
            m_BsdfSample	= Other.m_BsdfSample;
            m_LightNum		= Other.m_LightNum;
            m_LightSample	= Other.m_LightSample;

            return *this;
        }

        CUB_GPU void LargeStep(CRNG& Rnd)
        {
            m_BsdfSample.LargeStep(Rnd);
            m_LightSample.LargeStep(Rnd);

            m_LightNum = Rnd.Get1();
        }
    };

    class CCameraSample
    {
    public:
        float2	m_ImageXY;
        float2	m_LensUV;

        CUB_GPU CCameraSample(void)
        {
            m_ImageXY	= make_float2(0.0f);
            m_LensUV	= make_float2(0.0f);
        }

        CUB_GPU CCameraSample& CCameraSample::operator=(const CCameraSample& Other)
        {
            m_ImageXY	= Other.m_ImageXY;
            m_LensUV	= Other.m_LensUV;

            return *this;
        }

        CUB_GPU void LargeStep(float2& ImageUV, float2& LensUV, const int& X, const int& Y, const int& KernelSize)
        {
            m_ImageXY	= make_float2(X + ImageUV.x, Y + ImageUV.y);
            m_LensUV	= LensUV;
        }
    };


    CUB_CPU_GPU inline float3 Lerp(float T, const float3& C1, const float3& C2)
    {
        const float OneMinusT = 1.0f - T;
        return make_float3(OneMinusT * C1.x + T * C2.x, OneMinusT * C1.y + T * C2.y, OneMinusT * C1.z + T * C2.z);
    }
    class CLight
    {
    public:
        float			m_Theta;
        float			m_Phi;
        float			m_Width;
        float			m_InvWidth;
        float			m_HalfWidth;
        float			m_InvHalfWidth;
        float			m_Height;
        float			m_InvHeight;
        float			m_HalfHeight;
        float			m_InvHalfHeight;
        float			m_Distance;
        float			m_SkyRadius;
        float3			m_P;
        float3			m_Target;
        float3			m_N;
        float3			m_U;
        float3			m_V;
        float			m_Area;
        float			m_AreaPdf;
        float3      	m_Color;
        float3      	m_ColorTop;
        float3      	m_ColorMiddle;
        float3      	m_ColorBottom;
        int				m_T;

        CLight(void) :
            m_Theta(0.0f),
            m_Phi(0.0f),
            m_Width(1.0f),
            m_InvWidth(1.0f / m_Width),
            m_HalfWidth(0.5f * m_Width),
            m_InvHalfWidth(1.0f / m_HalfWidth),
            m_Height(1.0f),
            m_InvHeight(1.0f / m_Height),
            m_HalfHeight(0.5f * m_Height),
            m_InvHalfHeight(1.0f / m_HalfHeight),
            m_Distance(1.0f),
            m_SkyRadius(100.0f),
            m_Area(m_Width * m_Height),
            m_AreaPdf(1.0f / m_Area),
            m_T(0){
                          m_P = make_float3(1.0f, 1.0f, 1.0f);
                          m_Target = make_float3(0.0f, 0.0f, 0.0f);
                          m_N = make_float3(1.0f, 0.0f, 0.0f);
                          m_U = make_float3(1.0f, 0.0f, 0.0f);
                          m_V = make_float3(1.0f, 0.0f, 0.0f);
                          m_Color = make_float3(10.0f);
                          m_ColorTop = make_float3(10.0f);
                          m_ColorMiddle = make_float3(10.0f);
                          m_ColorBottom = make_float3(10.0f);
              }

        CUB_CPU_GPU CLight& operator=(const CLight& Other)
        {
            m_Theta				= Other.m_Theta;
            m_Phi				= Other.m_Phi;
            m_Width				= Other.m_Width;
            m_InvWidth			= Other.m_InvWidth;
            m_HalfWidth			= Other.m_HalfWidth;
            m_InvHalfWidth		= Other.m_InvHalfWidth;
            m_Height			= Other.m_Height;
            m_InvHeight			= Other.m_InvHeight;
            m_HalfHeight		= Other.m_HalfHeight;
            m_InvHalfHeight		= Other.m_InvHalfHeight;
            m_Distance			= Other.m_Distance;
            m_SkyRadius			= Other.m_SkyRadius;
            m_P					= Other.m_P;
            m_Target			= Other.m_Target;
            m_N					= Other.m_N;
            m_U					= Other.m_U;
            m_V					= Other.m_V;
            m_Area				= Other.m_Area;
            m_AreaPdf			= Other.m_AreaPdf;
            m_Color				= Other.m_Color;
            m_ColorTop			= Other.m_ColorTop;
            m_ColorMiddle		= Other.m_ColorMiddle;
            m_ColorBottom		= Other.m_ColorBottom;
            m_T					= Other.m_T;

            return *this;
        }

        // CUB_CPU_GPU void Update(const CBoundingBox& BoundingBox)
        // {
        //     m_InvWidth		= 1.0f / m_Width;
        //     m_HalfWidth		= 0.5f * m_Width;
        //     m_InvHalfWidth	= 1.0f / m_HalfWidth;
        //     m_InvHeight		= 1.0f / m_Height;
        //     m_HalfHeight	= 0.5f * m_Height;
        //     m_InvHalfHeight	= 1.0f / m_HalfHeight;
        //     m_Target		= BoundingBox.GetCenter(); //todo

        //     // Determine light position
        //     m_P.x = m_Distance * cosf(m_Phi) * sinf(m_Theta);
        //     m_P.z = m_Distance * cosf(m_Phi) * cosf(m_Theta);
        //     m_P.y = m_Distance * sinf(m_Phi);

        //     m_P += m_Target;

        //     // Determine area
        //     if (m_T == 0)
        //     {
        //         m_Area		= m_Width * m_Height;
        //         m_AreaPdf	= 1.0f / m_Area;
        //     }

        //     if (m_T == 1)
        //     {
        //         m_P				= BoundingBox.GetCenter();
        //         m_SkyRadius		= 1000.0f * length((BoundingBox.GetMaxP() - BoundingBox.GetMinP()));
        //         m_Area			= 4.0f * PI_F * powf(m_SkyRadius, 2.0f);
        //         m_AreaPdf		= 1.0f / m_Area;
        //     }

        //     // Compute orthogonal basis frame
        //     m_N = normalize(m_Target - m_P);
        //     m_U	= normalize(Cross(m_N, make_float3(0.0f, 1.0f, 0.0f)));
        //     m_V	= normalize(Cross(m_N, m_U));
        // }

        // Samples the light
        CUB_CPU_GPU float3 SampleL(const float3& P, RayExt& Rl, float& Pdf, CLightingSample& LS)
        {
            float3 L = make_float3(0);

            if (m_T == 0)
            {
                Rl.o	= m_P + ((-0.5f + LS.m_LightSample.m_Pos.x) * m_Width * m_U) + ((-0.5f + LS.m_LightSample.m_Pos.y) * m_Height * m_V);
                Rl.d	= normalize(P - Rl.o);
                L		= dot(Rl.d, m_N) > 0.0f ? Le(make_float2(0.0f)) : make_float3(0);
                Pdf		= fabsf(dot(Rl.d, m_N)) > 0.0f ? dot((P - Rl.o), (P - Rl.o)) / (fabsf(dot(Rl.d, m_N)) * m_Area) : 0.0f;
            }

            if (m_T == 1)
            {
                Rl.o	= m_P + m_SkyRadius * cuda::UniformSampleSphere(LS.m_LightSample.m_Pos);
                Rl.d	= normalize(P - Rl.o);
                L		= Le(make_float2(1.0f) - 2.0f * LS.m_LightSample.m_Pos);
                Pdf		= powf(m_SkyRadius, 2.0f) / m_Area;
            }

            Rl.min_t	= 0.0f;
            Rl.max_t	= length(P - Rl.o);

            return L;
        }

        // Intersect ray with light
        CUB_CPU_GPU bool Intersect(RayExt& R, float& T, float3& L, float2* pUV = NULL, float* pPdf = NULL)
        {
            if (m_T == 0)
            {
                // Compute projection
                const float dotN = dot(R.d, m_N);

                // Rays is co-planar with light surface
                if (dotN >= 0.0f)
                    return false;

                // Compute hit distance
                T = (-m_Distance - dot(R.o, m_N)) / dotN; // ?????????

                // Intersection is in ray's negative direction
                if (T < R.min_t || T > R.max_t)
                    return false;

                // Determine position on light
                const float3 Pl = R.o + normalize(R.d) * T;

                // Vector from point on area light to center of area light
                const float3 Wl = Pl - m_P;

                // Compute texture coordinates
                const float2 UV = make_float2(dot(Wl, m_U), dot(Wl, m_V));

                // Check if within bounds of light surface
                if (UV.x > m_HalfWidth || UV.x < -m_HalfWidth || UV.y > m_HalfHeight || UV.y < -m_HalfHeight)
                    return false;

                R.max_t = T;

                if (pUV)
                    *pUV = UV;

                if (dotN < 0.0f)
                    L = FromRGB(m_Color.x, m_Color.y, m_Color.z) / m_Area;
                else
                    L = make_float3(0);

                if (pPdf)
                    *pPdf = dot((R.o - Pl), (R.o - Pl)) / (dotN * m_Area);

                return true;
            }

            if (m_T == 1)
            {
                T = m_SkyRadius;

                // Intersection is in ray's negative direction
                if (T < R.min_t || T > R.max_t)
                    return false;

                R.max_t = T;

                float2 UV = make_float2(SphericalPhi(R.d) * INV_TWO_PI_F, SphericalTheta(R.d) * INV_PI_F);

                L	= Le(make_float2(1.0f) - 2.0f * UV);

                if (pPdf)
                    *pPdf = powf(m_SkyRadius, 2.0f) / m_Area;

                return true;
            }

            return false;
        }

        CUB_CPU_GPU float Pdf(const float3& P, const float3& Wi)
        {
            float3 L;
            float2 UV;
            float Pdf = 1.0f;

            RayExt Rl = RayExt(P, Wi, 0.0f, INF_MAX);

            if (m_T == 0)
            {
                float T = 0.0f;

                if (!Intersect(Rl, T, L, NULL, &Pdf))
                    return 0.0f;

                return powf(T, 2.0f) / (fabsf(dot(m_N, -Wi)) * m_Area);
            }

            if (m_T == 1)
            {
                return powf(m_SkyRadius, 2.0f) / m_Area;
            }

            return 0.0f;
        }

        CUB_CPU_GPU float3 Le(const float2& UV)
        {
            if (m_T == 0)
                return FromRGB(m_Color.x, m_Color.y, m_Color.z) / m_Area;

            if (m_T == 1)
            {
                if (UV.y > 0.0f)
                    return FromRGB(Lerp(fabs(UV.y), m_ColorMiddle, m_ColorTop));
                else
                    return FromRGB(Lerp(fabs(UV.y), m_ColorMiddle, m_ColorBottom));
            }

            return make_float3(0);
        }
    };

    class CLighting
    {
    public:
        CLighting(void) :
            m_NoLights(0)
        {
        }

        CUB_CPU_GPU CLighting& operator=(const CLighting& Other)
        {
            for (int i = 0; i < MAX_NO_LIGHTS; i++)
            {
                m_Lights[i] = Other.m_Lights[i];
            }

            m_NoLights = Other.m_NoLights;

            return *this;
        }

        void AddLight(const CLight& Light)
        {
    // 		if (m_NoLights >= MAX_NO_LIGHTS)
    // 			return;

            m_Lights[m_NoLights] = Light;

            m_NoLights = m_NoLights + 1;
        }

        void Reset(void)
        {
            m_NoLights = 0;
            //memset(m_Lights, 0 , MAX_NO_LIGHTS * sizeof(CLight));
        }

        CLight		m_Lights[MAX_NO_LIGHTS];
        int			m_NoLights;
    };


    struct PBVolumeRenderKernelParams{
        CUDAVolumeParams cu_volume;
        CUDARenderParams cu_render_params;
        CUDAPerFrameParams cu_per_frame_params;
        CUDAPageTable cu_page_table;
        cudaTextureObject_t cu_vtex[MaxCUDATextureCountPerGPU];
        cudaTextureObject_t cu_tf_tex;
        cudaTextureObject_t cu_2d_tf_tex;
        CUDAFrameBuffer framebuffer;

        CUDABufferView1D<BlockUID> missed_blocks;


        CLighting m_Lighting;
        unsigned int* randomseed1;
        unsigned int* randomseed2;
        int	m_ShadingType;
        float m_GradientFactor;
    };


#define INVALID_VALUE uint4{0, 0, 0, 0}

    CUB_GPU uint32_t GetHashValue(const uint4& key){
        auto p = reinterpret_cast<const uint32_t*>(&key);
        uint32_t v = p[0];
        for(int i = 1; i < 4; i++){
            v = v ^ (p[i] + 0x9e3779b9 + (v << 6) + (v >> 2));
        }
        return v;
    }

    CUB_GPU uint4 Query(const uint4& key, uint4 hash_table[][2]){
        uint32_t hash_v = GetHashValue(key);
        auto pos = hash_v % HashTableSize;
        int i = 0;
        bool positive = false;
        while(i < HashTableSize){
            int ii = i * i;
            pos += positive ? ii : -ii;
            pos %= HashTableSize;
            if(hash_table[pos][0] == key){
                return hash_table[pos][1];
            }
            if(positive)
                ++i;
            positive = !positive;
        }
        return INVALID_VALUE;
    }

    CUB_GPU inline float3 CalcVirtualSamplingPos(const PBVolumeRenderKernelParams & params, const float3& pos){
        return (pos - params.cu_volume.bound.low)/(params.cu_volume.bound.high - params.cu_volume.bound.low);
    }

    CUB_GPU float CalcDistToNearestBlockCenter(const PBVolumeRenderKernelParams & params, const float3& pos){
        float3 offset_in_volume = CalcVirtualSamplingPos(params, pos);
        offset_in_volume *= params.cu_volume.voxel_dim;
        uint3 voxel_coord = make_uint3(offset_in_volume.x, offset_in_volume.y, offset_in_volume.z);
        voxel_coord /= params.cu_volume.block_length;
        voxel_coord = voxel_coord * params.cu_volume.block_length + params.cu_volume.block_length / 2;
        offset_in_volume =  make_float3(voxel_coord) / params.cu_volume.voxel_dim;
        float3 center_pos = offset_in_volume * (params.cu_volume.bound.high - params.cu_volume.bound.low) + params.cu_volume.bound.low;
        return length(params.cu_per_frame_params.cam_pos - center_pos);
    }

    CUB_GPU uint32_t ComputeLod(const PBVolumeRenderKernelParams & params, const float3& pos){
        float dist = CalcDistToNearestBlockCenter(params, pos);
        for(uint32_t i = 0; i < MaxLodLevels; ++i){
            if(dist < params.cu_render_params.lod_policy[i])
                return i;
        }
        return MaxLodLevels - 1;
    }

    CUB_GPU VirtualSamplingResult VirtualSampling(const PBVolumeRenderKernelParams & params,
                                                  uint4 hash_table[][2],
                                                  float3 offset_in_volume, uint32_t sampling_lod){
        //to voxel coord
        float3 sampling_coord =  offset_in_volume * params.cu_volume.voxel_dim;
        uint3 voxel_coord = make_uint3(sampling_coord.x, sampling_coord.y, sampling_coord.z);
        uint32_t lod_block_length = params.cu_volume.block_length << sampling_lod;
        uint3 block_uid = voxel_coord / lod_block_length;
        float3 offset_in_block = ((voxel_coord - block_uid * lod_block_length) + fracf(sampling_coord)) / float(1 << sampling_lod);
        uint4 key = make_uint4(block_uid, sampling_lod);
        uint4 tex_coord = Query(key, hash_table);
        uint32_t tid = (tex_coord.w >> 16) & 0xffff;
        //            printf("sampling tid : %d, %d %d %d %d\n", tid, tex_coord.x, tex_coord.y, tex_coord.z, tex_coord.w & 0xffff);
        uint3 coord = make_uint3(tex_coord.x, tex_coord.y, tex_coord.z);
        VirtualSamplingResult ret{0, 0};
        if((tex_coord.w & 0xffff) & TexCoordFlag_IsValid){
            //                printf("sampling tex\n");
            // valid
            float3 sampling_pos = (coord * params.cu_volume.block_size + offset_in_block + params.cu_volume.padding) * params.cu_render_params.inv_tex_shape;

            ret.scalar = tex3D<float>(params.cu_vtex[tid], sampling_pos.x, sampling_pos.y, sampling_pos.z);
            //                cudaSurfaceObject_t sur;
            //                int data;
            //                surf2Dread(&data, sur, 1, 1);
            //                printf("sampling pos %f %f %f, value %f",
            //                       sampling_pos.x, sampling_pos.y, sampling_pos.z,
            //                       ret.scalar);

            ret.flag = tex_coord.w & 0xffff;
        }
        else{
            if(params.cu_per_frame_params.debug_mode == 1)
                printf("block not find : %d %d %d %d, %d %d %d %d\n",key.x, key.y, key.z, key.w,
                       tex_coord.x, tex_coord.y, tex_coord.z, tex_coord.w);
        }
        return ret;
    }


    CUB_GPU float4 ScalarToRGBA(const PBVolumeRenderKernelParams & params,
                                float scalar, uint32_t lod){
        //            if(scalar > 0.3f){
        //                return make_float4(1.f,0.5f,0.25f,0.6f);
        //            }
        //            else return make_float4(0.f);
        if(scalar < 0.3f) return make_float4(0.f);
        if(params.cu_per_frame_params.debug_mode == 3){
            return make_float4(scalar, scalar, scalar, 1.f);
        }
        auto color = tex3D<float4>(params.cu_tf_tex, scalar, 0.5f, 0.5f);
        //            if(scalar >= 1.f){
        //                printf("scalar 0 to rgba %f %f %f %f\n", color.x, color.y, color.z, color.w);
        //            }
        return color;
    }

    CUB_GPU float3 CalcN(const PBVolumeRenderKernelParams & params,
                                    uint4 hash_table[][2],
                                    const float3& pos,
                                    const float3& ray_dir,
                                    const float3& dt, uint32_t lod){
        //todo use texGrad???
        float3 N;
        float x1, x2;
        int missed = 0;
        float lod_t = 1 << lod;
        auto ret = VirtualSampling(params, hash_table, CalcVirtualSamplingPos(params,pos + dt * make_float3(lod_t, 0.f, 0.f)), lod);
        x1 = ret.scalar;
        if(ret.flag == 0){
            ++missed;
        }
        ret = VirtualSampling(params, hash_table, CalcVirtualSamplingPos(params,pos + dt * make_float3(-lod_t, 0.f, 0.f)), lod);
        x2 = ret.scalar;
        if(ret.flag == 0){
            ++missed;
        }
        N.x = x1 - x2;

        ret = VirtualSampling(params, hash_table, CalcVirtualSamplingPos(params,pos + dt * make_float3(0.f, lod_t, 0.f)), lod);
        x1 = ret.scalar;
        if(ret.flag == 0){
            ++missed;
        }
        ret = VirtualSampling(params, hash_table, CalcVirtualSamplingPos(params,pos + dt * make_float3(0.f, -lod_t, 0.f)), lod);
        x2 = ret.scalar;
        if(ret.flag == 0){
            ++missed;
        }
        N.y = x1 - x2;

        ret = VirtualSampling(params, hash_table, CalcVirtualSamplingPos(params,pos + dt * make_float3(0.f, 0.f, lod_t)), lod);
        x1 = ret.scalar;
        if(ret.flag == 0){
            ++missed;
        }
        ret = VirtualSampling(params, hash_table, CalcVirtualSamplingPos(params,pos + dt * make_float3(0.f, 0.f, -lod_t)), lod);
        x2 = ret.scalar;
        if(ret.flag == 0){
            ++missed;
        }
        N.z = x1 - x2;
        if(missed == 6){
            N = -ray_dir;
        }
        else
            N = -normalize(N);

        return N;
    }


    CUB_GPU float GetNormalizedIntensity(const float3& P)
    {
        const float Intensity = ((float)SHRT_MAX * tex3D(gTexDensity, P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));

        return (Intensity - gIntensityMin) * gIntensityInvRange;
    }

    CUB_GPU float GetOpacity(const float& NormalizedIntensity)
    {
        return tex1D(gTexOpacity, NormalizedIntensity);
    }

    CUB_GPU float3 GetDiffuse(const float& NormalizedIntensity)
    {
        float4 Diffuse = tex1D(gTexDiffuse, NormalizedIntensity);
        return make_float3(Diffuse.x, Diffuse.y, Diffuse.z);
    }

    CUB_GPU float3 GetSpecular(const float& NormalizedIntensity)
    {
        float4 Specular = tex1D(gTexSpecular, NormalizedIntensity);
        return make_float3(Specular.x, Specular.y, Specular.z);
    }

    CUB_GPU float GetRoughness(const float& NormalizedIntensity)
    {
        return tex1D(gTexRoughness, NormalizedIntensity);
    }


//    CUB_GPU inline float3 NormalizedGradient(const float3& P)
//    {
//        float3 Gradient;
//
//        Gradient.x = (GetNormalizedIntensity(P + make_float3(gGradientDeltaX)) - GetNormalizedIntensity(P - make_float3(gGradientDeltaX))) * gInvGradientDelta;
//        Gradient.y = (GetNormalizedIntensity(P + make_float3(gGradientDeltaY)) - GetNormalizedIntensity(P - make_float3(gGradientDeltaY))) * gInvGradientDelta;
//        Gradient.z = (GetNormalizedIntensity(P + make_float3(gGradientDeltaZ)) - GetNormalizedIntensity(P - make_float3(gGradientDeltaZ))) * gInvGradientDelta;
//
//        return normalize(Gradient);
//    }

    CUB_GPU float GradientMagnitude(const float3& P)
    {
        return ((float)SHRT_MAX * tex3D(gTexGradientMagnitude, P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
    }




    CUB_GPU float3 GetEmission(const float& NormalizedIntensity)
    {
        float4 Emission = tex1D(gTexEmission, NormalizedIntensity);
        return make_float3(Emission.x, Emission.y, Emission.z);
    }

    CUB_GPU bool NearestLight(PBVolumeRenderKernelParams & params, RayExt R, float3& LightColor, float3& Pl, CLight*& pLight, float* pPdf = NULL)
    {
        bool Hit = false;

        float T = 0.0f;

        RayExt RayCopy = R;

        float Pdf = 0.0f;

        for (int i = 0; i < params.m_Lighting.m_NoLights; i++)
        {
            if (params.m_Lighting.m_Lights[i].Intersect(RayCopy, T, LightColor, NULL, &Pdf))
            {
                Pl		= R.o + normalize(R.d) * T;
                pLight	= &params.m_Lighting.m_Lights[i];
                Hit		= true;
            }
        }

        if (pPdf)
            *pPdf = Pdf;

        return Hit;
    }


    CUB_GPU uint32_t Float4ToUInt(const float4& f){
        uint32_t ret = (uint32_t)(saturate(f.x) * 255u)
                       | ((uint32_t)(saturate(f.y) * 255u) << 8)
                       | ((uint32_t)(saturate(f.z) * 255u) << 16)
                       | ((uint32_t)(saturate(f.w) * 255u) << 24);
        return ret;
    }
    CUB_GPU float3 PostProcessing(const float4& color){
        float3 ret;
        ret.x = pow(color.x, 1.f / 2.2f);
        ret.y = pow(color.y, 1.f / 2.2f);
        ret.z = pow(color.z, 1.f / 2.2f);
        return ret;
    }

    CUB_GPU inline float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
    {
        float f = nf * fPdf, g = ng * gPdf;
        return (f * f) / (f * f + g * g);
    }


    CUB_CPU_GPU inline float2 ConcentricSampleDisk(const float2& U)
    {
        float r, theta;
        // Map uniform random numbers to $[-1,1]^2$
        float sx = 2 * U.x - 1;
        float sy = 2 * U.y - 1;
        // Map square to $(r,\theta)$
        // Handle degeneracy at the origin

        if (sx == 0.0 && sy == 0.0)
        {
            return make_float2(0.0f);
        }

        if (sx >= -sy)
        {
            if (sx > sy)
            {
                // Handle first region of disk
                r = sx;
                if (sy > 0.0)
                    theta = sy/r;
                else
                    theta = 8.0f + sy/r;
            }
            else
            {
                // Handle second region of disk
                r = sy;
                theta = 2.0f - sx/r;
            }
        }
        else
        {
            if (sx <= sy)
            {
                // Handle third region of disk
                r = -sx;
                theta = 4.0f - sy/r;
            }
            else
            {
                // Handle fourth region of disk
                r = -sy;
                theta = 6.0f + sx/r;
            }
        }

        theta *= PI_F / 4.f;

        return make_float2(r*cosf(theta), r*sinf(theta));
    }

    CUB_CPU_GPU inline float3 CosineWeightedHemisphere(const float2& U)
    {
        const float2 ret = ConcentricSampleDisk(U);
        return make_float3(ret.x, ret.y, sqrtf(max(0.f, 1.f - ret.x * ret.x - ret.y * ret.y)));
    }

    CUB_CPU_GPU inline bool SameHemisphere(const float3& Ww1, const float3& Ww2)
    {
        return Ww1.z * Ww2.z > 0.0f;
    }

    CUB_CPU_GPU inline float AbsCosTheta(const float3 &Ws)
    {
        return fabsf(Ws.z);
    }


    CUB_CPU_GPU inline float3 SphericalDirection(const float& SinTheta, const float& CosTheta, const float& Phi)
    {
        return make_float3(SinTheta * cosf(Phi), SinTheta * sinf(Phi), CosTheta);
    }

    CUB_CPU_GPU inline float3 SphericalDirection(float sintheta, float costheta, float phi, const float3& x, const float3& y, const float3& z)
    {
        return sintheta * cosf(phi) * x + sintheta * sinf(phi) * y + costheta * z;
    }

    CUB_CPU_GPU inline float3 SphericalDirection(const float& SinTheta, const float& CosTheta, const float& Phi, const float3& N)
    {
        const float3 Wl = SphericalDirection(SinTheta, CosTheta, Phi);

        const float3 u = normalize(cross(N, make_float3(0.0072f, 1.0f, 0.0034f)));
        const float3 v = normalize(cross(N, u));

        return make_float3(	u.x * Wl.x + v.x * Wl.y + N.x * Wl.z,
                            u.y * Wl.x + v.y * Wl.y + N.y * Wl.z,
                            u.z * Wl.x + v.z * Wl.y + N.z * Wl.z);
    }

    CUB_CPU_GPU inline float3 Cross(const float3 &v1, const float3 &v2)
    {
        return make_float3((v1.y * v2.z) - (v1.z * v2.y), (v1.z * v2.x) - (v1.x * v2.z), (v1.x * v2.y) - (v1.y * v2.x));
    };






    class CLambertian
    {
    public:
        CUB_CPU_GPU CLambertian(const float3& Kd)
        {
            m_Kd = Kd;
        }

        CUB_CPU_GPU ~CLambertian(void)
        {
        }

        CUB_CPU_GPU float3 F(const float3& Wo, const float3& Wi)
        {
            return m_Kd * INV_PI_F;
        }

        CUB_CPU_GPU float3 SampleF(const float3& Wo, float3& Wi, float& Pdf, const float2& U)
        {
            Wi = CosineWeightedHemisphere(U);

            if (Wo.z < 0.0f)
                Wi.z *= -1.0f;

            Pdf = this->Pdf(Wo, Wi);

            return this->F(Wo, Wi);
        }

        CUB_CPU_GPU float Pdf(const float3& Wo, const float3& Wi)
        {
            return SameHemisphere(Wo, Wi) ? AbsCosTheta(Wi) * INV_PI_F : 0.0f;
        }

        float3	m_Kd;
    };

    CUB_CPU_GPU inline float3 FrDiel(float cosi, float cost, const float3 &etai, const float3 &etat)
    {
        float3 Rparl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float3 Rperp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        return (Rparl*Rparl + Rperp*Rperp) / 2.f;
    }

    class CFresnel
    {
    public:
        CUB_CPU_GPU CFresnel(float ei, float et) :
        eta_i(ei),
            eta_t(et)
        {
        }

        CUB_CPU_GPU  ~CFresnel(void)
        {
        }

        CUB_CPU_GPU float3 Evaluate(float cosi)
        {
            // Compute Fresnel reflectance for dielectric
            cosi = clamp(cosi, -1.0f, 1.0f);

            // Compute indices of refraction for dielectric
            bool entering = cosi > 0.0f;
            float ei = eta_i, et = eta_t;

            if (!entering)
                swap(ei, et);

            // Compute _sint_ using Snell's law
            float sint = ei/et * sqrtf(max(0.f, 1.f - cosi*cosi));

            if (sint >= 1.0f)
            {
                // Handle total internal reflection
                return make_float3(1.0f);
            }
            else
            {
                float cost = sqrtf(max(0.f, 1.0f - sint * sint));
                return FrDiel(fabsf(cosi), cost, make_float3(ei), make_float3(et));
            }
        }

        float eta_i, eta_t;
    };

    class CBlinn
    {
    public:
        CUB_CPU_GPU CBlinn(const float& Exponent) :
        m_Exponent(Exponent)
        {
        }

        CUB_CPU_GPU ~CBlinn(void)
        {
        }

        CUB_CPU_GPU void SampleF(const float3& Wo, float3& Wi, float& Pdf, const float2& U)
        {
            // Compute sampled half-angle vector $\wh$ for Blinn distribution
            float costheta = powf(U.x, 1.f / (m_Exponent+1));
            float sintheta = sqrtf(max(0.f, 1.f - costheta*costheta));
            float phi = U.y * 2.f * PI_F;

            float3 wh = SphericalDirection(sintheta, costheta, phi);

            if (!SameHemisphere(Wo, wh))
                wh = make_float3(0.0f) - wh;

            // Compute incident direction by reflecting about $\wh$
            Wi = -Wo + 2.f * dot(Wo, wh) * wh;

            // Compute PDF for $\wi$ from Blinn distribution
            float blinn_pdf = ((m_Exponent + 1.f) * powf(costheta, m_Exponent)) / (2.f * PI_F * 4.f * dot(Wo, wh));

            if (dot(Wo, wh) <= 0.f)
                blinn_pdf = 0.f;

            Pdf = blinn_pdf;
        }

        CUB_CPU_GPU float Pdf(const float3& Wo, const float3& Wi)
        {
            float3 wh = normalize(Wo + Wi);

            float costheta = AbsCosTheta(wh);
            // Compute PDF for $\wi$ from Blinn distribution
            float blinn_pdf = ((m_Exponent + 1.f) * powf(costheta, m_Exponent)) / (2.f * PI_F * 4.f * dot(Wo, wh));

            if (dot(Wo, wh) <= 0.0f)
                blinn_pdf = 0.0f;

            return blinn_pdf;
        }

        CUB_CPU_GPU float D(const float3& wh)
        {
            float costhetah = AbsCosTheta(wh);
            return (m_Exponent+2) * INV_TWO_PI_F * powf(costhetah, m_Exponent);
        }

        float	m_Exponent;
    };

    class CMicrofacet
    {
    public:
        CUB_CPU_GPU CMicrofacet(const float3& Reflectance, const float& Ior, const float& Exponent) :
        m_R(Reflectance),
            m_Fresnel(Ior, 1.0f),
            m_Blinn(Exponent)
        {
        }

        CUB_CPU_GPU ~CMicrofacet(void)
        {
        }

        CUB_CPU_GPU float3 F(const float3& wo, const float3& wi)
        {
            float cosThetaO = AbsCosTheta(wo);
            float cosThetaI = AbsCosTheta(wi);

            if (cosThetaI == 0.f || cosThetaO == 0.f)
                return make_float3(0);

            float3 wh = wi + wo;

            if (wh.x == 0. && wh.y == 0. && wh.z == 0.)
                return make_float3(0);

            wh = normalize(wh);
            float cosThetaH = dot(wi, wh);

            float3 F = make_float3(1.0f);//m_Fresnel.Evaluate(cosThetaH);

            return m_R * m_Blinn.D(wh) * G(wo, wi, wh) * F / (4.f * cosThetaI * cosThetaO);
        }

        CUB_CPU_GPU float3 SampleF(const float3& wo, float3& wi, float& Pdf, const float2& U)
        {
            m_Blinn.SampleF(wo, wi, Pdf, U);

            if (!SameHemisphere(wo, wi))
                return make_float3(0);

            return this->F(wo, wi);
        }

        CUB_CPU_GPU float Pdf(const float3& wo, const float3& wi)
        {
            if (!SameHemisphere(wo, wi))
                return 0.0f;

            return m_Blinn.Pdf(wo, wi);
        }

        CUB_CPU_GPU float G(const float3& wo, const float3& wi, const float3& wh)
        {
            float NdotWh = AbsCosTheta(wh);
            float NdotWo = AbsCosTheta(wo);
            float NdotWi = AbsCosTheta(wi);
            float WOdotWh = fabsf(dot(wo, wh));

            return min(1.f, min((2.f * NdotWh * NdotWo / WOdotWh), (2.f * NdotWh * NdotWi / WOdotWh)));
        }

        float3		m_R;
        CFresnel		m_Fresnel;
        CBlinn		m_Blinn;

    };

    class CIsotropicPhase
    {
    public:
        CUB_CPU_GPU CIsotropicPhase(const float3& Kd) :
            m_Kd(Kd)
        {
        }

        CUB_CPU_GPU ~CIsotropicPhase(void)
        {
        }

        CUB_CPU_GPU float3 F(const float3& Wo, const float3& Wi)
        {
            return m_Kd * INV_PI_F;
        }

        CUB_CPU_GPU float3 SampleF(const float3& Wo, float3& Wi, float& Pdf, const float2& U)
        {
            Wi	= UniformSampleSphere(U);
            Pdf	= this->Pdf(Wo, Wi);

            return F(Wo, Wi);
        }

        CUB_CPU_GPU float Pdf(const float3& Wo, const float3& Wi)
        {
            return INV_4_PI_F;
        }

        float3	m_Kd;
    };

    class CBRDF
    {
    public:
        CUB_CPU_GPU CBRDF(const float3& N, const float3& Wo, const float3& Kd, const float3& Ks, const float& Ior, const float& Exponent) :
            m_Lambertian(Kd),
            m_Microfacet(Ks, Ior, Exponent),
            m_Nn(N),
            m_Nu(normalize(Cross(N, Wo))),
            m_Nv(normalize(Cross(N, m_Nu)))
        {
        }

        CUB_CPU_GPU ~CBRDF(void)
        {
        }

        CUB_CPU_GPU float3 WorldToLocal(const float3& W)
        {
            return make_float3(dot(W, m_Nu), dot(W, m_Nv), dot(W, m_Nn));
        }

        CUB_CPU_GPU float3 LocalToWorld(const float3& W)
        {
            return make_float3(	m_Nu.x * W.x + m_Nv.x * W.y + m_Nn.x * W.z,
                            m_Nu.y * W.x + m_Nv.y * W.y + m_Nn.y * W.z,
                            m_Nu.z * W.x + m_Nv.z * W.y + m_Nn.z * W.z);
        }

        CUB_CPU_GPU float3 F(const float3& Wo, const float3& Wi)
        {
            const float3 Wol = WorldToLocal(Wo);
            const float3 Wil = WorldToLocal(Wi);

            float3 R;

            R += m_Lambertian.F(Wol, Wil);
            R += m_Microfacet.F(Wol, Wil);

            return R;
        }

        CUB_CPU_GPU float3 SampleF(const float3& Wo, float3& Wi, float& Pdf, const CBrdfSample& S)
        {
            const float3 Wol = WorldToLocal(Wo);
            float3 Wil;

            float3 R;

            if (S.m_Component <= 0.5f)
            {
                m_Lambertian.SampleF(Wol, Wil, Pdf, S.m_Dir);
            }
            else
            {
                m_Microfacet.SampleF(Wol, Wil, Pdf, S.m_Dir);
            }

            Pdf += m_Lambertian.Pdf(Wol, Wil);
            Pdf += m_Microfacet.Pdf(Wol, Wil);

            R += m_Lambertian.F(Wol, Wil);
            R += m_Microfacet.F(Wol, Wil);

            Wi = LocalToWorld(Wil);

            return R;
        }

        CUB_CPU_GPU float Pdf(const float3& Wo, const float3& Wi)
        {
            const float3 Wol = WorldToLocal(Wo);
            const float3 Wil = WorldToLocal(Wi);

            float Pdf = 0.0f;

            Pdf += m_Lambertian.Pdf(Wol, Wil);
            Pdf += m_Microfacet.Pdf(Wol, Wil);

            return Pdf;
        }

        float3			m_Nn;
        float3			m_Nu;
        float3			m_Nv;
        CLambertian		m_Lambertian;
        CMicrofacet		m_Microfacet;
    };

    class CVolumeShader
    {
    public:
        enum EType
        {
            Brdf,
            Phase
        };

        CUB_CPU_GPU CVolumeShader(const EType& Type, const float3& N, const float3& Wo, const float3& Kd, const float3& Ks, const float& Ior, const float& Exponent) :
            m_Type(Type),
            m_Brdf(N, Wo, Kd, Ks, Ior, Exponent),
            m_IsotropicPhase(Kd)
        {
        }

        CUB_CPU_GPU ~CVolumeShader(void)
        {
        }

        CUB_CPU_GPU float3 F(const float3& Wo, const float3& Wi)
        {
            switch (m_Type)
            {
                case Brdf:
                    return m_Brdf.F(Wo, Wi);

                case Phase:
                    return m_IsotropicPhase.F(Wo, Wi);
            }

            return make_float3(1.0f);
        }

        CUB_CPU_GPU float3 SampleF(const float3& Wo, float3& Wi, float& Pdf, const CBrdfSample& S)
        {
            switch (m_Type)
            {
                case Brdf:
                    return m_Brdf.SampleF(Wo, Wi, Pdf, S);

                case Phase:
                    return m_IsotropicPhase.SampleF(Wo, Wi, Pdf, S.m_Dir);
            }
        }

        CUB_CPU_GPU float Pdf(const float3& Wo, const float3& Wi)
        {
            switch (m_Type)
            {
                case Brdf:
                    return m_Brdf.Pdf(Wo, Wi);

                case Phase:
                    return m_IsotropicPhase.Pdf(Wo, Wi);
            }

            return 1.0f;
        }

        EType				m_Type;
        CBRDF				m_Brdf;
        CIsotropicPhase		m_IsotropicPhase;
    };





    CUB_GPU bool IntersectBox(const RayExt& R, float* pNearT, float* pFarT)
    {
        const float3 InvR		= make_float3(1.0f, 1.0f, 1.0f) / R.d;
        const float3 BottomT	= InvR * (make_float3(gAaBbMin.x, gAaBbMin.y, gAaBbMin.z) - R.o);
        const float3 TopT		= InvR * (make_float3(gAaBbMax.x, gAaBbMax.y, gAaBbMax.z) - R.o);
        const float3 MinT		= fminf(TopT, BottomT);
        const float3 MaxT		= fmaxf(TopT, BottomT);
        const float LargestMinT = fmaxf(fmaxf(MinT.x, MinT.y), fmaxf(MinT.x, MinT.z));
        const float LargestMaxT = fminf(fminf(MaxT.x, MaxT.y), fminf(MaxT.x, MaxT.z));

        *pNearT = LargestMinT;
        *pFarT	= LargestMaxT;

        return LargestMaxT > LargestMinT;
    }

    CUB_GPU inline bool SampleDistanceRM(RayExt& R, CRNG& RNG, float3& Ps)
    {
        const int TID = threadIdx.y * blockDim.x + threadIdx.x;

        __shared__ float MinT[KRNL_SS_BLOCK_SIZE];
        __shared__ float MaxT[KRNL_SS_BLOCK_SIZE];

        if (!IntersectBox(R, &MinT[TID], &MaxT[TID]))
            return false;

        MinT[TID] = max(MinT[TID], R.min_t);
        MaxT[TID] = min(MaxT[TID], R.max_t);

        const float S = -log(RNG.Get1()) / gDensityScale;
        float Sum = 0.0f;
        float SigmaT = 0.0f;

        MinT[TID] += RNG.Get1() * gStepSize;

        // delta-tracking
        while (Sum < S)
        {
            Ps = R.o + MinT[TID] * R.d;

            if (MinT[TID] > MaxT[TID])
                return false;

            SigmaT	= gDensityScale * GetOpacity(GetNormalizedIntensity(Ps));

            Sum			+= SigmaT * gStepSize;
            MinT[TID]	+= gStepSize;
        }

        return true;
    }


    CUB_GPU inline bool FreePathRM(RayExt& R, CRNG& RNG)
    {
        const int TID = threadIdx.y * blockDim.x + threadIdx.x;

        __shared__ float MinT[KRNL_SS_BLOCK_SIZE];
        __shared__ float MaxT[KRNL_SS_BLOCK_SIZE];
        __shared__ float3 Ps[KRNL_SS_BLOCK_SIZE];

        if (!IntersectBox(R, &MinT[TID], &MaxT[TID]))
            return false;

        MinT[TID] = max(MinT[TID], R.min_t);
        MaxT[TID] = min(MaxT[TID], R.max_t);

        const float S	= -log(RNG.Get1()) / gDensityScale;
        float Sum		= 0.0f;
        float SigmaT	= 0.0f;

        MinT[TID] += RNG.Get1() * gStepSizeShadow;

        // delta-tracking
        while (Sum < S)
        {
            Ps[TID] = R.o + MinT[TID] * R.d;

            if (MinT[TID] > MaxT[TID])
                return false;

            SigmaT	= gDensityScale * GetOpacity(GetNormalizedIntensity(Ps[TID]));

            Sum			+= SigmaT * gStepSizeShadow;
            MinT[TID]	+= gStepSizeShadow;
        }

        return true;
    }


    CUB_GPU inline bool NearestIntersection(RayExt R, PBVolumeRenderKernelParams params, float& T)
    {
        float MinT = 0.0f, MaxT = 0.0f;

        if (!IntersectBox(R, &MinT, &MaxT))
            return false;

        MinT = max(MinT, R.min_t);
        MaxT = min(MaxT, R.max_t);

        float3 Ps;

        T = MinT;

        while (T < MaxT)
        {
            Ps = R.o + T * R.d;

            if (GetOpacity(GetNormalizedIntensity(Ps)) > 0.0f)
                return true;

            T += gStepSize;
        }

        return false;
    }



    CUB_GPU float3 EstimateDirectLight(PBVolumeRenderKernelParams & params, const CVolumeShader::EType& Type, const float& Density, CLight& Light, CLightingSample& LS, const float3& Wo, const float3& Pe, const float3& N, CRNG& RNG)
    {
        float3 Ld = make_float3(0), Li = make_float3(0), F = make_float3(0);
        float3 tdiffuse = GetDiffuse(Density);
        float3 tspecular = GetSpecular(Density);

        CVolumeShader Shader(Type, N, Wo, FromRGB(tdiffuse.x, tdiffuse.y, tdiffuse.z), FromRGB(tspecular.x, tspecular.y, tspecular.z), 2.5f/*params.m_IOR*/, GetRoughness(Density));

        RayExt Rl;

        float LightPdf = 1.0f, ShaderPdf = 1.0f;

        float3 Wi, P, Pl;

        Li = Light.SampleL(Pe, Rl, LightPdf, LS);

        CLight* pLight = NULL;

        Wi = make_float3(0.0f) - Rl.d;

        F = Shader.F(Wo, Wi);

        ShaderPdf = Shader.Pdf(Wo, Wi);

        //todo: FreePathRM calc tr
        if (!IsBlack(Li) && ShaderPdf > 0.0f && LightPdf > 0.0f && !FreePathRM(Rl, RNG))
        {
            const float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, ShaderPdf);

            if (Type == CVolumeShader::Brdf)
                Ld += F * Li * fabsf(dot(Wi, N)) * WeightMIS / LightPdf;

            if (Type == CVolumeShader::Phase)
                Ld += F * Li * WeightMIS / LightPdf;
        }

        F = Shader.SampleF(Wo, Wi, ShaderPdf, LS.m_BsdfSample);

        if (!IsBlack(F) && ShaderPdf > 0.0f)
        {
            if (NearestLight(params, RayExt(Pe, Wi, 0.0f, FLT_MAX), Li, Pl, pLight, &LightPdf))
            {
                LightPdf = pLight->Pdf(Pe, Wi);
                RayExt tpray = RayExt(Pl, normalize(Pe - Pl), 0.0f, length(Pe - Pl));
                if (LightPdf > 0.0f && !IsBlack(Li) && !FreePathRM(tpray, RNG))
                {
                    const float WeightMIS = PowerHeuristic(1.0f, ShaderPdf, 1.0f, LightPdf);

                    if (Type == CVolumeShader::Brdf)
                        Ld += F * Li * fabsf(dot(Wi, N)) * WeightMIS / ShaderPdf;

                    if (Type == CVolumeShader::Phase)
                        Ld += F * Li * WeightMIS / ShaderPdf;
                }
            }
        }

        return Ld;
    }

    CUB_GPU float3 UniformSampleOneLight(PBVolumeRenderKernelParams & params, const CVolumeShader::EType& Type, const float& Density, const float3& Wo, const float3& Pe, const float3& N, CRNG& RNG, const bool& Brdf)
    {
        const int NumLights = params.m_Lighting.m_NoLights;

        if (NumLights == 0)
            return make_float3(0);

        CLightingSample LS;

        LS.LargeStep(RNG);

        const int WhichLight = (int)floorf(LS.m_LightNum * (float)NumLights);

        CLight& Light = params.m_Lighting.m_Lights[WhichLight];

        return (float)NumLights * EstimateDirectLight(params, Type, Density, Light, LS, Wo, Pe, N, RNG);
    }


    CUB_KERNEL void PBVolumeRenderKernel(PBVolumeRenderKernelParams params){
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(x >= params.cu_per_frame_params.frame_width || y >= params.cu_per_frame_params.frame_height)
            return;


        const unsigned int thread_count = blockDim.x * blockDim.y;
        const unsigned int thread_idx = threadIdx.x + blockDim.x * threadIdx.y;
        const unsigned int load_count = (HashTableSize + thread_count - 1) / thread_count;
        const unsigned int thread_beg = thread_idx * load_count;
        const unsigned int thread_end = min(thread_beg + load_count, HashTableSize);
        __shared__ uint4 hash_table[HashTableSize][2];

        auto& table = params.cu_page_table.table;
        for(int i = 0; i < HashTableSize; ++i){
            auto& block_uid = table.at(i).first;
            auto& tex_coord = table.at(i).second;
            hash_table[i][0] = uint4{block_uid.x, block_uid.y, block_uid.z, block_uid.w};
            hash_table[i][1] = uint4{tex_coord.sx, tex_coord.sy, tex_coord.sz,
                                     ((uint32_t)tex_coord.tid << 16) | tex_coord.flag};
        }

        __syncthreads();




        CRNG RNG(&(params.randomseed1[y * params.cu_per_frame_params.frame_width + x]), &(params.randomseed2[y * params.cu_per_frame_params.frame_width + x]));



        float3 Lv = make_float3(0), Li = make_float3(0);



        const float2 UV = make_float2(x, y) + RNG.Get2();


        RayExt ray;
        ray.o = params.cu_per_frame_params.cam_pos;
        float scale = tanf(0.5f * params.cu_per_frame_params.fov);
        float ix = (UV.x + params.cu_render_params.mpi_node_offset.x + 0.5f) / params.cu_per_frame_params.frame_width - 0.5f;
        float iy = (UV.y + params.cu_render_params.mpi_node_offset.y + 0.5f) / params.cu_per_frame_params.frame_height - 0.5f;





        ray.d = params.cu_per_frame_params.cam_dir + params.cu_per_frame_params.cam_up * scale * iy
                + params.cu_per_frame_params.cam_right * scale * ix * params.cu_per_frame_params.frame_w_over_h;


        ray.min_t = 0.0f;
	    ray.max_t = 1500000.0f;

        float3 pe, pl;

        CLight* pLight = NULL;

        if (SampleDistanceRM(ray, RNG, pe))
        {
            if (NearestLight(params, RayExt(ray.o, ray.d, 0.0f, length(pe - ray.o)), Li, pl, pLight))
            {

                // pView->m_FrameEstimateXyza.Set(float3a(Lv.c[0], Lv.c[1], Lv.c[2]), X, Y);
			    // return;
                // auto [color, depth] = RayCastVolume(params, hash_table, ray);

                float4 color = make_float4(Lv, 0);
                color = make_float4(PostProcessing(color), color.w);
                // exposure gamma color-grading tone-mapping...
                //            y = params.cu_per_frame_params.frame_height - 1 - y;
                float depth = 0; // ?
                params.framebuffer.color.at(x, y) = Float4ToUInt(color);
                params.framebuffer.depth.at(x, y) = depth;
                return;
            }

            uint32_t cur_lod = 0;
            float3 sampling_pos = CalcVirtualSamplingPos(params, pe);

            auto [flag, scalar] = VirtualSampling(params, hash_table, sampling_pos, cur_lod);


            float4 mapping_color = ScalarToRGBA(params, scalar, cur_lod);


            float3 gN = CalcN(params, hash_table, pe, ray.d, params.cu_volume.voxel_space, cur_lod);



            const float D = scalar;

            float3 ge = GetEmission(D);
            Lv += FromRGB(ge.x, ge.y, ge.z); // todo


            switch (params.m_ShadingType)
            {
                case 0:
                {
                    Lv += UniformSampleOneLight(params, CVolumeShader::Brdf, D, normalize(make_float3(0)-ray.d), pe, gN, RNG, true);
                    break;
                }

                case 1:
                {
                    Lv += 0.5f * UniformSampleOneLight(params, CVolumeShader::Phase, D, normalize(make_float3(0)-ray.d), pe, gN, RNG, false);
                    break;
                }

                case 2:
                {
                    const float GradMag = GradientMagnitude(pe) * gIntensityInvRange;

                    const float PdfBrdf = (1.0f - __expf(-params.m_GradientFactor * GradMag));

                    if (RNG.Get1() < PdfBrdf)
                        Lv += UniformSampleOneLight(params, CVolumeShader::Brdf, D, normalize(make_float3(0)-ray.d), pe, gN, RNG, true);
                    else
                        Lv += 0.5f * UniformSampleOneLight(params, CVolumeShader::Phase, D, normalize(make_float3(0)-ray.d), pe, gN, RNG, false);

                    break;
                }
            }
        }
        else
        {
            if (NearestLight(params, RayExt(ray.o, ray.d, 0.0f, INF_MAX), Li, pl, pLight))
                Lv = Li;
        }


        float4 color = make_float4(PostProcessing(make_float4(Lv, 0)));

        params.framebuffer.color.at(x, y) = Float4ToUInt(color);
        params.framebuffer.depth.at(x, y) = 0;

    }

}





class PBVolumeRendererPrivate{
  public:
    struct{
        CUDAContext ctx;
        CUDAStream stream;
        PBVolumeRenderKernelParams kernel_params;
    };



    struct{
        Handle<CUDAHostBuffer> tf1d;
        Handle<CUDAHostBuffer> tf2d;
        Handle<CUDATexture> cu_tf_tex;
        Handle<CUDATexture> cu_2d_tf_tex;
        int tf_dim = 0;
    };

    void GenTFTex(int dim){
        if(tf_dim == dim) return;
        tf_dim = dim;
        tf1d.Destroy();
        tf2d.Destroy();
        cu_tf_tex.Destroy();
        cu_2d_tf_tex.Destroy();

        size_t bytes_1d = sizeof(typename TransferFunc::Value) * dim;
        tf1d = NewHandle<CUDAHostBuffer>(ResourceType::Buffer, bytes_1d, cub::e_cu_host, ctx);

        size_t bytes_2d = sizeof(typename TransferFunc::Value) * dim * dim;
        tf2d = NewHandle<CUDAHostBuffer>(ResourceType::Buffer, bytes_2d, cub::e_cu_host, ctx);

        cub::texture_resc_info resc_info{cub::e_float, 4, {256, 1, 1}};
        cub::texture_view_info view_info; view_info.read = cub::e_raw;
        cu_tf_tex = NewHandle<CUDATexture>(ResourceType::Buffer, resc_info, view_info, ctx);

        resc_info.extent = {256, 256, 1};
        cu_2d_tf_tex = NewHandle<CUDATexture>(ResourceType::Buffer, resc_info, view_info, ctx);
    }

    struct{
        Ref<HostMemMgr> host_mem_mgr_ref;
        Ref<GPUMemMgr> gpu_mem_mgr_ref;
        Ref<GPUVTexMgr> gpu_vtex_mgr_ref;
        Ref<GPUPageTableMgr> gpu_pt_mgr_ref;
        Ref<FixedHostMemMgr> fixed_host_mem_mgr_ref;

        Handle<GridVolume> volume;
    };

    struct{
        float render_space_ratio;
        bool use_shared_host_mem = false;
        size_t fixed_host_mem_bytes;
        size_t vtex_cnt;
        Int3 vtex_shape;
    };

    // data sts for loading blocks
    struct{
        std::vector<BlockUID> intersect_blocks;
        std::vector<GPUPageTableMgr::PageTableItem> block_infos;

    };


    struct{
        Float3 lod0_block_length_space;
        UInt3 lod0_block_dim;
        BoundingBox3D volume_bound;
        int max_lod;
        Mat4 camera_proj_view;
        Float3 camera_pos;
        LevelOfDist lod;
    };

    UnifiedRescUID uid;

    std::mutex g_mtx;

    static UnifiedRescUID GenRescUID(){
        static std::atomic<size_t> g_uid = 1;
        auto uid = g_uid.fetch_add(1);
        return GenUnifiedRescUID(uid, UnifiedRescType::PBVolRenderer);
    }
};

PBVolumeRenderer::PBVolumeRenderer(const PBVolumeRendererCreateInfo &info)
{
    _ = std::make_unique<PBVolumeRendererPrivate>();


}

PBVolumeRenderer::~PBVolumeRenderer()
{

}

void PBVolumeRenderer::Lock()
{
    _->g_mtx.lock();
}

void PBVolumeRenderer::UnLock()
{
    _->g_mtx.unlock();
}

UnifiedRescUID PBVolumeRenderer::GetUID() const
{
    return _->uid;
}

void PBVolumeRenderer::BindGridVolume(Handle<GridVolume> volume)
{
    // check if re-create fixed_host_mem_mgr and gpu_vtex_mgr
    auto should_recreate_mem_mgr = [&]()->std::pair<bool, bool>{
        std::pair<bool,bool> ret = {false, false};
        if(!_->gpu_vtex_mgr_ref.IsValid())
            ret.second = true;
        if(_->volume.IsValid()){
            auto desc0 = _->volume->GetDesc();
            auto desc1 = volume->GetDesc();
            if (desc0.block_length + desc0.padding != desc1.block_length + desc1.padding)
                ret = {true, true};
        }
        else ret = {true, true};
        if(_->use_shared_host_mem) ret.first = false;
        else if(!_->fixed_host_mem_mgr_ref.IsValid()) ret.first = true;
        return ret;
    };
    auto [should_host, should_gpu] = should_recreate_mem_mgr();
    auto desc = volume->GetDesc();
    {
        size_t block_size = (desc.block_length + desc.padding * 2) * desc.bits_per_sample *
                            desc.samples_per_voxel / 8;
        block_size *= block_size * block_size;
        size_t block_num = _->fixed_host_mem_bytes / block_size;

        if(should_host){
            FixedHostMemMgr::FixedHostMemMgrCreateInfo fixed_info;
            fixed_info.host_mem_mgr = _->host_mem_mgr_ref;
            fixed_info.fixed_block_size = block_size;
            fixed_info.fixed_block_num = block_num;

            auto fixed_host_mem_uid = _->host_mem_mgr_ref->RegisterFixedHostMemMgr(fixed_info);
            _->fixed_host_mem_mgr_ref = _->host_mem_mgr_ref->GetFixedHostMemMgrRef(fixed_host_mem_uid);
        }

        if(should_gpu){
            GPUVTexMgr::GPUVTexMgrCreateInfo vtex_info;
            vtex_info.gpu_mem_mgr = _->gpu_mem_mgr_ref;
            vtex_info.host_mem_mgr = _->host_mem_mgr_ref;
            vtex_info.vtex_count = _->vtex_cnt;
            vtex_info.vtex_shape = _->vtex_shape;
            vtex_info.bits_per_sample = desc.bits_per_sample;
            vtex_info.samples_per_channel = desc.samples_per_voxel;
            vtex_info.vtex_block_length = (desc.block_length + desc.padding * 2);
            vtex_info.is_float = desc.is_float;
            vtex_info.exclusive = true;

            auto vtex_uid = _->gpu_mem_mgr_ref->RegisterGPUVTexMgr(vtex_info);
            _->gpu_vtex_mgr_ref = _->gpu_mem_mgr_ref->GetGPUVTexMgrRef(vtex_uid);
            _->gpu_pt_mgr_ref = _->gpu_vtex_mgr_ref->GetGPUPageTableMgrRef();
        }
    }


    {
        //bind vtex
        auto texes = _->gpu_vtex_mgr_ref->GetAllTextures();
        for(auto [unit, tex] : texes){
            _->kernel_params.cu_vtex[unit] = tex->_get_tex_handle();
        }

        //bind volume params
        _->kernel_params.cu_volume.block_length = desc.block_length;
        _->kernel_params.cu_volume.padding = desc.padding;
        _->kernel_params.cu_volume.voxel_dim = {(float)desc.shape.x,
                                                (float)desc.shape.y,
                                                (float)desc.shape.z};
        _->kernel_params.cu_volume.voxel_space = {_->render_space_ratio * desc.voxel_space.x,
                                                  _->render_space_ratio * desc.voxel_space.y,
                                                  _->render_space_ratio * desc.voxel_space.z};
        _->kernel_params.cu_volume.bound = {
            {0.f, 0.f, 0.f},
            _->kernel_params.cu_volume.voxel_space * _->kernel_params.cu_volume.voxel_dim};
    }
}

void PBVolumeRenderer::SetRenderParams(const RenderParams &render_params)
{

}

void PBVolumeRenderer::SetPerFrameParams(const PerFrameParams &per_frame_params)
{
    _->kernel_params.cu_per_frame_params.cam_pos = { Transform(per_frame_params.cam_pos) };
    _->kernel_params.cu_per_frame_params.fov = per_frame_params.fov;
    _->kernel_params.cu_per_frame_params.cam_dir = { Transform(per_frame_params.cam_dir) };
    _->kernel_params.cu_per_frame_params.frame_width = per_frame_params.frame_width;
    _->kernel_params.cu_per_frame_params.cam_right = { Transform(per_frame_params.cam_right) };
    _->kernel_params.cu_per_frame_params.frame_height = per_frame_params.frame_height;
    _->kernel_params.cu_per_frame_params.cam_up = { Transform(per_frame_params.cam_up) };
    _->kernel_params.cu_per_frame_params.frame_w_over_h = per_frame_params.frame_w_over_h;
//    _->kernel_params.cu_per_frame_params.debug_mode = per_frame_params.debug_mode;
}

void PBVolumeRenderer::Render(Handle<FrameBuffer> frame)
{

    bool render_frame_completed = false;

    auto load_resource = [&](){

    };

    auto render_pass = [&](){

    };


    try
    {
        while (render_frame_completed)
        {

            load_resource();

            render_pass();

        }
    }
    catch(const std::exception err){
        LOG_ERROR("PBVolumeRenderer::Render error for complete frame: {}", err.what());
    }
}

VISER_END