#include "lightTransport.h"

#ifndef OPTICAL_DEPTH_LUT_DIM
#   define OPTICAL_DEPTH_LUT_DIM float4(64,32,64,32)
#endif

#ifndef NUM_PARTICLE_LAYERS
#   define NUM_PARTICLE_LAYERS 1
#endif

#ifndef SRF_SCATTERING_IN_PARTICLE_LUT_DIM
#   define SRF_SCATTERING_IN_PARTICLE_LUT_DIM float3(32,64,16)
#endif

#ifndef VOL_SCATTERING_IN_PARTICLE_LUT_DIM
#   define VOL_SCATTERING_IN_PARTICLE_LUT_DIM float4(32,64,32,8)
#endif

#ifndef THREAD_GROUP_SIZE
#   define THREAD_GROUP_SIZE 64
#endif

#define NUM_INTEGRATION_STEPS 64

//Global scope surface to bind to
surface<void, cudaSurfaceType3D> surfaceOpticalDepthWrite;
surface<void, cudaSurfaceType3D>   surfaceScatteringWrite;

/*
 *  GENERAL: Device functions
 *  =======  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
 __device__ void spherical2cartesian(){

}

// Computes direction from the zenith and azimuth angles in XZY (Y Up) coordinate system
__device__ float3 ZenithAzimuthAngleToDirectionXZY(float zenithAngle, float azimuthAngle){
    //       Y   Zenith
    //       |  /
    //       | / /'
    //       |  / '
    //       | /  '
    //       |/___'________X
    //      / \  -Azimuth
    //     /   \  '
    //    /     \ '
    //   Z       \'

    float zenithSin, zenithCos, azimuthSin, azimuthCos;
    sincos( zenithAngle,  zenithSin,  zenithCos);
    sincos(azimuthAngle, azimuthSin, azimuthCos);

    float3 direction;
    direction.y = zenithCos;
    direction.x = zenithSin * azimuthCos;
    direction.z = zenithSin * azimuthSin;
    
    return direction;
}

// Computes the zenith and azimuth angles in XZY (Y Up) coordinate system from direction
__device__ void DirectionToZenithAzimuthAngleXZY(float3 &direction, float &zenithAngle, float &azimuthAngle){
    float zenithCos = direction.y;
    zenithAngle = acos(zenithCos);
    //float fZenithSin = sqrt( max(1 - zenithCos*zenithCos, 1e-10) );

    float azimuthCos = direction.x;// / fZenithSin;
    float azimuthSin = direction.z;// / fZenithSin;

    azimuthAngle = atan2(azimuthSin, azimuthCos);
}

// Constructs local XYZ (Z Up) frame from Up and Inward vectors
__device__ void ConstructLocalFrameXYZ(float3 &up, float3 &inward, 
                                       float3 & X, float3 & Y, float3 & Z){
    //      Z (Up)
    //      |    Y  (Inward)
    //      |   /
    //      |  /
    //      | /  
    //      |/
    //       -----------> X
    //
    Z = normalize(up);
    X = normalize(cross(inward, Z));
    Y = normalize(cross(Z, X));
}

// Computes direction in local XYZ (Z Up) frame from zenith and azimuth angles
__device__ float3 GetDirectionInLocalFrameXYZ(float3 localX, 
                                              float3 localY, 
                                              float3 localZ,
                                              float  localZenithAngle,
                                              float  localAzimuthAngle){
    // Compute sin and cos of the angle between ray direction and local zenith
    float dirLocalSinZenithAngle, dirLocalCosZenithAngle;
    sincos(localZenithAngle, dirLocalSinZenithAngle, dirLocalCosZenithAngle);

    // Compute sin and cos of the local azimuth angle
    float dirLocalAzimuthCos, dirLocalAzimuthSin;
    sincos(localAzimuthAngle, dirLocalAzimuthSin, dirLocalAzimuthCos);
    
    // Reconstruct view ray
    return localZ * dirLocalCosZenithAngle + 
           dirLocalSinZenithAngle * (dirLocalAzimuthCos * localX + dirLocalAzimuthSin * localY );
}

// Computes zenith and azimuth angles in local XYZ (Z Up) frame from the direction
__device__ void ComputeLocalFrameAnglesXYZ( float3 &localX, 
                                            float3 &localY, 
                                            float3 &localZ,
                                            float3 &rayDir,
                                            float  &localZenithAngle,
                                            float  &localAzimuthAngle){
    localZenithAngle = acos(saturate( dot(localZ,rayDir) ));

    // Compute azimuth angle in the local frame
    float viewDirLocalAzimuthCos = dot(rayDir,localX);
    float viewDirLocalAzimuthSin = dot(rayDir,localY);

    localAzimuthAngle = atan2(viewDirLocalAzimuthSin,viewDirLocalAzimuthCos);
}

// Get Ray Sphere Intersection
// http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
__device__ void GetRaySphereIntersection(float3 &rayOrigin,    float3 &rayDirection, 
                                         float3 &sphereCenter, float  &sphereRadius
                                         float2 &intersections){
   rayOrigin -= sphereCenter;
   float A = dot(rayDirection, &rayDirection);
   float B = 2 * dot(&rayOrigin, &rayDirection);
   float C = dot(rayOrigin, rayOrigin) - sphereRadius*sphereRadius;
   float D = B*B - 4*A*C;

   // If discriminant is negative, there are no real roots hence the ray misses the
   // sphere
   if( D<0 ){
       intersections = float2(-1,-1);
   }
   else{
       D = sqrt(D);
       intersections = float2(-B - D, -B + D) / (2*A); // A must be positive here!!
   }
}


__device__ float HGPhaseFunc(float cosTheta, const float g = 0.9){
    return (1/(4*PI) * (1 - g*g)) / pow( max((1 + g*g) - (2*g)*cosTheta,0), 3.f/2.f);
}



/*
 *  OPTICAL DEPTH: Device functions
 *  =============  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
 
    /*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/

 #define SAMPLE_4D_LUT(tex3DLUT, LUT_DIM, f4LUTCoords, fLOD, Result)  \
 {                                                               \
     float3 f3UVW;                                               \
     f3UVW.xy = f4LUTCoords.xy;                                  \
     float fQSlice = f4LUTCoords.w * LUT_DIM.w - 0.5;            \
     float fQ0Slice = floor(fQSlice);                            \
     float fQWeight = fQSlice - fQ0Slice;                        \
                                                                 \
     f3UVW.z = (fQ0Slice + f4LUTCoords.z) / LUT_DIM.w;           \
                                                                 \
     Result = lerp(                                              \
         tex3DLUT.SampleLevel(samLinearWrap, f3UVW, fLOD),       \
         /* frac() assures wraparound filtering of w coordinate*/                            \
         tex3DLUT.SampleLevel(samLinearWrap, frac(f3UVW + float3(0,0,1/LUT_DIM.w)), fLOD),   \
         fQWeight);                                                                          \
 }

 void ParticleScatteringLUTToWorldParams(in float4 f4LUTCoords, 
                                        out float3 f3StartPosUSSpace,
                                        out float3 f3ViewDirUSSpace,
                                        out float3 f3LightDirUSSpace,
                                        in uniform bool bSurfaceOnly){
    f3LightDirUSSpace = float3(0,0,1);
    float fStartPosZenithAngle = f4LUTCoords.x * PI;
    f3StartPosUSSpace = float3(0,0,0);
    sincos(fStartPosZenithAngle, f3StartPosUSSpace.x, f3StartPosUSSpace.z);

    float3 f3LocalX, f3LocalY, f3LocalZ;
    ConstructLocalFrameXYZ(-f3StartPosUSSpace, f3LightDirUSSpace, f3LocalX, f3LocalY, f3LocalZ);

    if( !bSurfaceOnly )
    {
    float fDistFromCenter = f4LUTCoords.w;
    // Scale the start position according to the distance from center
    f3StartPosUSSpace *= fDistFromCenter;
    }

    float fViewDirLocalAzimuth = (f4LUTCoords.y - 0.5) * (2 * PI); 
    float fViewDirLocalZenith = f4LUTCoords.z * ( bSurfaceOnly ? (PI/2) : PI );
    f3ViewDirUSSpace = GetDirectionInLocalFrameXYZ(f3LocalX, f3LocalY, f3LocalZ, fViewDirLocalZenith, fViewDirLocalAzimuth);
}

// All parameters must be defined in the unit sphere (US) space
float4 WorldParamsToParticleScatteringLUT(  in float3 f3StartPosUSSpace, 
                                            in float3 f3ViewDirInUSSpace, 
                                            in float3 f3LightDirInUSSpace,
                                            in uniform bool bSurfaceOnly){
                                            float4 f4LUTCoords = 0;

    float fDistFromCenter = 0;
    if( !bSurfaceOnly ){
        // Compute distance from center and normalize start position
        fDistFromCenter = length(f3StartPosUSSpace);
        f3StartPosUSSpace /= max(fDistFromCenter, 1e-5);
    }
    float fStartPosZenithCos = dot(f3StartPosUSSpace, f3LightDirInUSSpace);
    f4LUTCoords.x = acos(fStartPosZenithCos);

    float3 f3LocalX, f3LocalY, f3LocalZ;
    ConstructLocalFrameXYZ(-f3StartPosUSSpace, f3LightDirInUSSpace, f3LocalX, f3LocalY, f3LocalZ);

    float fViewDirLocalZenith, fViewDirLocalAzimuth;
    ComputeLocalFrameAnglesXYZ(f3LocalX, f3LocalY, f3LocalZ, f3ViewDirInUSSpace, fViewDirLocalZenith, fViewDirLocalAzimuth);
    f4LUTCoords.y = fViewDirLocalAzimuth;
    f4LUTCoords.z = fViewDirLocalZenith;

    // In case the parameterization is performed for the sphere surface, the allowable range for the 
    // view direction zenith angle is [0, PI/2] since the ray should always be directed into the sphere.
    // Otherwise the range is whole [0, PI]
    f4LUTCoords.xyz = f4LUTCoords.xyz / float3(PI, 2*PI, bSurfaceOnly ? (PI/2) : PI) + float3(0, 0.5, 0);
    if( bSurfaceOnly )
        f4LUTCoords.w = 0;
    else
        f4LUTCoords.w = fDistFromCenter;
    
    if( bSurfaceOnly )
        f4LUTCoords.xz = clamp(f4LUTCoords.xyz, 0.5/SRF_SCATTERING_IN_PARTICLE_LUT_DIM, 1-0.5/SRF_SCATTERING_IN_PARTICLE_LUT_DIM).xz;
    else
        f4LUTCoords.xzw = clamp(f4LUTCoords, 0.5/VOL_SCATTERING_IN_PARTICLE_LUT_DIM, 1-0.5/VOL_SCATTERING_IN_PARTICLE_LUT_DIM).xzw;

    return f4LUTCoords;
}

/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/


// All noise functions are designed for values on integer scale.
// They are tuned to avoid visible periodicity for both positive and
// negative coordinates within a few orders of magnitude.
// https://www.shadertoy.com/view/4dS3Wd
__device__ float hash(float  n){ 
    return fract(sin(n) * 1e4); 
}
__device__ float hash(float2 p) { 
    return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); 
}
__device__ float noise(float3 x) {
    const vec3 step = vec3(110, 241, 171);

    vec3 i = floor(x);
    vec3 f = fract(x);
 
    // For performance, compute the base input to a 1D hash from the integer part of the argument and the 
    // incremental change to the 1D based on the 3D -> 1D wrapping
    float n = dot(i, step);

    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}
// By Morgan McGuire @morgan3d, http://graphicscodex.com
// Reuse permitted under the BSD license.

__device__ float GetRandomDensity(float3 pos, float startFreq, int n_Octaves = 3, float amplitudeScale = 0.6){
    float noiseFrame = 0;
    float amplitude = 1;
    float fFreq = startFreq;
    for(int i=0; i < n_Octaves; ++i){
        noiseFrame += ( noise( pos*fFreq ) - 0.5 ) * amplitude;
        fFreq *= 1.7;
        amplitude *= amplitudeScale;
    }
    return noiseFrame;
}

__device__ float GetMetabolDensity(float r){
    float r2 = r*r;
    float r4 = r2*r2;
    float r6 = r4*r2;
    return saturate(-4.0/9.0 * r6 + 17.0/9.0 * r4 - 22.0/9.0 * r2 + 1);
}

__device__ float ComputeDensity(float3 currPos){
	float distToCenter   = length(currPos);
    float metabolDensity = GetMetabolDensity(distToCenter);
	float density = 0.f;

    density = saturate( 1.0*saturate(metabolDensity) + 1*pow(metabolDensity,0.5)*(GetRandomDensity(currPos + 0.5, 0.15, 4, 0.7 )) );

	return density;
}

__device__ float2 PreComputeOpticalDepth(){
    // This shader computes level 0 of the maximum density mip map

    float3 normalizedStartPos, rayDir;
    //OpticalDepthLUTCoordsToWorldParams( float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.xy), normalizedStartPos, rayDir );
    
    // Intersect the view ray with the unit sphere:
    float2 rayIsecs;
    // normalizedStartPos  is located exactly on the surface; slightly move start pos inside the sphere
    // to avoid precision issues
    GetRaySphereIntersection(normalizedStartPos + rayDir*1e-4, rayDir, 0, 1.f, rayIsecs);
    
    if( rayIsecs.x > rayIsecs.y )
        return 0;

    float3 endPos = normalizedStartPos + rayDir * rayIsecs.y;
    float numSteps = 64;  // -------------------------------------------------------------- #$
    float3 f3Step = (endPos - normalizedStartPos) / numSteps;

    float totalDensity = 0;
    for(float fStepNum=0.5; fStepNum < numSteps; ++fStepNum){
        float3 f3CurrPos = normalizedStartPos + f3Step * fStepNum;
        float density = ComputeDensity(f3CurrPos);
        totalDensity += density;
    }
    return totalDensity / numSteps;
}


/*
 *  SINGLE SCATTERING: Device functions
 *  =================  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
// float PrecomputeSingleSctrPS(SScreenSizeQuadVSOutput In)
__device__ float PrecomputeSingleSctrPS(){

    //float4 LUTCoords = float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.xy);
 
    float3 entryPointUSSpace, viewRayUSSpace, lightDirUSSpace;
    ParticleScatteringLUTToWorldParams(LUTCoords, entryPointUSSpace, viewRayUSSpace, lightDirUSSpace, false);
 
    // Intersect view ray with the unit sphere:
    float2 rayIsecs;
    // f3NormalizedStartPos  is located exactly on the surface; slightly move the start pos inside the sphere
    // to avoid precision issues
    float3 biasedEntryPoint = entryPointUSSpace + viewRayUSSpace*1e-4;
    GetRaySphereIntersection(biasedEntryPoint, viewRayUSSpace, 0, 1.f, rayIsecs);

    if( rayIsecs.y < rayIsecs.x )
        return 0;
    float3 endPos = biasedEntryPoint + viewRayUSSpace * rayIsecs.y;
 
    float numSteps = NUM_INTEGRATION_STEPS;
    float3 step = (endPos - entryPointUSSpace) / numSteps;
    float stepLen = length(step);
    float cloudMassToCamera = 0;
    float particleRadius = g_GlobalCloudAttribs.fReferenceParticleRadius;
    float inscattering = 0;
    for(float stepNum=0.5; stepNum < numSteps; ++stepNum){
        float3 currPos = entryPointUSSpace + step * stepNum;
        float  cloudMassToLight = 0;
        GetRaySphereIntersection(currPos, lightDirUSSpace, 0, 1.f, rayIsecs);
         
        if( rayIsecs.y > rayIsecs.x ){
            // Since we are using the light direction (not direction on light), we have to use 
            // the first intersection point:
            cloudMassToLight = abs(rayIsecs.x) * particleRadius;
        }
 
        float totalLightAttenuation = exp( -g_GlobalCloudAttribs.fAttenuationCoeff * (cloudMassToLight + cloudMassToCamera) );
        inscattering += totalLightAttenuation * g_GlobalCloudAttribs.fScatteringCoeff;
        cloudMassToCamera += stepLen * particleRadius;
    }
 
    return inscattering * stepLen * particleRadius;
 }









/*
 *  MULTIPLE SCATTERING: Device functions
 *  ===================  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

//float GatherScatteringPS(SScreenSizeQuadVSOutput In) : SV_Target{
__device__ float GatherScatteringPS(){
    //float4 LUTCoords = float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.xy);

    float3 posUSSpace, viewRayUSSpace, lightDirUSSpace;
    ParticleScatteringLUTToWorldParams(LUTCoords, posUSSpace, viewRayUSSpace, lightDirUSSpace, false);

    float3 localX, localY, localZ;
    ConstructLocalFrameXYZ(-normalize(posUSSpace), lightDirUSSpace, localX, localY, localZ);

    float gatheredScattering = 0;
    float totalSolidAngle = 0;
    const float numZenithAngles  = VOL_SCATTERING_IN_PARTICLE_LUT_DIM.z;
    const float numAzimuthAngles = VOL_SCATTERING_IN_PARTICLE_LUT_DIM.y;
    const float zenithSpan  =   PI;
    const float azimuthSpan = 2*PI;
    for(float ZenithAngleNum = 0.5; ZenithAngleNum < numZenithAngles; ++ZenithAngleNum)
        for(float AzimuthAngleNum = 0.5; AzimuthAngleNum < numAzimuthAngles; ++AzimuthAngleNum){

            float  ZenithAngle = ZenithAngleNum/numZenithAngles * zenithSpan;
            float  AzimuthAngle = (AzimuthAngleNum/numAzimuthAngles - 0.5) * azimuthSpan;
            float3 currDir = GetDirectionInLocalFrameXYZ(localX, localY, localZ, ZenithAngle, AzimuthAngle);
            float4 currDirLUTCoords = WorldParamsToParticleScatteringLUT(posUSSpace, currDir, lightDirUSSpace, false);
            float  currDirScattering = 0;

            SAMPLE_4D_LUT(g_tex3DPrevSctrOrder, VOL_SCATTERING_IN_PARTICLE_LUT_DIM, currDirLUTCoords, 0, currDirScattering);

            if( g_GlobalCloudAttribs.f4Parameter.w == 1 ){
                currDirScattering *= HGPhaseFunc( dot(-currDir, lightDirUSSpace) );
            }
            currDirScattering *= HGPhaseFunc( dot(currDir, viewRayUSSpace), 0.7 );

            float fdZenithAngle  = zenithSpan / numZenithAngles;
            float fdAzimuthAngle = azimuthSpan / numAzimuthAngles * sin(ZenithAngle);
            float diffSolidAngle = fdZenithAngle * fdAzimuthAngle;
            totalSolidAngle += diffSolidAngle;
            gatheredScattering += currDirScattering * diffSolidAngle;
        }
    
    // Total solid angle should be 4*PI. Renormalize to fix discretization issues
    gatheredScattering *= 4*PI / totalSolidAngle;

    return gatheredScattering;
}


//float ComputeScatteringOrderPS(SScreenSizeQuadVSOutput In) : SV_Target
//{
__device__ float ComputeScatteringOrderPS(){

    //float4 startPointLUTCoords = float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.xy);

    float3 posUSSpace, viewRayUSSpace, lightDirUSSpace;
    //ParticleScatteringLUTToWorldParams(startPointLUTCoords, posUSSpace, viewRayUSSpace, lightDirUSSpace, false);

    // Intersect view ray with the unit sphere:
    float2 rayIsecs;
    // f3NormalizedStartPos  is located exactly on the surface; slightly move start pos inside the sphere
    // to avoid precision issues
    float3 biasedPos = posUSSpace + viewRayUSSpace*1e-4;
    GetRaySphereIntersection(biasedPos, viewRayUSSpace, 0, 1.f, rayIsecs);
    
    if( rayIsecs.y < rayIsecs.x )
        return 0;

    float3 endPos = biasedPos + viewRayUSSpace * rayIsecs.y;
    float  numSteps = max(VOL_SCATTERING_IN_PARTICLE_LUT_DIM.w*2, NUM_INTEGRATION_STEPS)*2;
    float3 step = (endPos - posUSSpace) / numSteps;
    float stepLen = length(step);
    float cloudMassToCamera = 0;
    //float particleRadius = g_GlobalCloudAttribs.fReferenceParticleRadius;
    float inscattering = 0;

    float prevGatheredSctr = 0;
    SAMPLE_4D_LUT(g_tex3DGatheredScattering, VOL_SCATTERING_IN_PARTICLE_LUT_DIM, startPointLUTCoords, 0, prevGatheredSctr);
    
    // Light attenuation == 1
    for(float stepNum=1; stepNum <= numSteps; ++stepNum){
        float3 currPos = posUSSpace + step * stepNum;

        cloudMassToCamera += stepLen * particleRadius;
        float attenuationToCamera = exp( -g_GlobalCloudAttribs.fAttenuationCoeff * fCloudMassToCamera );

        float4 currDirLUTCoords = WorldParamsToParticleScatteringLUT(currPos, viewRayUSSpace, lightDirUSSpace, false);
        float gatheredScattering = 0;
        SAMPLE_4D_LUT(g_tex3DGatheredScattering, VOL_SCATTERING_IN_PARTICLE_LUT_DIM, currDirLUTCoords, 0, gatheredScattering);
        gatheredScattering *= attenuationToCamera;

        inscattering += (gatheredScattering + prevGatheredSctr) /2;
        prevGatheredSctr = gatheredScattering;
    }

    return inscattering * stepLen * particleRadius * g_GlobalCloudAttribs.fScatteringCoeff;
}







/*
 *  Kernels CUDA
 *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
 __global__ void OpticalDepthKernel(dim3 texDim){
//
//  Get position
//  ............
    int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if(x >= texDim.x || y >= texDim.y || z >= texDim.z){
		return;
    }
    

    /*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/

    
}

 __global__ void ScatteringKernel(dim3 texDim){
//
//  Get position
//  ............
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if(x >= texDim.x || y >= texDim.y || z >= texDim.z){
        return;
    }
    

}


/*
 *  Lounch Kernels CUDA
 *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
extern "C"
void launchOpticalDepthKernel(cudaArray *cudaImageArray, dim3 texDim){
//
//  Dimension setting
//  .................
    dim3 blockDim(8, 8, 8);
    dim3  gridDim(texDim.x/blockDim.x, texDim.y/blockDim.y, texDim.z/blockDim.z);
    
    //Bind voxel array to a writable CUDA surface
    cudaBindSurfaceToArray(surfaceOpticalDepthWrite, cudaImageArray);

//
//  Run Kernel (optical depth)
//  ..........................
    OpticalDepthKernel<<< gridDim, blockDim >>>(texDim);

}

extern "C"
void launchScatteringKernel(cudaArray *cudaImageArray, dim3 texDim){
//
//  Dimension setting
//  .................
    dim3 blockDim(8, 8, 8);
    dim3  gridDim(texDim.x/blockDim.x, texDim.y/blockDim.y, texDim.z/blockDim.z);
    
    //Bind voxel array to a writable CUDA surface
    cudaBindSurfaceToArray(surfaceScatteringWrite, cudaImageArray);

//
//  Run Kernel (scattering)
//  .......................
    ScatteringKernel<<< gridDim, blockDim >>>(texDim);

}
