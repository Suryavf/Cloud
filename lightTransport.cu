#include "lightTransport.h"

#define OPTICAL_DEPTH_LUT_DIM float4(64,32,64,32)
#define NUM_PARTICLE_LAYERS 1

#define SRF_SCATTERING_IN_PARTICLE_LUT_DIM make_float3(32,64,16)
#define VOL_SCATTERING_IN_PARTICLE_LUT_DIM make_float4(32,64,32,8)
#define THREAD_GROUP_SIZE 64
#define NUM_INTEGRATION_STEPS 64
#define PI                              3.1415928f
#define _fReferenceParticleRadius 200.0f
#define _fAttenuationCoeff        0.07f
#define _fScatteringCoeff         0.07f

//Global scope surface to bind to
surface<void, cudaSurfaceType3D> surfaceOpticalDepthWrite;
surface<void, cudaSurfaceType3D>   surfaceScatteringWrite;



/*
 *  GENERAL: Device functions
 *  =======  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
__device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
__device__ float3 operator*(const float3 &v, const float &a) {
    return make_float3(v.x*a, v.y*a, v.z*a);
}
__device__ float3 operator*(const float &a, const float3 &v) {
    return make_float3(v.x*a, v.y*a, v.z*a);
}
__device__ float3 neg(const float3 &a){
    return make_float3( - a.x, - a.y, - a.z);
}
__device__ float3 prod(float a, float3 v){
    return make_float3( a*v.x, a*v.y , a*v.z );
}
__device__ float3 frac(float3 v, float a){
    return make_float3( v.x/a , v.y/a , v.z/a );
}
__device__ float2 frac(float2 v, float a){
    return make_float2( v.x/a , v.y/a );
}
__device__ float dot(float3 a, float3 b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ float3 p2p(float3 a, float3 b){
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
__device__ float3 normalize(float3 v){
    return prod( rsqrt(dot(v,v)) , v );//rsqrt(dot(v,v))  *v;
}
__device__ float length(float3 v){
  return sqrt(dot(v,v));
}
__device__ float clamp(float x, float a, float b){
  return max(a, min(b, x));
}
__device__ float fract(float  x){ 
    return  x - floor(x); 
}
__device__ float3 floor(float3 v){
    return make_float3( floor(v.x), floor(v.y), floor(v.z) );
}
__device__ float3 fract(float3 v){
    return make_float3( fract(v.x), fract(v.y), fract(v.z) );
}
__device__ float3 lerp(float3 a, float3 b, float w){
  return a + prod(w,(b-a));
}
__device__ float lerp(float a, float b, float w){
  return a + w*(b-a);
}
__device__ float3 cross(float3 a, float3 b){
  return make_float3(  a.y*b.z - a.z*b.y , -a.x*b.z + a.z*b.x , a.x*b.y - a.y*b.x );
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
    sincosf( zenithAngle,  &zenithSin,  &zenithCos);
    sincosf(azimuthAngle, &azimuthSin, &azimuthCos);

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
__device__ void ConstructLocalFrameXYZ(float3 up, float3 inward, 
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
    sincosf(localZenithAngle, &dirLocalSinZenithAngle, &dirLocalCosZenithAngle);

    // Compute sin and cos of the local azimuth angle
    float dirLocalAzimuthCos, dirLocalAzimuthSin;
    sincosf(localAzimuthAngle, &dirLocalAzimuthSin, &dirLocalAzimuthCos);
    
    // Reconstruct view ray
    //return localZ * dirLocalCosZenithAngle + dirLocalSinZenithAngle * (dirLocalAzimuthCos * localX + dirLocalAzimuthSin * localY );
    return    prod(dirLocalCosZenithAngle                   ,localZ) 
            + prod(dirLocalSinZenithAngle*dirLocalAzimuthCos,localX)
            + prod(dirLocalSinZenithAngle*dirLocalAzimuthSin,localY);
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
__device__ void GetRaySphereIntersection(float3  rayOrigin,    float3  rayDirection, 
                                         float3  sphereCenter, float  &sphereRadius,
                                         float2 &intersections){
   rayOrigin = rayOrigin - sphereCenter;
   float A =     dot(rayDirection, rayDirection);
   float B = 2 * dot(rayOrigin,    rayDirection);
   float C = dot(rayOrigin, rayOrigin) - sphereRadius*sphereRadius;
   float D = B*B - 4*A*C;

   // If discriminant is negative, there are no real roots hence the ray misses the
   // sphere
   if( D<0 ){
       intersections = make_float2(-1,-1);
   }
   else{
       D = sqrt(D);
       intersections = make_float2( (-B - D)/(2*A) , (-B + D)/(2*A) ); // A must be positive here!!
   }
}

__device__ float HGPhaseFunc(float cosTheta, const float g = 0.9){
    return (1/(4*PI) * (1 - g*g)) / pow( max((1 + g*g) - (2*g)*cosTheta,0.0f), 3.f/2.f);
}



/*
 *  OPTICAL DEPTH: Device functions
 *  =============  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#define LERP(v0,v1,t) (1-t)*v0 + t*v1
#define SAMPLE_4D_LUT(tex3D, LUT_DIM, f4LUTCoords, Result){   \
     float3 f3UVW;                                               \
     f3UVW.x = f4LUTCoords.x;                                  \
     f3UVW.y = f4LUTCoords.y;                                  \
     float fQSlice = f4LUTCoords.w * LUT_DIM.w - 0.5;            \
     float fQ0Slice = floor(fQSlice);                            \
     float fQWeight = fQSlice - fQ0Slice;                        \
                                                                 \
     f3UVW.z = (fQ0Slice + f4LUTCoords.z) / LUT_DIM.w;           \
                                                                               \
     Result = LERP( texture3D(tex3D, f3UVW ).x, texture3D(tex3D, frac(f3UVW + float3(0,0,1/LUT_DIM.w)) ).x, fQWeight); \
 } \
 

 __device__ void ParticleScatteringLUTToWorldParams(float4 f4LUTCoords, 
                                                    float3 &f3StartPosUSSpace,
                                                    float3 &f3ViewDirUSSpace,
                                                    float3 &f3LightDirUSSpace){
    f3LightDirUSSpace = make_float3(0,0,1);
    float fStartPosZenithAngle = f4LUTCoords.x * PI;
    f3StartPosUSSpace = make_float3(0,0,0);
    sincosf(fStartPosZenithAngle, &f3StartPosUSSpace.x, &f3StartPosUSSpace.z);
    
    // Constructs local XYZ (Z Up) frame from Up and Inward vectors
    float3 f3LocalX, f3LocalY, f3LocalZ;
    f3LocalZ = normalize(neg(f3StartPosUSSpace));
    f3LocalX = normalize(cross(f3LightDirUSSpace, f3LocalZ));
    f3LocalY = normalize(cross(f3LocalZ, f3LocalX));

    bool bSurfaceOnly = true;
    if( !bSurfaceOnly ){
        float fDistFromCenter = f4LUTCoords.w;
        // Scale the start position according to the distance from center
        f3StartPosUSSpace = prod(fDistFromCenter,f3StartPosUSSpace);//*= fDistFromCenter;
    }

    float fViewDirLocalAzimuth = (f4LUTCoords.y - 0.5) * (2 * PI); 
    float fViewDirLocalZenith = f4LUTCoords.z * (PI/2) ;
    f3ViewDirUSSpace = GetDirectionInLocalFrameXYZ(f3LocalX, f3LocalY, f3LocalZ, fViewDirLocalZenith, fViewDirLocalAzimuth);
}

// All parameters must be defined in the unit sphere (US) space
__device__ float4 WorldParamsToParticleScatteringLUT(   float3 f3StartPosUSSpace, 
                                                        float3 f3ViewDirInUSSpace, 
                                                        float3 f3LightDirInUSSpace  ){
    
    float4 f4LUTCoords = make_float4(0,0,0,0);
    float fDistFromCenter = 0;
    
    // Compute distance from center and normalize start position
    fDistFromCenter = length(f3StartPosUSSpace);
    f3StartPosUSSpace = frac( f3StartPosUSSpace, max(fDistFromCenter, 1e-5) );

    float fStartPosZenithCos = dot(f3StartPosUSSpace, f3LightDirInUSSpace);
    f4LUTCoords.x = acos(fStartPosZenithCos);

    float3 f3LocalX, f3LocalY, f3LocalZ;
    ConstructLocalFrameXYZ(neg(f3StartPosUSSpace), f3LightDirInUSSpace, f3LocalX, f3LocalY, f3LocalZ);

    float fViewDirLocalZenith, fViewDirLocalAzimuth;
    ComputeLocalFrameAnglesXYZ(f3LocalX, f3LocalY, f3LocalZ, f3ViewDirInUSSpace, fViewDirLocalZenith, fViewDirLocalAzimuth);
    f4LUTCoords.y = fViewDirLocalAzimuth;
    f4LUTCoords.z = fViewDirLocalZenith;

    // In case the parameterization is performed for the sphere surface, the allowable range for the 
    // view direction zenith angle is [0, PI/2] since the ray should always be directed into the sphere.
    // Otherwise the range is whole [0, PI]
    f4LUTCoords.x = f4LUTCoords.x/   PI       ;
    f4LUTCoords.y = f4LUTCoords.y/(2*PI) + 0.5;
    f4LUTCoords.z = f4LUTCoords.z/(PI/2)      ;
    f4LUTCoords.w = 0;
    
    f4LUTCoords.x = clamp(f4LUTCoords.x, 0.5/SRF_SCATTERING_IN_PARTICLE_LUT_DIM.x, 1-0.5/SRF_SCATTERING_IN_PARTICLE_LUT_DIM.x);
    f4LUTCoords.z = clamp(f4LUTCoords.z, 0.5/SRF_SCATTERING_IN_PARTICLE_LUT_DIM.z, 1-0.5/SRF_SCATTERING_IN_PARTICLE_LUT_DIM.z);
    
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
    float3 step = make_float3(110, 241, 171);

    float3 i = floor(x);
    float3 f = fract(x);
 
    // For performance, compute the base input to a 1D hash from the integer part of the argument and the 
    // incremental change to the 1D based on the 3D -> 1D wrapping
    float n = dot(i, step);

    float3 u = f * f;
    float3 t = make_float3(3.0,3.0,3.0) - prod(2.0,f);
    u = u * t;

    return lerp(lerp(lerp( hash(n + dot(step, make_float3(0, 0, 0))), hash(n + dot(step, make_float3(1, 0, 0))), u.x),
                     lerp( hash(n + dot(step, make_float3(0, 1, 0))), hash(n + dot(step, make_float3(1, 1, 0))), u.x), u.y),
                lerp(lerp( hash(n + dot(step, make_float3(0, 0, 1))), hash(n + dot(step, make_float3(1, 0, 1))), u.x),
                     lerp( hash(n + dot(step, make_float3(0, 1, 1))), hash(n + dot(step, make_float3(1, 1, 1))), u.x), u.y), u.z);
}
// By Morgan McGuire @morgan3d, http://graphicscodex.com
// Reuse permitted under the BSD license.

__device__ float GetRandomDensity(float3 pos, float startFreq, int n_Octaves = 3, float amplitudeScale = 0.6){
    float noiseFrame = 0;
    float amplitude = 1;
    float fFreq = startFreq;
    for(int i=0; i < n_Octaves; ++i){
        noiseFrame += ( noise( prod(fFreq,pos) ) - 0.5 ) * amplitude;
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

    density = saturate( 1.0*saturate(metabolDensity) + 1*pow(metabolDensity,0.5)*(GetRandomDensity(currPos + make_float3(0.5,0.5,0.5) , 0.15, 4, 0.7 )) );
	return density;
}

__device__ float2 PreComputeOpticalDepth(float3 normalizedStartPos, float3 rayDir){
    // This shader computes level 0 of the maximum density mip map

    //float3 normalizedStartPos, rayDir;
    //OpticalDepthLUTCoordsToWorldParams( float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.xy), normalizedStartPos, rayDir );
    
    // Intersect the view ray with the unit sphere:
    float2 rayIsecs;
    // normalizedStartPos  is located exactly on the surface; slightly move start pos inside the sphere
    // to avoid precision issues
    float3 vec = normalizedStartPos + prod(1e-4,rayDir);
    float  radius = 1.0f;
    GetRaySphereIntersection(vec, rayDir, make_float3(0,0,0), radius, rayIsecs);
    
    if( rayIsecs.x > rayIsecs.y )
        return make_float2(0,0);
    
    float  numSteps = NUM_INTEGRATION_STEPS;
    float3 endPos = normalizedStartPos + rayDir * rayIsecs.y;
    float3 f3Step = frac(endPos - normalizedStartPos,numSteps);

    float totalDensity = 0;
    for(float fStepNum=0.5; fStepNum < numSteps; ++fStepNum){
        float3 f3CurrPos = normalizedStartPos + f3Step * fStepNum;
        float density = ComputeDensity(f3CurrPos);
        totalDensity += density;
    }
    return make_float2(totalDensity/numSteps,totalDensity/numSteps);
}


/*
 *  SINGLE SCATTERING: Device functions
 *  =================  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
// float PrecomputeSingleSctrPS(SScreenSizeQuadVSOutput In)
__device__ float PrecomputeSingleSctrPS(float4 LUTCoords){

    //float4 LUTCoords = float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.xy);
 
    float3 entryPointUSSpace, viewRayUSSpace, lightDirUSSpace;
    ParticleScatteringLUTToWorldParams(LUTCoords, entryPointUSSpace, viewRayUSSpace, lightDirUSSpace);
 
    // Intersect view ray with the unit sphere:
    float2 rayIsecs;
    // f3NormalizedStartPos  is located exactly on the surface; slightly move the start pos inside the sphere
    // to avoid precision issues
    float  radius = 1.0f;
    float3 biasedEntryPoint = entryPointUSSpace + viewRayUSSpace*1e-4;
    GetRaySphereIntersection(biasedEntryPoint, viewRayUSSpace, make_float3(0,0,0), radius, rayIsecs);

    if( rayIsecs.y < rayIsecs.x )
        return 0;
    float3 endPos = biasedEntryPoint + viewRayUSSpace * rayIsecs.y;
 
    float numSteps = NUM_INTEGRATION_STEPS;
    float3 step = frac(endPos - entryPointUSSpace,numSteps);
    float stepLen = length(step);
    float cloudMassToCamera = 0;
    float particleRadius = _fReferenceParticleRadius; 
    float inscattering = 0;
    for(float stepNum=0.5; stepNum < numSteps; ++stepNum){
        float3 currPos = entryPointUSSpace + step * stepNum;
        float  cloudMassToLight = 0;
        GetRaySphereIntersection(currPos, lightDirUSSpace, make_float3(0,0,0), radius, rayIsecs);
         
        if( rayIsecs.y > rayIsecs.x ){
            // Since we are using the light direction (not direction on light), we have to use 
            // the first intersection point:
            cloudMassToLight = abs(rayIsecs.x) * particleRadius;
        }
 
        float totalLightAttenuation = exp( -_fAttenuationCoeff * (cloudMassToLight + cloudMassToCamera) );
        inscattering += totalLightAttenuation * _fScatteringCoeff;
        cloudMassToCamera += stepLen * particleRadius;
    }
 
    return inscattering * stepLen * particleRadius;
 }




/*
 *  MULTIPLE SCATTERING: Device functions
 *  ===================  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

//float GatherScatteringPS(SScreenSizeQuadVSOutput In) : SV_Target{
__device__ float GatherScatteringPS(float4 LUTCoords){
    //float4 LUTCoords = float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.xy);

    float3 posUSSpace, viewRayUSSpace, lightDirUSSpace;
    ParticleScatteringLUTToWorldParams(LUTCoords, posUSSpace, viewRayUSSpace, lightDirUSSpace);

    float3 localX, localY, localZ;
    ConstructLocalFrameXYZ( neg(normalize(posUSSpace)), lightDirUSSpace, localX, localY, localZ);

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
            float4 currDirLUTCoords = WorldParamsToParticleScatteringLUT(posUSSpace, currDir, lightDirUSSpace);
            float  currDirScattering = 0;

            //SAMPLE_4D_LUT(surfaceScatteringWrite, VOL_SCATTERING_IN_PARTICLE_LUT_DIM, currDirLUTCoords, currDirScattering);   
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
__device__ float ComputeScatteringOrderPS(float3 viewRayUSSpace, float3 lightDirUSSpace){

    //float4 startPointLUTCoords = float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.xy);

    //ParticleScatteringLUTToWorldParams(startPointLUTCoords, posUSSpace, viewRayUSSpace, lightDirUSSpace, false);

    // Intersect view ray with the unit sphere:
    float2 rayIsecs;
    // f3NormalizedStartPos  is located exactly on the surface; slightly move start pos inside the sphere
    // to avoid precision issues
    float3 posUSSpace = make_float3(0.0,0.0,0.0);
    float3 biasedPos = posUSSpace + prod(1e-4,viewRayUSSpace);

    float  radius = 1.0f;
    GetRaySphereIntersection(biasedPos, viewRayUSSpace, make_float3(0,0,0), radius, rayIsecs);
    
    if( rayIsecs.y < rayIsecs.x )
        return 0;

    float3 endPos = biasedPos + viewRayUSSpace * rayIsecs.y;
    float  numSteps = max(VOL_SCATTERING_IN_PARTICLE_LUT_DIM.w*2, float(NUM_INTEGRATION_STEPS))*2;
    float3 step = frac(endPos - posUSSpace, numSteps);
    float stepLen = length(step);
    float fCloudMassToCamera = 0;
    float particleRadius = _fReferenceParticleRadius;
    float inscattering = 0;

    float prevGatheredSctr = 0;
    //SAMPLE_4D_LUT(g_tex3DGatheredScattering, VOL_SCATTERING_IN_PARTICLE_LUT_DIM, startPointLUTCoords,prevGatheredSctr);
    
    // Light attenuation == 1
    for(float stepNum=1; stepNum <= numSteps; ++stepNum){
        float3 currPos = posUSSpace + step * stepNum;

        fCloudMassToCamera += stepLen * particleRadius;
        float attenuationToCamera = exp( -_fAttenuationCoeff * fCloudMassToCamera );

        float4 currDirLUTCoords = WorldParamsToParticleScatteringLUT(currPos, viewRayUSSpace, lightDirUSSpace);
        float gatheredScattering = 0;
        //SAMPLE_4D_LUT(g_tex3DGatheredScattering, VOL_SCATTERING_IN_PARTICLE_LUT_DIM, currDirLUTCoords, gatheredScattering);
        gatheredScattering *= attenuationToCamera;

        inscattering += (gatheredScattering + prevGatheredSctr) /2;
        prevGatheredSctr = gatheredScattering;
    }

    return inscattering * stepLen * particleRadius * _fScatteringCoeff;
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
    
    // z = w*z
    int w = z%16;//z -floor(z/16)*16;
        z = (z-w)/16;

    float stepangle = PI/16;
    float   phiS = x * stepangle;
    float thetaS = y * stepangle;
    float   phiV = z * stepangle;
    float thetaV = w * stepangle;

    //float3 raydir = ZenithAzimuthAngleToDirectionXZY(phiV, thetaV);
    float4 LUTCoords = make_float4(phiS,thetaS,phiV,thetaV);
    float val = PrecomputeSingleSctrPS(LUTCoords);
    surf3Dwrite(val,surfaceOpticalDepthWrite,x*sizeof(float),y,z);
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
    
    float stepangle = PI/16;
    float   phiS = 0 * stepangle; // Simetrico!!
    float thetaS = x * stepangle;
    float   phiV = y * stepangle;
    float thetaV = z * stepangle;

    float3 viewdir = ZenithAzimuthAngleToDirectionXZY(phiV, thetaV);
    float3  raydir = ZenithAzimuthAngleToDirectionXZY(phiS, thetaS);

    float val = ComputeScatteringOrderPS(viewdir, raydir);
    surf3Dwrite(val,surfaceScatteringWrite,x*sizeof(float),y,z);
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
