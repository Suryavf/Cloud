#include "lightTransport.h"

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




/*
 *  OPTICAL DEPTH: Device functions
 *  =============  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
 

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
 *  SCATTERING: Device functions
 *  ==========  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */







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
