#include "lightTransport.h"

//Global scope surface to bind to
surface<void, cudaSurfaceType3D> surfaceOpticalDepthWrite;
surface<void, cudaSurfaceType3D>   surfaceScatteringWrite;

/*
 *  Device functions
 *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
 __device__ void spherical2cartesian(){

 }

__device__ void GetRaySphereIntersection(float3 &rayOrigin,    float3 &rayDirection, 
                                         float3 &sphereCenter, float  &sphereRadius
                                         float2 &intersections){
    // http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
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
    float fNumSteps = NUM_INTEGRATION_STEPS;  // -------------------------------------------------------------- #$
    float3 f3Step = (endPos - normalizedStartPos) / fNumSteps;
    float fTotalDensity = 0;
    for(float fStepNum=0.5; fStepNum < fNumSteps; ++fStepNum){
        float3 f3CurrPos = normalizedStartPos + f3Step * fStepNum;
        float fDensity = ComputeDensity(f3CurrPos);
        fTotalDensity += fDensity;
    }
    return fTotalDensity / fNumSteps;
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
