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
    dim3 block_dim(8, 8, 8);
    dim3 grid_dim(texDim.x/block_dim.x, texDim.y/block_dim.y, texDim.z/block_dim.z);
    
    //Bind voxel array to a writable CUDA surface
    cudaBindSurfaceToArray(surfaceOpticalDepthWrite, cudaImageArray);

}

extern "C"
void launchScatteringKernel(cudaArray *cudaImageArray, dim3 texDim){
//
//  Dimension setting
//  .................
    dim3 block_dim(8, 8, 8);
    dim3 grid_dim(texDim.x/block_dim.x, texDim.y/block_dim.y, texDim.z/block_dim.z);
    
    //Bind voxel array to a writable CUDA surface
    cudaBindSurfaceToArray(surfaceScatteringWrite, cudaImageArray);

}
