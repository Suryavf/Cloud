#include <iostream>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

extern "C"
void launchOpticalDepthKernel(cudaArray *cudaImageArray, dim3 texDim);

extern "C"
void   launchScatteringKernel(cudaArray *cudaImageArray, dim3 texDim);
