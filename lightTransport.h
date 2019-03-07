#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "cublas_v2.h"

extern "C"
void launchOpticalDepthKernel(cudaArray *cudaImageArray, dim3 texDim);

extern "C"
void   launchScatteringKernel(cudaArray *cudaImageArray, dim3 texDim);
