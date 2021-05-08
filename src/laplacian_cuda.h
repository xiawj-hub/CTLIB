#ifndef LAPLACIAN_CUDA_H
#define LAPLACIAN_CUDA_H

#include <torch/extension.h>

torch::Tensor laplacian_cuda_forward(torch::Tensor input, int k);

#endif