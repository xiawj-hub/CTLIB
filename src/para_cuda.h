#ifndef PARA_CUDA_H
#define PARA_CUDA_H
#include <torch/extension.h>

torch::Tensor prj_para_cuda(torch::Tensor image, torch::Tensor options);
torch::Tensor prj_t_para_cuda(torch::Tensor projection, torch::Tensor options);
torch::Tensor bprj_t_para_cuda(torch::Tensor image, torch::Tensor options);
torch::Tensor bprj_para_cuda(torch::Tensor projection, torch::Tensor options);
torch::Tensor fbp_para_cuda(torch::Tensor projection, torch::Tensor options);

#endif