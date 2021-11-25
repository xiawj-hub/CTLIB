#ifndef FAN_EA_CUDA_H
#define FAN_EA_CUDA_H

#include <torch/extension.h>

torch::Tensor prj_fan_ea_cuda(torch::Tensor image, torch::Tensor options);
torch::Tensor prj_t_fan_ea_cuda(torch::Tensor projection, torch::Tensor options);
torch::Tensor bprj_t_fan_ea_cuda(torch::Tensor image, torch::Tensor options);
torch::Tensor bprj_fan_ea_cuda(torch::Tensor image, torch::Tensor options);
torch::Tensor fbp_fan_ea_cuda(torch::Tensor projection, torch::Tensor options);

#endif