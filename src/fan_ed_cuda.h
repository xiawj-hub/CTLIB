#ifndef FAN_ED_CUDA_H
#define FAN_ED_CUDA_H

#include <torch/extension.h>

torch::Tensor prj_fan_ed_cuda(torch::Tensor image, torch::Tensor options);
torch::Tensor prj_t_fan_ed_cuda(torch::Tensor projection, torch::Tensor options);
torch::Tensor bprj_t_fan_ed_cuda(torch::Tensor image, torch::Tensor options);
torch::Tensor bprj_fan_ed_cuda(torch::Tensor image, torch::Tensor options);
torch::Tensor bprj_sv_fan_ed_cuda(torch::Tensor image, torch::Tensor options);
torch::Tensor fbp_fan_ed_cuda(torch::Tensor projection, torch::Tensor options);

#endif
