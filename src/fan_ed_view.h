#ifndef FAN_ED_VIEW_H
#define FAN_ED_VIEW_H

#include <torch/extension.h>

torch::Tensor prj_fan_ed_view_cuda(torch::Tensor image, torch::Tensor options, torch::Tensor prj_views);
torch::Tensor bprj_fan_ed_view_cuda(torch::Tensor projection, torch::Tensor options, torch::Tensor prj_views);

#endif