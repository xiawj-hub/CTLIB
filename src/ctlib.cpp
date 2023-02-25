#include <torch/extension.h>
#include "fan_ed_cuda.h"
#include "fan_ea_cuda.h"
#include "para_cuda.h"
#include "laplacian_cuda.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define Fan_Equal_Distance 0
#define Fan_Equal_Angle 1
#define Para 2

torch::Tensor backprojection_t(torch::Tensor image, torch::Tensor options) {
  CHECK_INPUT(image);
  CHECK_INPUT(options);
  int scan_type = options[static_cast<int>(options.size(0))-1].item<int>();
  if (scan_type == Fan_Equal_Distance){
    return bprj_t_fan_ed_cuda(image, options);
  } else if (scan_type == Fan_Equal_Angle) {
    return bprj_t_fan_ea_cuda(image, options);
  } else if (scan_type == Para) {
    return bprj_t_para_cuda(image, options);
  } else {
    exit(0);
  }
}

torch::Tensor backprojection(torch::Tensor projection, torch::Tensor options) {
  CHECK_INPUT(projection);
  CHECK_INPUT(options);
  int scan_type = options[static_cast<int>(options.size(0))-1].item<int>();
  if (scan_type == Fan_Equal_Distance){
    return bprj_fan_ed_cuda(projection, options);
  } else if (scan_type == Fan_Equal_Angle) {
    return bprj_fan_ea_cuda(projection, options);
  } else if (scan_type == Para) {
    return bprj_para_cuda(projection, options);
  } else {
    exit(0);
  }
}

torch::Tensor backprojection_sv(torch::Tensor projection, torch::Tensor options) {
  CHECK_INPUT(projection);
  CHECK_INPUT(options);
  int scan_type = options[static_cast<int>(options.size(0))-1].item<int>();
  if (scan_type == Fan_Equal_Distance){
    return bprj_sv_fan_ed_cuda(projection, options);
  } else {
    exit(0);
  }
}

torch::Tensor projection(torch::Tensor image, torch::Tensor options) {
  CHECK_INPUT(image);
  CHECK_INPUT(options);
  int scan_type = options[static_cast<int>(options.size(0))-1].item<int>();
  if (scan_type == Fan_Equal_Distance){
    return prj_fan_ed_cuda(image, options);
  } else if (scan_type == Fan_Equal_Angle) {
    return prj_fan_ea_cuda(image, options);
  } else if (scan_type == Para) {
    return prj_para_cuda(image, options);
  } else {
    exit(0);
  }
}

torch::Tensor projection_t(torch::Tensor projection, torch::Tensor options) {
  CHECK_INPUT(projection);
  CHECK_INPUT(options);
  int scan_type = options[static_cast<int>(options.size(0))-1].item<int>();
  if (scan_type == Fan_Equal_Distance){
    return prj_t_fan_ed_cuda(projection, options);
  } else if (scan_type == Fan_Equal_Angle) {
    return prj_t_fan_ea_cuda(projection, options);
  } else if (scan_type == Para) {
    return prj_t_para_cuda(projection, options);
  } else {
    exit(0);
  }
}

torch::Tensor fbp(torch::Tensor projection, torch::Tensor options) {
  CHECK_INPUT(projection);
  CHECK_INPUT(options);
  int scan_type = options[static_cast<int>(options.size(0))-1].item<int>();
  if (scan_type == Fan_Equal_Distance){
    return fbp_fan_ed_cuda(projection, options);
  } else if (scan_type == Fan_Equal_Angle) {
    return fbp_fan_ea_cuda(projection, options);
  } else if (scan_type == Para) {
    return fbp_para_cuda(projection, options);
  } else {
    exit(0);
  }
}

torch::Tensor laplacian(torch::Tensor input, int k) {
  CHECK_INPUT(input);
  return laplacian_cuda_forward(input, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("projection", &projection, "CT projection (CUDA)");
  m.def("projection_t", &projection_t, "Transpose of CT projection (CUDA)");
  m.def("backprojection_t", &backprojection_t, "Transpose of backprojection (CUDA)");
  m.def("backprojection", &backprojection, "CT backprojection (CUDA)");
  m.def("backprojection_sv", &backprojection_sv, "CT backprojection single view (CUDA)");
  m.def("fbp", &fbp, "CT filtered backprojection with RL filter (CUDA)");
  m.def("laplacian", &laplacian, "Graph Laplacian computation (CUDA)");
}
