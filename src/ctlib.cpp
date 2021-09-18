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

torch::Tensor fbp_projection(torch::Tensor image, torch::Tensor options, int scan_type) {
  CHECK_INPUT(image);
  CHECK_INPUT(options);
  if (scan_type == Fan_Equal_Distance){
    return fbp_prj_fan_ed_cuda(image, options);
  } else if (scan_type == Fan_Equal_Angle) {
    return fbp_prj_fan_ea_cuda(image, options);
  } else if (scan_type == Para) {
    return fbp_prj_para_cuda(image, options);
  } else {
    exit(0);
  }
}

torch::Tensor fbp_backprojection(torch::Tensor projection, torch::Tensor options, int scan_type) {
  CHECK_INPUT(projection);
  CHECK_INPUT(options);
  if (scan_type == Fan_Equal_Distance){
    return fbp_bprj_fan_ed_cuda(image, options);
  } else if (scan_type == Fan_Equal_Angle) {
    return fbp_bprj_fan_ea_cuda(image, options);
  } else if (scan_type == Para) {
    return fbp_bprj_para_cuda(image, options);
  } else {
    exit(0);
  }
}

torch::Tensor projection(torch::Tensor image, torch::Tensor options, int scan_type) {
  CHECK_INPUT(image);
  CHECK_INPUT(options);
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

torch::Tensor backprojection(torch::Tensor projection, torch::Tensor options, int scan_type) {
  CHECK_INPUT(projection);
  CHECK_INPUT(options);
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

torch::Tensor fbp(torch::Tensor projection, torch::Tensor options, int scan_type) {
  CHECK_INPUT(projection);
  CHECK_INPUT(options);
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
  m.def("backprojection", &backprojection, "Transpose of CT projection (CUDA)");
  m.def("fbp_projection", &fbp_projection, "Transpose of backprojection (CUDA)");
  m.def("fbp_backprojection", &fbp_backprojection, "CT backprojection (CUDA)");
  m.def("fbp", &fbp, "CT filtered backprojection (CUDA)");
  m.def("laplacian", &laplacian, "Graph Laplacian computation (CUDA)");
}