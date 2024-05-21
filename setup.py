from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ctlib',
    version='0.2.0',
    author='Wenjun Xia',
    ext_modules=[
        CUDAExtension('ctlib', [
            'src/ctlib.cpp',
            'src/fan_ed_kernel.cu',
            'src/fan_ea_kernel.cu',
            'src/para_kernel.cu',
            'src/laplacian_cuda_kernel.cu',
        ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
