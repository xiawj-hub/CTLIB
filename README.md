# CTLIB
A lib of CT projector and back-projector based on PyTorch

If you use the code, please cite our work
```
@article{xia2021magic,
  title={MAGIC: Manifold and Graph Integrative Convolutional Network for Low-Dose CT Reconstruction},
  author={Xia, Wenjun and Lu, Zexin and Huang, Yongqiang and Shi, Zuoqiang and Liu, Yan and Chen, Hu and Chen, Yang and Zhou, Jiliu and Zhang, Yi},
  journal={IEEE Transactions on Medical Imaging},
  year={2021},
  publisher={IEEE}
}
```
## Installation
Make sure PyTorch and CUDA have been installed correctly and then
```
python setup.py install
```

For linux installation, remove 'library_dirs' and 'extra_link_args' in setup.py first.

You may fail to install this lib because
1) Your linux version is Ubuntu 18.04 or higher, this lib will be fialed to complie with g++ 7. You need to install the lower version of g++ and gcc.
2) The bug of pytorch. You can search the error code and find the solution in Stackoverflow.

## API
projection(image, options, scan_type)
backprojection(projection, options, scan_type)
fbp_projection(image, options, scan_type)
fbp_backprojection(projection, options, scan_type)
fbp(projection, options, scan_typ)
laplacian(input, k)
