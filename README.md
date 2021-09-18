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
``projection(image, options, scan_type)``: Projector of CT

``backprojection(projection, options, scan_type)``: Transpose of projector

``fbp_projection(image, options, scan_type)``: Transpose of backprojector

``fbp_backprojection(projection, options, scan_type)``: Backprojector of CT

``fbp(projection, options, scan_typ)``: FBP with RL filter

``laplacian(input, k)``: Computation of adjancency matrix

``scan_type``: int, ``0`` is equal distance fan beam, ``1`` is euql angle fan beam and ``2`` is parallel beam

``image``: 4D torch tensor, $B\times 1\times H\times W$,

``projection``: 4D torch tensor, $B\times 1\times V\times D$, V is the total number of scanning views, D is the total number of detector bins

``options``: 11D torch vector, scanning geometry parameters, including

\t``views``: Number of scanning views

``dets``: Number of detector bins

``width`` and ``height``: Spatial resolution of images

``dImg``: Physical length of a pixel

``dDet``: Interval between two adjacent detector bins, especially, ``rad`` for equal angle fan beam

``Ang0``: Starting angle

``dAng``: Interval between two adjacent scanning views: ``rad``

``s2r``: The distance between x-ray source and rotation center

``d2r``: The distance between detector and roration center

``binshift``: The shift of the detector


