# CTLIB
A lib of CT projector and back-projector based on PyTorch

Coded with distance driven method [1, 2]

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

For linux installation, remove `library_dirs` and `extra_link_args` in setup.py first.

You may fail to install this lib because
1) Your linux version is Ubuntu 18.04 or higher, this lib will be fialed to complie with g++ 7. You need to install the lower version of g++ and gcc.
2) The bug of pytorch. You can search the error code and find the solution in Stackoverflow.

## API
``projection(image, options)``: Projector of CT

``projection_t(projection, options)``: Transpose of projector

``backprojection_t(image, options)``: Transpose of backprojector

``backprojection(projection, options)``: Backprojector of CT

``fbp(projection, options)``: FBP with RL filter

``laplacian(input, k)``: Computation of adjancency matrix in [3]

``image``: 4D torch tensor, B x 1 x H x W,

``projection``: 4D torch tensor, B x 1 x V x D, V is the total number of scanning views, D is the total number of detector bins

``options``: 12D torch vector for fan beam and 9D torch vector for parallel beam, scanning geometry parameters, including

``views``: Number of scanning views

``dets``: Number of detector bins

``width`` and ``height``: Spatial resolution of images

``dImg``: Physical length of a pixel

``dDet``: Interval between two adjacent detector bins, especially, interval is ``rad`` for equal angle fan beam

``Ang0``: Starting angle

``dAng``: Interval between two adjacent scanning views: ``rad``

``s2r``: The distance between x-ray source and rotation center, not needed in parallel beam

``d2r``: The distance between detector and roration center, not needed in parallel beam

``binshift``: The shift of the detector

``scan_type``: ``0`` is equal distance fan beam, ``1`` is euql angle fan beam and ``2`` is parallel beam

[1] B. De Man and S. Basu, “Distance-driven projection and backprojection,”
in IEEE Nucl. Sci. Symp. Conf. Record, vol. 3, 2002, pp. 1477–80.

[2] B. De Man and S. Basu, “Distance-driven projection and backprojection in three dimensions,”
Phys. Med. Biol., vol. 49, no. 11, p. 2463, 2004.

[3] Xia, W., Lu, Z., Huang, Y., Shi, Z., Liu, Y., Chen, H., ... & Zhang, Y. (2021). MAGIC: Manifold and Graph Integrative Convolutional Network for Low-Dose CT Reconstruction. IEEE Transactions on Medical Imaging.
