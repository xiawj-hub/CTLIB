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
The following is the step-by-step instruction to install this lib using conda

Create Conda environment
```
conda create -n ctlib
```
Enter the enviroment and install pytorch
```
conda activate ctlib
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Notice: Usually the cudatoolkit installed from nvidia channel will provide the complier. But if you install the previous pytorch version using earlier cudatoolkit which is not from nvidia channel, the cudatoolkit may not include the complier. The later installation will report the error that can't find nvcc. If so, you need install the lib of cudatoolkit-dev as follows:
```
conda install cudatoolkit-dev -c conda-forge
```
Move the the directory of this lib, and then compile and install
```
python setup.py install
```

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
