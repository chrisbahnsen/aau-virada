## Is it Raining Outside? Detection of Rainfall using General-Purpose Surveillance Cameras

This repository contains the code and scripts for the paper *Is it Raining Outside? Detection of Rainfall using General-Purpose Surveillance Cameras*

The paper investigates rain detection with general-purpose surveillance cameras. The previous state-of-the-art method is compared with a 3D Convolutional Neural Network, on data from two different traffic crossings recorded in Aalborg, Denmark.

As this is a collection of research code, one might find some occasional rough edges. We have tried to clean up the code to a decent level but if you encounter a bug or a regular mistake, please report it in our issue tracker. 

### The AAU Visual Rain Dataset (VIRADA)
The evaluation code is built around the AAU VIRADA dataset which is published at [Zenodo](https://zenodo.org/record/4715681).

### Code references

The [NVidia Video Loader (NVVL)](https://github.com/NVIDIA/nvvl) framework was used for loading video snippets efficiently. The [Video Super-Resolution](https://github.com/NVIDIA/nvvl/tree/master/examples/pytorch_superres) example was used as the basis for the provided docker file

The pytorch implementation of the basis C3D network was derived from [David Abati](https://github.com/DavideA/c3d-pytorch)'s implementation.

The Bossu implementation utilizes OpenCV v. 3.1.0

### License

All code is licensed under the MIT license, except for the modified NVVL code, which is subject to its own license.

### Acknowledgements
Please cite the following paper if you use our code or dataset:

```TeX
@InProceedings{Haurum_2019_CVPR_Workshops,
author = {Haurum, Joakim Bruslund and Bahnsen, Chris H. and Moeslund, Thomas B.},
title = {Is it Raining Outside? Detection of Rainfall using General-Purpose Surveillance Cameras},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
} 
```