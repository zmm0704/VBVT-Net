# VBVT-Net

run `python hologic_sim_test_our.py` to obtain the propcessed 3D image patches

# Note
GPU memory >= 36 GB
```
./data: 10 pairs of test samples, each pair data incldudes: 4D VVBP-Tensor patch, 3D image patch, 3D label patch, 3D binary calcification mask patch
./model: the pretrained model parameter
./result: saving the 3D processed image patches.
check_result.m: show the results in MATLAB
```