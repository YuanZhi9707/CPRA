# CPRA
This is a re-implementation of our paper [1] and for non-commercial use only. 
            --------------------------------------------------
            dataset    Rain100H   Rain 100L  Rain12   Rain1400
            --------------------------------------------------
            SSIM       0.832       0.971	   0.959	   0.930
            --------------------------------------------------
            PSNR      26.02	       35.53	   35.16	   31.11
            --------------------------------------------------
You need to install Python with Pytorch-GPU to run this code.

Usage:

1. Preparing training data: put rainy images into "/input"

2. Run
1）You can run testrain100L and testrain12, model is saved in "/logs/Rain100L".

2）Rain100H and Realrain use the same training model.

If this code and dataset help your research, please cite our related papers:
[1]
