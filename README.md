# CPRA
This is a re-implementation of our paper [1] and for non-commercial use only.<br>
| dataset | Rain100H | Rain100L | Rain12 | Rain1200 |
| --- | --- | --- | --- | --- |
| SSIM |0.901|0.986|0.962|0.936|  
| PSNR |29.66|39.63|35.88|34.59|

You need to install Python with Pytorch-GPU to run this code.<br>

Usage:<br>

1. Preparing training data: put rainy images into "/input"<br>

2. Run<br>
1）You can run testrain100L and testrain12, model is saved in "/logs/Rain100L".<br>
2）Rain100H and Realrain use the same training model.<br>

If this code and dataset help your research, please cite our related papers:<br>
[1]
