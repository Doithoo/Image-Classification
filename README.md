# CNN for Garbage classification
### Introduction

This repository is built for garbage classification task, which contains full code and dataset. 

### Usage
Train:

   ```shell
   cd Image-classification
   cd Code
   mkdir save_weight
   python train.py
   ```

   

Test:

   ```shell
   python predict.py
   ```

### Performance

Overall result:

|  lMethod  |  Acc  | Params  | Flops(MACs) |
| :-------: | :---: | :-----: | :---------: |
|  alexnet  | 0.816 | 14.59M  |   310.07M   |
|   vgg11   | 0.855 | 128.79M |    7.63G    |
|   vgg13   | 0.859 | 128.98M |   11.34G    |
|   vgg16   | 0.832 | 134.29M |    15.5G    |
|   vgg19   | 0.824 | 139.59M |   19.66G    |
| resnet18  | 0.855 | 11.18M  |    1.82G    |
| resnet34  | 0.867 | 21.29M  |    3.68G    |
| resnet50  | 0.871 | 23.52M  |    4.12G    |
| resnet101 | 0.832 | 42.51M  |    7.85G    |

