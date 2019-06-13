# Train.py 
## Arguments
- **nepochs** 
Number of the max nepochs.
- **batch** 
Size of one batch. Default =128
- **gpu**
Default = 0
- **lr**
Learning Rate. Default = 0.0001
- **pretrain**
Train a new model (False) or continue training (True).
- **model**
Models to train. Used Model: res, inc, IncResV2, dense ...
- **mode** 
normal, noise, adv...
- **noise** 
If training in noise mode, it's the size of noise perturbation.

  noise: 25, 15 ....
  
  l_2 score: 34.99, 21.45 ...

# ImageLoad.py


```
from ImageLoad import TianchiDataset, RandomCrop, Rescale

if args.model == 'res':
    from ImageLoad import ToTensorRescale as ToTensor
else:
    from ImageLoad import ToTensor
```
Please note that ResNet is trained with images regularize to [0, 1/255], and others are trained with those regularize to [0,1]. By the way, output of inception V3 should be ``` model(x)[0]```
Because the model(x) is a tuple of 2 elements.

# clean.py
To clean the dataset, including exif error, dimension error 
