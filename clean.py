import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import warnings
from ImageLoad import TianchiDataset, RandomCrop, Rescale
from ImageLoad import ToTensor




if __name__ == '__main__':
    
    warnings.filterwarnings('error',category=UserWarning)

    train_data = TianchiDataset("../train.csv",
                          "../IJCAI_2019_AAAC_train",
                          transform=transforms.Compose([Rescale(400), RandomCrop(299), ToTensor()]))
    test_data = TianchiDataset("../test.csv",
                            "../IJCAI_2019_AAAC_train",
                          transform=transforms.Compose([Rescale(400), RandomCrop(299), ToTensor()]))

    frame1 = pd.read_csv("../train.csv")
    frame2 = pd.read_csv("../test.csv")
    root_dir = "../IJCAI_2019_AAAC_train"

    
    for idx in range(len(frame1)):

        img_name = os.path.join(root_dir, frame1.iloc[idx, 0])
        try:
            image = io.imread(img_name)
        except UserWarning:
            print('here!')
            with open('wrong.pth', 'a+') as f:
                f.write(str(image.shape))
                f.write(img_name)
        else:
            # print(image.shape)
            if len(image.shape) != 3:

                with open('wrong.pth', 'a+') as f:
                    f.write(str(image.shape))
                    f.write(img_name)
    
    for idx in range(len(frame2)):

        img_name = os.path.join(root_dir, frame2.iloc[idx, 0])
        try:
            image = io.imread(img_name)
        except UserWarning:
            print('here')
            with open('wrong.pth', 'a+') as f:
                f.write(str(image.shape))
                f.write(img_name)
        else:
            # print(image.shape)
            if len(image.shape) != 3:

                with open('wrong.pth', 'a+') as f:
                    f.write(str(image.shape))
                    f.write(img_name)



