import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from skimage import io, color, transform
import argparse
import time
import matplotlib.pyplot as plt
from ImageLoad import Rescale, ToTensor, TianchiDataset
from attack import _fgs, _pgd, _mi_fgm, _mi_fgm_t
from models import resnet
from models import densenet
from models import IncResV2
from models import InceptionV4


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='inception')
parser.add_argument('--attack', type=str, default='fgs', help="fgs, pgd, mi, mi_t, mask, test...")
parser.add_argument('--epsilon', type=float, default=40)
parser.add_argument('--data', type=str, default="../IJCAI_2019_AAAC_dev_data/dev_data/")
parser.add_argument('--output', type=str, default="./output")
parser.add_argument('--mask_size', type=int, default=250)

args = parser.parse_args()



if __name__ == "__main__":

    output_path = args.output
    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = []


    if args.model == 'incres':
        model = IncResV2.inceptionresnetv2(num_classes=110).to(device).train()
        model.load_state_dict(torch.load("model_log/incres_adv_mi.pth", map_location=device_name))
        model = {"model":model,
                 "ratio": 1.,
                 "mode": None,
                 "weight": 1.}
        models.append(model)

    elif args.model == 'ens_1':
        model_1 = IncResV2.inceptionresnetv2(num_classes=110).to(device)
        model_1.load_state_dict(torch.load("model_log/incres_normal_64.pth", map_location=device_name))

        model_2 = resnet.resnet34(num_classes=110).to(device)
        model_2.load_state_dict(torch.load("model_log/res_normal_64.pth", map_location=device_name))

        model_3 = InceptionV4.inceptionv4(num_classes=110).to(device)
        model_3.load_state_dict(torch.load("model_log/incv4_normal_63.pth", map_location=device_name))

        model_4 = densenet.densenet169(num_classes=110).to(device)
        model_4.load_state_dict(torch.load("model_log/des_normal_63.pth", map_location=device_name))
        
        models = [{
            "model": model_1,
            "ratio": 1.,
            "mode": None,
            "weight":0.333
        }, {
            "model": model_2,
            "ratio": 1.,
            "mode": None,
            "weight":0.333
        }, {
            "model": model_3,
            "ratio": 1,
            "mode": None,
            "weight":0.333
        }, {"model": model_4,
            "ratio": 1.,
            "mode": None,
            "weight":0.333
        }]
    
    elif args.model == 'ens_2':

        model_1 = IncResV2.inceptionresnetv2(num_classes=110).to(device)
        model_1.load_state_dict(torch.load("model_log/incres_adv_pgd.pth", map_location=device_name))

        model_2 = resnet.resnet34(num_classes=110).to(device)
        model_2.load_state_dict(torch.load("model_log/res_normal_64.pth", map_location=device_name))

        model_3 = IncResV2.inceptionresnetv2(num_classes=110).to(device)
        model_3.load_state_dict(torch.load("model_log/incres_adv_mi.pth", map_location=device_name))

        model_4 = densenet.densenet169(num_classes=110).to(device)
        model_4.load_state_dict(torch.load("model_log/des_normal_63.pth", map_location=device_name))
        
        models = [{
            "model": model_1,
            "ratio": 1.,
            "mode": None,
            "weight":0.333
        }, {
            "model": model_2,
            "ratio": 1.,
            "mode": None,
            "weight":0.333
        }, {
            "model": model_3,
            "ratio": 1,
            "mode": None,
            "weight":0.333
        }, {"model": model_4,
            "ratio": 1.,
            "mode": None,
            "weight":0.333
        }]

    else:
        print("No such models!")
        exit()


    data = TianchiDataset(os.path.join(args.data, "dev.csv"), args.data,
                            transform=transforms.Compose([ToTensor()]))

    loader = DataLoader(data, batch_size=10, shuffle=False)


    errs = [0]*len(models)
    t_errs = [0]*len(models)

    l_2 = 0

    
    for idx , data in enumerate(loader):

        x = data['image'].to(device).type(torch.float32)/255.
        y = data['label'].to(device).type(torch.long)
        t = data['target'].to(device).type(torch.long)
        path = data['file']
    
        # x = 5*torch.randn(*x.size()).to(device)/255 + x

        if args.attack == 'fgs':
            adv = torch.clamp(_fgs(models, x, t, args.epsilon),0,1)*255
        elif args.attack == 'pgd':
            adv = torch.clamp(_pgd(models, x, y, args.epsilon, niters=7, alpha=100),0,1)*255
        elif args.attack == 'mi':
            adv = torch.clamp(_mi_fgm(models, x, y, args.epsilon, niters=5),0,1)*255
        elif args.attack == 'mi_t':
            adv = torch.clamp(_mi_fgm_t(models, x, t, args.epsilon, niters=10),0,1)*255
        else:
            adv = x*255

        print(x.size())
        with torch.no_grad():
            index = 0
            for it in models:
                if it["mode"] is None:
                    errs[index] = errs[index] + ((it["model"](adv/255).max(1)[1] != y).float().sum()/adv.size(0)).item()
                    t_errs[index] += ((it["model"](adv/255).max(1)[1] == t).float().sum()/adv.size(0)).item()
                elif it["mode"] == 0:
                    errs[index] += ((it["model"](adv / 255)[0].max(1)[1] != y).float().sum() / adv.size(0)).item()
                    t_errs[index] += ((it["model"](adv / 255)[0].max(1)[1] == t).float().sum() / adv.size(0)).item()
                index = index + 1

        for i  in range(x.size(0)):
            io.imsave(os.path.join(output_path,path[i]), torch.clamp(adv[i].cpu().type(torch.uint8),0,255).detach().numpy().transpose((1,2,0)))
            l_2 = np.mean(np.sqrt(np.sum((adv[i].cpu().detach().numpy().reshape((-1,3))-255.00*x[i].cpu().detach().numpy().reshape(-1,3))**2, axis=1)))
            print(l_2)
    
    l_2 = l_2 / len(loader)

    for i in range(len(errs)):
        errs[i] = errs[i] / len(loader)
    for i in range(len(t_errs)):
        t_errs[i] = t_errs[i] / len(loader)


    end = time.time()
    print('Time cost:{}'.format(end-start))
    # print('Non target rate: {0:.2f} target rate: {1:.2f}'.format(err_1, t_err_1))
    for i in range(len(errs)):
        print('Non target rate: {0:.2f} target rate: {1:.2f}'.format(errs[i], t_errs[i]))

    print("Size of Perturbation: {}".format(l_2))


    

    
            

        



    

    

    
    
