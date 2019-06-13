import os
import argparse
import time
import torch
import torch.nn as nn
import pandas as pd

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
from ImageLoad import TianchiDataset, RandomCrop, Rescale, ToTensor
from attack import _pgd_train
from models import resnet
from models import densenet
from models import IncResV2
from models import InceptionV4



parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--batch', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--pretrain', type=bool, default=True)
parser.add_argument('--model', type=str, default='res', 
                    help='Used Model: res, inc, incres, dense')
parser.add_argument('--mode', type=str, default='normal',
                    help='normal, noise, adv')
parser.add_argument('--noise', type=float, default=15)
parser.add_argument('--epsilon', type=float, default=35)
args = parser.parse_args()

def accuracy(model, dataset_loader):
    total_correct = 0
    for data in dataset_loader:
        x = data['image'].to(device).type(torch.float32)
        y = data['label'].to(device)
        total_correct += ((model(x).max(1)[1] == y).float().sum()).item()
        
    return total_correct / len(dataset_loader.dataset)


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = TianchiDataset("../train.csv",
                          "../IJCAI_2019_AAAC_train",
                          transform=transforms.Compose([Rescale(400), RandomCrop(299), ToTensor()]))
    test_data = TianchiDataset("../test.csv",
                            "../IJCAI_2019_AAAC_train",
                          transform=transforms.Compose([Rescale(400), RandomCrop(299), ToTensor()]))

    dataloader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=True)
    batches_per_epoch = len(dataloader)
    

    if args.model == 'res':
        model = resnet.resnet34(num_classes=110).to(device)
    elif args.model == 'dense':
        model = densenet.densenet169(num_classes=110).to(device)
    elif args.model == 'incres':
        model = IncResV2.inceptionresnetv2(num_classes=110).to(device)
    elif args.model == 'incv4':
        model = InceptionV4.inceptionv4(num_classes=110).to(device)


    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    best_acc = 0
    length = len(dataloader)
    
    for epoch in range(args.nepochs):
        start_time = time.time()
        print(len(dataloader))
        for i, data in enumerate(dataloader):
            
            model.train()
            optimizer.zero_grad()
            x = data['image'].to(device).type(torch.float32)
            y = data['label'].to(device)
            out = model(x)
            loss = loss_function(out, y)
            loss.backward()
            optimizer.step()

            


            if args.mode == 'adv':
                del loss, out
                optimizer.zero_grad()
                noise = args.noise*torch.randn(*x.size()).to(device)/255
                x_adv = torch.tensor(torch.clamp(_pgd_train(model, x, x+noise, y, args.epsilon, niters=10, alpha=100),0,1))
                del noise
                out = model(x_adv)
                loss = loss_function(out, y)
                loss.backward()
                optimizer.step()

                if i%100 == 0:
                    torch.save(model.state_dict(), 'model_log/'+args.model+'_adv.pth')
            
            correct = ((out.max(1)[1]==y).float().sum()).item()/28
            print("\r{}/{},".format(i+1,length) + str(loss) + 'Acc:{}'.format(correct), end = ' ')

        with torch.no_grad():
            # train_acc = accuracy(model, dataloader)
            model.eval()
            test_acc  = accuracy(model, test_loader)
            if test_acc > best_acc:
                torch.save(model.state_dict(), 'model_log/'+args.model+ "_" +args.mode+'.pth')
                best_acc = test_acc
            
            end_time = time.time()
            print("Time cost:{}m".format((end_time-start_time)/60))
            print("Epoch:{0}, Test Acc:{1}".format(epoch,test_acc))
