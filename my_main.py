# to only test the model so that can achieve on github
import argparse
import shutil
import os
import time
import warnings
import torch.nn as nn
import torch.utils.data
from resnet_models import resnet19
from pathlib import Path
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
# from utils import *
import xlwt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--dts', type=str, default='CIFAR10')
parser.add_argument('--model', type=str, default='MSNN')
parser.add_argument('--beta', type = float, default = 1e-4)
parser.add_argument('--result_path', type = Path, default = './result' )
parser.add_argument('--T', type=int, default=2)
opt = parser.parse_args()

torch.cuda.set_device(opt.gpu)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.backends.cudnn.deterministic = True



transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# 1.to do: dataset location
test_dataset = dsets.CIFAR10(root=, train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False,drop_last = True,num_workers = 4)
# 2.to do: .pth location
pretrained_model = 
model = resnet19()
model.T = opt.T
model.to(device)
model.load_state_dict(torch.load(pretrained_model))


model.eval()
total = 0
correct = 0
with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        out = model(inputs)
        out = torch.mean(out, dim=1)
        pred = out.max(1)[1]
        total += targets.size(0)
        correct += (pred ==targets).sum()
    acc = 100.0 * correct.item() / total
    
    print('Test correct: %d Accuracy: %.2f%%' % (correct, acc))

