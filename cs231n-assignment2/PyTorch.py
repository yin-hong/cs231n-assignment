import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import  torchvision.datasets as dset
from torch.utils.data import sampler
import torchvision.transforms as T
import numpy as np
import copy
import timeit
class ChunkSampler(sampler.Sampler):

    def __init__(self,num_samples,start=0):
        self.num_samples=num_samples
        self.start=start

    def __iter__(self):
        return iter(range(self.start,self.start+self.num_samples))

    def __len__(self):
        return self.num_samples

NUM_TRAIN=49000
NUM_VAL=1000
cifar10_train=dset.CIFAR10('/home/hongyin/file/',train=True,transform=T.ToTensor(),download=False)
load_train=DataLoader(cifar10_train,batch_size=64,sampler=ChunkSampler(NUM_TRAIN,0))
cifar10_val=dset.CIFAR10('/home/hongyin/file/',train=False,transform=T.ToTensor(),download=False)
load_val=DataLoader(cifar10_val,batch_size=64,sampler=ChunkSampler(NUM_VAL,NUM_TRAIN))
cifar10_test=dset.CIFAR10('/home/hongyin/file/',train=True,transform=T.ToTensor(),download=False)
load_test=DataLoader(cifar10_test,batch_size=64)

datatype=torch.FloatTensor
print_every=100
def reset(m):
    if hasattr(m,'reset'):
        m.reset_parameters()

class Flatten(nn.Module):
    def forward(self,x):
        N,C,H,W=x.size()
        return x.view(N,-1)


simple_model=nn.Sequential(
    nn.Conv2d(3,32,7,2),
    nn.ReLU(inplace=True),
    Flatten(),
    nn.Linear(5408,10)
)
#simple_model.type(datatype)
#loss_fn=nn.CrossEntropyLoss().type(datatype)
#optimizer=optim.Adam(simple_model.parameters().parameters(),lr=1e-2)


fix_model_base=nn.Sequential(
    nn.Conv2d(3,32,kernel_size=7,stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(2,2),
    Flatten(),
    nn.Linear(13*13*32,1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024,10)
)
"""
fix_model_base.type(datatype)
loss_fn=nn.CrossEntropyLoss().type(datatype)
optimizer=optim.RMSprop(fix_model_base.parameters())

x=torch.randn(64,3,32,32).type(datatype)
x=Variable(x)
ans=fix_model_base(x)

result=np.array_equal(np.array(ans.size()),np.array([64,10]))
print(result)
"""
"""
gpu_dtype=torch.cuda.FloatTensor
if torch.cuda.is_available():
    print('is available')
    fix_model_base.type(gpu_dtype)
    x=torch.randn(64,3,32,32).type(gpu_dtype)
    x=Variable(x)
    ans=fix_model_base(x)
    result=np.array_equal(np.array(ans.size()),np.array([64,10]))
    print(result)
"""
print('it is available')
gpu_dtype = torch.cuda.FloatTensor
fix_model_base_gpu = copy.deepcopy(fix_model_base).type(gpu_dtype)
x_gpu_type = torch.randn(64, 3, 32, 32).type(gpu_dtype)
x_gpu_type = Variable(x_gpu_type)
ans = fix_model_base_gpu(x_gpu_type)
result = np.array_equal(np.array(ans.size()), np.array([64, 10]))
print(result)




loss_fn=nn.CrossEntropyLoss().type(gpu_dtype)
optimizer=optim.RMSprop(fix_model_base_gpu.parameters(),lr=1e-3)

#fix_model_base_gpu.train()
"""
for i,data in enumerate(load_train):
    x,y=data
    x=Variable(x.type(gpu_dtype))
    y=Variable(y.type(gpu_dtype).long())
    scores=fix_model_base_gpu(x)
    loss=loss_fn(scores,y)
    if (i+1)%100==0:
        print(i,loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""

def train(model,optimizer,loss_fn,num_epoch=1):
    for epoch in range(num_epoch):
        print('Starting epoch %d / %d'%(epoch+1,num_epoch))
        model.train()
        for t,data in enumerate(load_train):
            x,y=data
            x=Variable(x.type(gpu_dtype))
            y=Variable(y.type(gpu_dtype).long())
            scores=model(x)
            loss=loss_fn(scores,y)
            if (t+1)%print_every==0:
                print('t = %d, loss = %.4f'%(t+1,loss.data[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



def check_accuracy(model,loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')

    else:
        print('Checking accuracy on test set')

    num_correct=0
    num_samples=0
    model.eval()
    for t,data in enumerate(loader):
        x,y=data
        x_var=Variable(x.type(gpu_dtype),volatile=True)
        scores=model(x_var)
        _,preds=scores.data.cpu().max(1)
        num_correct=num_correct+(y==preds).sum()
        num_samples=num_samples+y.size()[0]
    acc=float(num_correct)/num_samples
    print('Got %d / %d correct (%.2f)'%(num_correct,num_samples,100*acc))



torch.cuda.random.manual_seed(12345)
fix_model_base_gpu.apply(reset)
train(fix_model_base_gpu,optimizer,loss_fn,num_epoch=1)
check_accuracy(fix_model_base_gpu,load_val)



