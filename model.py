import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Conv2D(nn.Module):
    def __init__(self, in_size, out_size, kernal_size, padding=0, stride=1, bias=False):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernal_size, padding=padding, stride=stride, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_size, eps=0.001, affine=True)
    
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        
        return F.relu(x)


class BlockA(nn.Module):
    def __init__(self, activation_scale: float=1):
        super(BlockA, self).__init__()
        self.conv1 = Conv2D(256, 32, 1)
        
        self.conv2 = Conv2D(256, 32, 1)
        self.conv3 = Conv2D(32, 32, 3, padding=1)
        
        self.conv4 = Conv2D(256, 32, 1)
        self.conv5 = Conv2D(32, 32, 3, padding=1)
        self.conv6 = Conv2D(32, 32, 3, padding=1)
        
        self.conv7 = nn.Conv2d(96, 256, kernel_size=1)
        self.scale = activation_scale
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        
        y = self.conv2(inputs)
        y = self.conv3(y)
        
        z = self.conv4(inputs)
        z = self.conv5(z)
        z = self.conv6(z)
        
        w = torch.cat([x, y, z], axis=1)
        w = self.conv7(w)
        
        ret = inputs + self.scale * w
        return F.relu(ret)


# In[7]:


class BlockB(nn.Module):
    def __init__(self, activation_scale: float=1):
        super(BlockB, self).__init__()
        self.conv1 = Conv2D(896, 128, 1)
        
        self.conv2 = Conv2D(896, 128, 1)
        self.conv3 = Conv2D(128, 128, (1,7), padding=(0,3))
        self.conv4 = Conv2D(128, 128, (7,1), padding=(3,0))
        
        self.conv5 = nn.Conv2d(256, 896, kernel_size=1)
        self.scale = activation_scale
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        
        y = self.conv2(inputs)
        y = self.conv3(y)
        y = self.conv4(y)
        
        w = torch.cat([x, y], axis=1)
        w = self.conv5(w)
        
        ret = inputs + self.scale * w
        return F.relu(ret)


# In[8]:


class BlockC(nn.Module):
    def __init__(self, activation_scale: float=1, activation=True):
        super(BlockC, self).__init__()
        self.conv1 = Conv2D(1792, 192, 1)
        
        self.conv2 = Conv2D(1792, 192, 1)
        self.conv3 = Conv2D(192, 192, (1,3), padding=(0,1))
        self.conv4 = Conv2D(192, 192, (3,1), padding=(1,0))
        
        self.conv5 = nn.Conv2d(384, 1792, kernel_size=1)
        self.scale = activation_scale
        self.activation = activation
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        
        y = self.conv2(inputs)
        y = self.conv3(y)
        y = self.conv4(y)
        
        w = torch.cat([x, y], axis=1)
        w = self.conv5(w)
        
        ret = inputs + self.scale * w
        
        if self.activation:
            return F.relu(ret)
        else:
            return ret


# In[9]:


class ReductionBlockA(nn.Module):
    def __init__(self, k, l):
        super(ReductionBlockA, self).__init__()
        self.conv1 = Conv2D(256, 384, 3, stride=2)
        
        self.conv2 = Conv2D(256, k, 1)
        self.conv3 = Conv2D(k, l, 3, padding=1)
        self.conv4 = Conv2D(l, 256, 3, stride=2)
        
        self.maxpool = nn.MaxPool2d(3, stride=2)
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        
        y = self.conv2(inputs)
        y = self.conv3(y)
        y = self.conv4(y)
        
        z = self.maxpool(inputs)
        
        w = torch.cat([x, y, z], axis=1)
        
        return w


# In[10]:


class ReductionBlockB(nn.Module):
    def __init__(self):
        super(ReductionBlockB, self).__init__()
        self.conv1 = Conv2D(896, 256, 1)
        self.conv2 = Conv2D(256, 384, 3, stride=2)
        
        self.conv3 = Conv2D(896, 256, 1)
        self.conv4 = Conv2D(256, 256, 3, stride=2)
        
        self.conv5 = Conv2D(896, 256, 1)
        self.conv6 = Conv2D(256, 256, 3, padding=1)
        self.conv7 = Conv2D(256, 256, 3, stride=2)
        
        self.maxpool = nn.MaxPool2d(3, stride=2)
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        
        y = self.conv3(inputs)
        y = self.conv4(y)
        
        z = self.conv5(inputs)
        z = self.conv6(z)
        z = self.conv7(z)
        
        w = self.maxpool(inputs)
        
        q = torch.cat([x, y, z, w], axis=1)
        
        return q


# In[11]:


class InceptionresnetV1(nn.Module):
    def __init__(self):
        super(InceptionresnetV1, self).__init__()
        self.stem = nn.Sequential(
                    Conv2D(3, 32, 3, stride=2),
                    Conv2D(32, 32, 3),
                    Conv2D(32, 64, 3, padding=1),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    Conv2D(64, 80, 1),
                    Conv2D(80, 192, 3),
                    Conv2D(192, 256, 3, stride=2))
        
        self.A = nn.Sequential(*[BlockA(0.17) for _ in range(5)])
        self.r_AB = ReductionBlockA(192, 192)
        self.B = nn.Sequential(*[BlockB(0.1) for _ in range(10)])
        self.r_BC = ReductionBlockB()
        self.C = nn.Sequential(*[BlockC(0.2) if i != 5 else BlockC(1, False) for i in range(6)])
        
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AvgPool2d(8)
        self.flatten = nn.Flatten()
        self.mlp = nn.Linear(1792, 512, bias=False)        
    
    def forward(self, x):
        x = self.stem(x)
        x = self.A(x)
        x = self.r_AB(x)
        x = self.B(x)
        x = self.r_BC(x)
        x = self.C(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.mlp(x)
        
        return F.normalize(x, p=2, dim=1)

