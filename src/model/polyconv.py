import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_scatter import scatter_max, scatter_mean, scatter_sum
import numpy as np

dtype = torch.double
device = torch.device("cuda")



def conv(Input,Adj,C,conv_num,b,conv_p,conv_w):
    y = Input[:,Adj[b, :conv_num[1]]].unsqueeze(-1)
    x = Input[:,C[b, :conv_num[1]]].unsqueeze(-1)
    tmp = Input[:,:conv_num[0]].unsqueeze(-1)


    fxy = (torch.cat([torch.ones_like(y),x,y,torch.pow(x,2),x*y,torch.pow(y,2)],-1).unsqueeze(-1))*conv_p.unsqueeze(1).unsqueeze(-1)
    yfxy = fxy * (y.unsqueeze(2))
    yfxy = torch.sum(scatter_mean(yfxy,C[b,:conv_num[1]],dim=1).unsqueeze(0),dim=3)
    

    x_prime  = torch.cat([2*torch.ones_like(tmp),2*tmp,torch.zeros_like(tmp),2*torch.pow(tmp,2),torch.zeros_like(tmp),(2/3)*torch.ones_like(tmp)],-1).unsqueeze(-1)
    fx  = x_prime*(conv_p.unsqueeze(1).unsqueeze(-1))
    fx = torch.sum(fx,dim=2)

    fxy_cond = torch.true_divide(yfxy,fx)
    fxy_cond = conv_w(fxy_cond)[:,:,:,0]

    return fxy_cond


def pool_max(Input, Adj, C, pool_num, conv_num,b):
    x = Input[:,Adj[b, :conv_num[1]]]
    x = scatter_max(x,C[b,:conv_num[1]],dim=1)[0]
    x = x[:, :pool_num[0]].float()
    return x

def pool_mean(Input, Adj, C, pool_num, conv_num,b):
    x = Input[:,Adj[b, :conv_num[1]]]
    x = scatter_mean(x,C[b,:conv_num[1]],dim=1)
    x = x[:, :pool_num[0]].float()
    return x


def network(Input,adj1,adj2,adj3,adj4,c1,c2,c3,c4,ver_num,CONV, IN):
    Input = torch.transpose(Input,2,1)
    Output = []

    for b in range(Input.shape[0]):
        # First layer
        x1 = torch.tanh(IN[0](conv(Input[b,:,:].float(),adj1,c1,ver_num[b,:,0],b,CONV[0],CONV[1])))[0,:,:]
        x1_pool = torch.cat([Input[b,:,:ver_num[b,0,1]].float(),pool_max(x1, adj1, c1[:,:], ver_num[b,:,1], ver_num[b,:,0], b)],dim=0)
        
        # Second layer
        x2 = torch.tanh(IN[1](conv(x1_pool,adj2,c2,ver_num[b,:,1],b,CONV[2],CONV[3])))[0,:,:]
        x2_pool = torch.cat([Input[b,:,:ver_num[b,0,2]].float(),pool_max(x2, adj2, c2[:,:], ver_num[b,:,2], ver_num[b,:,1], b)],dim=0)
        
        # Third layer
        x3 = torch.tanh(IN[2](conv(x2_pool,adj3,c3,ver_num[b,:,2],b,CONV[4],CONV[5])))[0,:,:]
        x3_pool = torch.cat([Input[b,:,:ver_num[b,0,3]].float(),pool_max(x3, adj3, c3[:,:], ver_num[b,:,3], ver_num[b,:,2], b)],dim=0)
        
        # Forth layer
        x4 = torch.tanh(IN[3](conv(x3_pool,adj4,c4,ver_num[b,:,3],b,CONV[6],CONV[7])))[0,:,:]


        Output.append(torch.mean(x4,dim=1).unsqueeze(0))
    
    Output = torch.cat(Output,dim=0)
    return Output.float()
