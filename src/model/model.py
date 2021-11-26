if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from base import BaseModel
# import torchvision.models as models
from model.polyconv import network
from utils.util import apply_funcs
from torch.autograd import Variable
device = torch.device("cuda")

class Shape3DModel(BaseModel):
    def __init__(self, num_classes, model):
        super().__init__()
        self.num_classes = num_classes
        self.model = model


        if model == "polynet":
            self.Conv1_p = torch.nn.Parameter(0.01*torch.randn(6,6))
            self.Conv1_w = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=[1,1])

            self.Conv2_p = torch.nn.Parameter(0.01*torch.randn(64+6,6))
            self.Conv2_w = nn.Conv2d(in_channels=64+6, out_channels=128, kernel_size=[1,1])

            self.Conv3_p = torch.nn.Parameter(0.01*torch.randn(128+6,6))
            self.Conv3_w = nn.Conv2d(in_channels=128+6, out_channels=256, kernel_size=[1,1])

            self.Conv4_p = torch.nn.Parameter(0.01*torch.randn(256+6,6))
            self.Conv4_w = nn.Conv2d(in_channels=256+6, out_channels=512, kernel_size=[1,1])
            



            self.fc6 = nn.Linear(512, 1024)
            self.fc7 = nn.Linear(1024, 1024)
            self.fc8 = nn.Linear(1024, self.num_classes)

            self.in1 = nn.InstanceNorm1d(64,affine=True)
            self.in2 = nn.InstanceNorm1d(128,affine=True)
            self.in3 = nn.InstanceNorm1d(256,affine=True)
            self.in4 = nn.InstanceNorm1d(512,affine=True)
            
            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(1024)
            self.bn3 = nn.BatchNorm1d(num_classes)
            
            self.drop1 = nn.Dropout(p=0.5)
            self.drop2 = nn.Dropout(p=0.5)

        else:
            print("Check model name in config.json")
            raise

    def forward(self, input,adj1,adj2,adj3,adj4,c1,c2,c3,c4,ver_num,task):
        def _closeness_constraint(netcoeff):  # (num_funcs, 6)
            # 6 parameters
            B = torch.zeros((netcoeff.shape[0], 3, 3), device='cuda')
            triu_idcs = torch.triu_indices(row=3, col=3, offset=0).to('cuda')
            B[:, triu_idcs[0], triu_idcs[1]] = netcoeff  # vector to upper triangular matrix
            B[:, triu_idcs[1], triu_idcs[0]] = netcoeff  # B: symm. matrix
            A = torch.bmm(B, B)  # A = B**2  // A: symm. positive definite (num_funcs, 3,3)

            # [1, x, y, x^2, xy, y^2] 
            p4coeff = torch.zeros((netcoeff.shape[0], 6), device='cuda')
            p4coeff[:, 0] = A[:, 0,0]  # 1
            p4coeff[:, 3] = A[:, 1,1]  # x^2
            p4coeff[:, 5] = A[:, 2,2]  # y^2

            p4coeff[:, 1] = A[:, 1,0]+A[:, 0,1]  # x
            p4coeff[:, 2] = A[:, 2,0]+A[:, 0,2]  # y
            p4coeff[:, 4] = A[:, 1,2]+A[:, 2,1]  # xy
            return p4coeff


        if self.model == "polynet":
            Conv1_p = _closeness_constraint(self.Conv1_p)
            Conv2_p = _closeness_constraint(self.Conv2_p)
            Conv3_p = _closeness_constraint(self.Conv3_p)
            Conv4_p = _closeness_constraint(self.Conv4_p)


            CONV = [Conv1_p,self.Conv1_w,Conv2_p,self.Conv2_w,Conv3_p,self.Conv3_w,Conv4_p,self.Conv4_w]
            IN = [self.in1,self.in2,self.in3,self.in4]

            x = network(input,adj1.long(),adj2.long(),adj3.long(),adj4.long(),c1.long(),c2.long(),c3.long(),c4.long(),ver_num.long(), CONV, IN)


            x = self.drop1(torch.relu(self.bn1(self.fc6(x))))
            x = self.drop2(torch.relu(self.bn2(self.fc7(x))))
            x = torch.log_softmax(self.bn3(self.fc8(x)),dim=-1)
   


        return x # [batch, num_classes]


if __name__ == "__main__":

    netcoeff = torch.randn((64, 21, 16)).to('cuda')




