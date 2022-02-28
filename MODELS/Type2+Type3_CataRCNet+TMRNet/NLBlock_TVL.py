import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
from torch.nn import DataParallel
import numpy as np

LB_sequnce_length = 30


# temporal varition layer    
class TVL(nn.Module):
    def __init__(self):
        super(TVL, self).__init__()
        self.TVL1 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.TVL2 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.TVL3 = nn.Conv1d(512, 512, kernel_size=7, padding=3)
        self.maxpool_4 = nn.MaxPool1d(2, stride=1)
        self.maxpool = nn.AdaptiveMaxPool2d((512,1))

    def forward(self, Lt): # long feature 
        Lt = Lt.transpose(1, 2)
        
        y0 = Lt.transpose(1, 2)
        y0 = y0.view(-1,LB_sequnce_length,512,1)
        
        # conv k=3
        x1 = self.TVL1(Lt)          
        y1 = x1.transpose(1, 2) 
        y1 = y1.view(-1,LB_sequnce_length,512,1)
        
        # conv k=5
        x2 = self.TVL2(Lt)
        y2 = x2.transpose(1, 2)
        y2 = y2.view(-1,LB_sequnce_length,512,1)
        
        # conv k=7
        x3 = self.TVL3(Lt)
        y3 = x3.transpose(1, 2)
        y3 = y3.view(-1,LB_sequnce_length,512,1)

        # max k =2
        x4 = F.pad(Lt, (1,0), mode='constant', value=0)
        x4 = self.maxpool_4(x4)
        y4 = x4.transpose(1, 2)
        y4 = y4.view(-1,LB_sequnce_length,512,1)


        y = torch.cat((y0,y1,y2,y3,y4), dim=3)  # concatenation 
        y = self.maxpool(y)
        y = y.view(-1,LB_sequnce_length,512)
        
        return y




# non-local block
class NLBlock(nn.Module):
    def __init__(self):
        super(NLBlock, self).__init__()
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 512)
        self.layer_norm = nn.LayerNorm([1, 512])
        self.dropout = nn.Dropout(0.2)

        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)
        init.xavier_uniform_(self.linear4.weight)

    # ct current feature 1*512, Lt long feature bank L*512
    def forward(self, ct, Lt):  
        ct_1 = ct.view(-1, 1, 512)
        ct_1 = self.linear1(ct_1)
        
        Lt_1 = self.linear2(Lt)
        Lt_1 = Lt_1.transpose(1, 2)
        
        # multiply
        SL = torch.matmul(ct_1, Lt_1)       
        SL = SL * ((1/512)**0.5)
        SL = F.softmax(SL, dim=2)
        
        Lt_2 = self.linear3(Lt)
        SLL = torch.matmul(SL, Lt_2)
        # layer norm
        SLL = self.layer_norm(SLL)
        SLL = F.relu(SLL)
        SLL = self.linear4(SLL)
        SLL = self.dropout(SLL)
        SLL = SLL.view(-1, 512) 
        
        nlb =  ct+SLL
        return nlb



