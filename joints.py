import torch.nn as nn
import torch
import torch.nn.functional as F

class JointModel(nn.Module):
    def __init__(self):
        super(JointModel, self).__init__()
        self.fc1 = nn.Linear(28, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32,1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x

    def compute_loss(self, xc, yc):
        loss = F.mse_loss(xc,yc)

        return {
            'loss': loss 
        }

class Joint():
    def __init__(self,model,optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss = []
    
    def giveAction(self,x):
        return self.model(torch.tensor(x))
    
    def UpdateParameters(self,A,Ahat):
        self.model.train()
        self.optimizer.zero_grad()
        lossd = self.model.compute_loss(torch.tensor(A), Ahat)
        loss = lossd['loss']
        self.loss.append(loss.item())
        loss.backward()
        self.optimizer.step()




