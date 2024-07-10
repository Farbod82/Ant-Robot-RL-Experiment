import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import numpy as np

class BrainModel(nn.Module):
    def __init__(self):
        super(BrainModel, self).__init__()
        self.fc1 = nn.Linear(8+28, 64)
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

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
    

class Brain():
    def __init__(self,model,optimizer):
        self.model = model
        self.optimizer = optimizer
        self.losses = []
    
    def PredictReward(self,x):
        return self.model(torch.tensor(x))
    
    def UpdateParameters(self,A,Ahat):
        self.model.train()
        self.optimizer.zero_grad()
        lossd = self.model.compute_loss(A, torch.tensor(Ahat))
        loss = lossd['loss']
        self.losses.append(loss.item())
        loss.backward()
        self.optimizer.step()

    def TryActions(self,actions,state):
        state = state.tolist()
        new_actions = actions
        reward = self.model(torch.tensor(state + actions))
        for i in range(100):
            ind = random.randint(0,7)
            random_float = np.random.uniform(low=-1.0, high=1.0)
            actions[ind] = random_float
            self.model.eval()
            out = self.model(torch.tensor(state + actions))
            if out > reward:
                new_actions = actions
        # eps = np.random.uniform(low=0,high=1)
        # if (eps < 0.5):
            # new_actions = np.random.uniform(low=-1,high=1,size=8).tolist()
        return new_actions


            
