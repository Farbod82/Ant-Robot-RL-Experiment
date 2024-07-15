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
        self.memorySize = 150
        self.model = model
        self.optimizer = optimizer
        self.losses = []
        self.memory = np.array((self.memorySize, 28+28+8))
        self.memoryCounter = 0
    
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
        inTnesor = torch.tensor(state + actions)
        inTnesor = torch.nn.Parameter(inTnesor, requires_grad=True)
        optim = torch.optim.SGD([inTnesor], lr=1e-1)
        for step in range(15):
            # Forward pass
            # inTnesor = torch.tensor(state + x.detach().tolist())
            output = self.model(inTnesor)
            # Compute the loss (negative of the output)
            loss = -output.mean()

            # Backpropagate the gradients
            loss.backward()
            # Update the first num_optimized_inputs of the input tensor
            optim.step()
            optim.zero_grad()
            # print(inTnesor)

        return inTnesor.detach()[28:]      

                
