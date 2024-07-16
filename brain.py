import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import numpy as np

class BrainModel(nn.Module):
    def __init__(self):
        super(BrainModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc21 = nn.Linear(51200, 512)
        self.fc22 = nn.Linear(512, 64)
        self.fc1 = nn.Linear(8+28+64, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
    def forward(self, x,xMem):
        xMem = torch.unsqueeze(torch.unsqueeze(xMem, 0), 0)
        x2 = F.relu(self.conv1(xMem))
        x2 = self.pool1(x2)
        x2 = F.relu(self.conv2(x2))
        x2 = self.pool2(x2)
        x2 = x2.view(-1, 51200)
        x2 = F.relu(self.fc21(x2))
        x2 = self.fc22(x2)
        x = x.view(1, -1)
        x = torch.cat((x, x2), dim=1)
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
        self.memorySize = 200
        self.model = model
        self.optimizer = optimizer
        self.losses = []
        self.memory = np.zeros((self.memorySize, 28+28+8+1),dtype="float32")
        self.memoryCounter = 0
    
    def PredictReward(self,x):
        return self.model(torch.tensor(x).to("cuda"),torch.tensor(self.memory).to("cuda")).to("cpu")
    
    def UpdateParameters(self,A,Ahat):
        self.model.train()
        self.optimizer.zero_grad()
        Ahat = torch.tensor([[Ahat]])
        lossd = self.model.compute_loss(A, Ahat)
        loss = lossd['loss']
        self.losses.append(loss.item())
        loss.backward()
        self.optimizer.step()

    def TryActions(self,actions,state):
        inTnesor = torch.tensor(state.tolist() + actions).to("cuda")
        inTnesor = torch.nn.Parameter(inTnesor, requires_grad=True)
        optim = torch.optim.SGD([inTnesor], lr=1e-1)
        for step in range(15):
            # Forward pass
            # inTnesor = torch.tensor(state + x.detach().tolist())
            output = self.model(inTnesor,torch.tensor(self.memory).to("cuda"))
            # Compute the loss (negative of the output)
            loss = -output.mean()

            # Backpropagate the gradients
            loss.backward()
            # Update the first num_optimized_inputs of the input tensor
            optim.step()
            optim.zero_grad()
            # print(inTnesor)

        return inTnesor.to("cpu").detach()[28:]    
    
    def updateMemory(self,memPart):
        medians = np.median(self.memory, axis=0)
        for i in range(self.memorySize):
            self.memory[i,:] = memPart if self.memory[i,:].tolist() == medians.tolist() else self.memory[i,:]
            break
    
    
    

                
