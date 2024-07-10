from joints import Joint,JointModel
from brain import Brain, BrainModel
import torch
import gym
import pybulletgym 
import numpy as np
import time
import matplotlib.pyplot as plt



class AntRobot():
    
    def __init__(self):
        self.joints = []
        self.brain = None
        self.eps = 0.9
        self.minEps = 0.05
    def InitParts(self):
        for i in range(8):
            model = JointModel()
            optimizer = torch.optim.AdamW(model.parameters(),lr = 0.001)
            joint = Joint(model,optimizer)
            self.joints.append(joint)
        b_model = BrainModel()
        b_optimizer = torch.optim.AdamW(model.parameters(),lr = 0.001)
        self.brain = Brain(b_model,b_optimizer)
    
    def RunRobot(self):
        env = gym.make('AntPyBulletEnv-v0')
        env.render(mode='human')
        state = env.reset()
        while True:
            action = []
            actions_update = []
            eps = np.random.uniform(low=0,high=1)
            if (eps < self.eps):
                action = np.random.uniform(low=-1,high=1,size=8).tolist()
            else:
                for i in range(8):
                    acti = self.joints[i].giveAction(state)
                    action.append(acti.item())
                    actions_update.append(acti)
            reward_pred = self.brain.PredictReward(state.tolist()+action)
            new_state, reward, done, info = env.step(action)
            # print(f"Reward: {reward:.2f}")
            self.brain.UpdateParameters(reward_pred,reward)
            actions_bar = self.brain.TryActions(action,state)
            if (eps >= self.eps):
                for i in range(8):
                    self.joints[i].UpdateParameters(actions_bar[i],actions_update[i])
            state = new_state
            env.render()  # Render the environment
            # time.sleep(0.1)  # Wait a bit before the next step
            if done:
                print("Episode finished.")
                self.eps *= 0.95
                if self.eps < self.minEps:
                    self.eps = self.minEps
                state = env.reset()
                x = [i for i in range(len(self.brain.losses))]
                print(len(x))
                # time.sleep(0.5)
                # if len(x) % 1000 == 0:
                    # print("helllllllllllllllllllllllllo")
                if (len(x) > 50000):
                    plt.plot(x, self.brain.losses)
                    plt.show()
                # break
        env.close()
        




ant = AntRobot()
ant.InitParts()
ant.RunRobot()