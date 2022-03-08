import sys
import torch
from torch import optim
from tqdm import trange
import yaml
#################################################

class Trainer:
    def __init__(self, config=None, train_dataloader=None, test_dataloader=None, eval_dataloader=None, CNN=None, device = None):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.eval_dataloader = eval_dataloader
        self.network = CNN
        self.cfg = yaml.load(open(config), Loader=yaml.SafeLoader)
        self.device = device

#################################################

def train(self):
    self.network.train()
    optimizer = optim.Adam(self.network.parameters(), lr = self.cfg["lr"])
    for epoch in range(0, self.cfg["lr"]):
        print(" Epoch:  ", epoch)
        iterator = iter(self.train_dataloader)
        for index in trange(0, len(self.train_dataloader)):
            img, points2D, points3D, DoFs = iterator.next()
            img, points2D, points3D, DoFs = img.to(self.device), points2D.to(self.device), points3D.to(self.device), DoFs.to(self.device)
            optimizer.zero_grad()
            px, py, conf = self.network(img)
            predictedPoints = torch.stack([px, py], dim=1)
            error = predictedPoints - points2D
            Loss_pos = torch.abs(error)
            Loss_conf = torch.abs(conf - torch.exp(self.cfg["tau"] * torch.norm(error)))
            Loss = self.cfg["beta"] * Loss_pos + self.cfg["lbd"] * Loss_conf
            Loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            test()

################################################

def test(self):
    Losses = []
    self.network.eval()
    iterator = iter(self.test_dataloader)
    for index in range(0, len(self.test_dataloader)):
        img, points2D, points3D, DoFs = iterator.next()
        img, points2D, points3D, DoFs = img.to(self.device), points2D.to(self.device), points3D.to(
            self.device), DoFs.to(self.device)
        px, py, conf = self.network(img)
        predictedPoints = torch.stack([px, py], dim=1)
        error = predictedPoints - points2D
        Loss_pos = torch.abs(error)
        Loss_conf = torch.abs(conf - torch.exp(self.cfg["tau"] * torch.norm(error)))
        Loss = self.cfg["beta"] * Loss_pos + self.cfg["lbd"] * Loss_conf
        Losses.append(float(Loss))
    print("Test Loss: ", torch.mean(Losses))

################################################

def eval():
    return None



