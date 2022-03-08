import torch
from torch import nn
from dataloader import getDataloader
import argparse
from encoder import Darknet
from decoder import Pose2DLayer
from CNNFunctions import Trainer
from correspondenceNet import correspondenceNet
################################################

def main(args):
    """ Train the network"""

    # TODO: Allow to load and saved trained model
    # TODO: Incorporate Tensorboard?

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_dataloader, test_dataloader, eval_dataloader =  getDataloader()

    # Create new model
    print("-> Create New Model...")
    encoder = Darknet()
    decoder = Pose2DLayer()
    CNN = correspondenceNet(encoder, decoder)

    # Move everything to device
    if torch.cuda.is_available():
        CNN = nn.DataParallel(CNN)
        torch.multiprocessing.set_start_method('spawn')
    CNN = CNN.to(device)

################################################

    # Optimize model
    trainer = Trainer(config="configs/configuration.yml", train_dataloader=train_dataloader, test_dataloader = test_dataloader,
                  eval_dataloader = eval_dataloader, CNN = CNN, device = device)
    trainer.test()
    trainer.train()

################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    main(args)