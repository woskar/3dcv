import pickle
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.functions import str_to_coords

##############################################################

class DataSetTrain():
    """ Dataset instance"""
    def __init__(self, imgID, vehicleIndex, boxes, points2D, points3D):
        self.imgID = imgID
        self.vehicleIndex = vehicleIndex
        self.boxes = boxes
        self.points2D = points2D
        self.points3D = points3D
        self.train = pd.read_csv('data/pku-autonomous-driving/train.csv')

    def __getitem__(self, index):
        # TODO: Vorher points2D und points3D erstellen
        imgID = self.imgID[index]
        vehicleIndex = self.vehicleIndex[index]
        str = str(self.train.loc[self.train['ImageId'] == imgID]['PredictionString'])
        coords = str_to_coords(str)
        model_type, yaw, pitch, roll, x, y, z = list(coords[vehicleIndex].values())
        DoFs = [yaw, pitch, roll, x, y, z]
        img = cv2.imread('data/pku-autonomous-driving/' + "/train_images/" + imgID + '.jpg')
        box = self.boxes[index]
        sx, sy, ex, ey = box
        points2D = self.points2D[index]
        points3D = self.points3D[index]
        # Apply binary map
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[sx:ex, sy:ey] = 255
        img = cv2.bitwise_and(img, img, mask=mask)
        # TODO: Resizing???
        return torch.tensor(img), torch.tensor(points2D), torch.tensor(points3D), torch.tensor(DoFs)

    def __len__(self):
        return len(self.imagePaths)

##############################################################

class DataSetEval():
    """ Dataset instance"""
    def __init__(self, imgID, vehicleIndex, boxes):
        self.imgID = imgID
        self.vehicleIndex = vehicleIndex
        self.boxes = boxes

    def __getitem__(self, index):
        imgID = self.imgID[index]
        vehicleIndex = self.vehicleIndex[index]
        img = cv2.imread('data/pku-autonomous-driving/' + "/train_images/" + imgID + '.jpg')
        box = self.boxes[index]
        sx, sy, ex, ey = box
        # Apply binary map
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[sx:ex, sy:ey] = 255
        img = cv2.bitwise_and(img, img, mask=mask)
        # TODO: Resizing???
        return torch.tensor(img)

    def __len__(self):
        return len(self.imagePaths)

##############################################################

def getDataloader():
    """ Create train and test dataloader"""

    with open("image_IDS_train", "rb") as fp:
        image_IDS_train = pickle.load(fp)

    with open("correspondences_train", "rb") as fp:
        correspondences = pickle.load(fp)

    # TODO: Create the actual lists
    imgID, vehicleIndex, boxes, points2D, points3D = None, None, None, None, None

    trainLimit = int(0.9*len(image_IDS_train))
    testLimit = len(image_IDS_train) - trainLimit

    imgID_train = imgID[:trainLimit]
    imgID_test = imgID[-testLimit:]

    vehicleIndex_train = vehicleIndex[:trainLimit]
    vehicleIndex_test = vehicleIndex[-testLimit:]

    boxes_train = boxes[:trainLimit]
    boxes_test = boxes[-testLimit:]

    points2D_train = points2D[:trainLimit]
    points2D_test = points2D[-testLimit:]

    points3D_train = points3D[:trainLimit]
    points3D_test = points3D[-testLimit:]

    dataset_Train = DataSetTrain(imgID_train, vehicleIndex_train, boxes_train, points2D_train, points3D_train)
    dataset_Test = DataSetTrain(imgID_test, vehicleIndex_test, boxes_test, points2D_test, points3D_test)
    dataset_Eval = DataSetEval(imgID_test, vehicleIndex_test, boxes_test)

    dataloader_Train = DataLoader(dataset=dataset_Train)
    dataloader_Test = DataLoader(dataset=dataset_Test)
    dataloader_Eval = DataLoader(dataset=dataset_Eval)

    return dataloader_Train, dataloader_Test, dataloader_Eval
