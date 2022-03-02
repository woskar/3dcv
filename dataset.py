import os
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AutonomousDrivingDataset(Dataset):
    '''Dataset for the >Peking University/Baidu - Autonomous Driving< - Challenge.'''

    def __init__(self, image_data, mask_data, csv_file, camera_matrix, car_models, transform=None):
        '''Initializes the Autonomous Driving Dataset.
            args:
                image_data      :   [string]    path to the image data 
                mask_data       :   [string]    path to the image masks
                csv_file        :   [string]    path to csv file containing imageID-predictionstring pairs
                camera_matrix   :   [list]      list of intrinsic camera parameters of form [fx, fy, cx, cy]
                car models      :   [string]    path to the car models
            kwargs:
                transform       :   [list]      list of transformations to apply to each datapoint
        '''
        self.image_path         = image_data
        self.mask_path          = mask_data
        self.prediction_frame   = pd.read_csv(csv_file)
        self.intrinsic_matrix   = torch.tensor([[camera_matrix[0], 0, camera_matrix[2]],
                                            [0, camera_matrix[1], camera_matrix[3]],
                                            [0, 0, 1]])
        self.car_models         = car_models
        if not transform is None:
            self.transform          = transforms.Compose(transform)

    def __len__(self):
        return len(self.prediction_frame)

    def string2sixD(self, string):
        '''Transforms the prediction string from the prediction frame to a 6d torch tensor
            containing of the form [yaw, pitch, roll, x, y, z].
            args:
                string          : [string]      string from the prediction frame
        '''

        split       = string.split()
        sixD_coords = torch.tensor([float(x) for x in split])

        return sixD_coords
        
    def __getitem__(self, idx):
        '''Returns an Tensor-Image - Tensor-6d-coordinates pair.
            args:  
                idx         :   [int]       index of the instance
        '''
        if torch.is_tensor(idx):
            idx     = idx.tolist()
        
        img_path    = os.path.join(self.image_path, self.prediction_frame.iloc[idx, 0] + ".jpg")
        
        y           = self.string2sixD(self.prediction_frame.iloc[idx, 1])
        
        X           = Image.open(img_path)
        if not self.transform is None:
            X           = self.transform(X)

        return X, y

    

IMAGE_DATA  = "./data/test_images"
MASK_DATA   = "./data/test_masks"
CSV_FILE    = "./data/sample_submission.csv"
CAM_MTX     = [1,1,1,1]
CAR_MODELS  = "./data/car_models"

Dataset     = AutonomousDrivingDataset(IMAGE_DATA, MASK_DATA, CSV_FILE, CAM_MTX, CAR_MODELS, transform=[transforms.ToTensor()])
Dataset.__getitem__(1)
