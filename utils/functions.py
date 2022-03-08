import json
import pickle
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import warnings
import torch
warnings.filterwarnings("ignore")
from math import sin, cos
from tqdm import trange

################################################

# Converts the ground truth prediction strings to an ordered dictionary
def str_to_coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

# Convert 2D correspondence predictions to clusters
def convertOutputToClusters(px, py, conf):
    px = px.reshape(px.shape[0], px.shape[1]*px.shape[2], px.shape[3])
    py = py.reshape(py.shape[0], py.shape[1] * py.shape[2], py.shape[3])
    conf = conf.reshape(conf.shape[0], conf.shape[1] * conf.shape[2], conf.shape[3])
    points = torch.stack([px, py, conf], dim=1)
    points = torch.swapaxes(points, 1, 3)
    return points

# Converts the ground truth prediction dictionary to a submission string
def coords_to_str(coords):
    s = []
    for c in coords:
        for n in range(7):
            s.append(str(c[n]))
    return ' '.join(s)

# Create rotation matrix from euler angles
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))

# Calculates the ground truth 2D projections of 9 keypoints
def getGroundTruth2DProjections(img, vehicleFile, yaw, pitch, roll, x, y, z, visFlag = False):

    camera_matrix = np.array([[2304.5479, 0, 1686.2379],
                              [0, 2305.8757, 1354.9849],
                              [0, 0, 1]], dtype=np.float32)

    with open(vehicleFile) as src:
        data = json.load(src)
        vertices = np.array(data['vertices'])
        vertices[:, 1] = -vertices[:, 1]
        triangles = np.array(data['faces']) - 1
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]
    yaw, pitch, roll = -pitch, -yaw, -roll
    Rt = np.eye(4)
    t = np.array([x, y, z])
    Rt[:3, 3] = t
    Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
    Rt = Rt[:3, :]
    P = np.array([[0, 0, 0, 1],
                  [x_l, y_l, -z_l, 1],
                  [x_l, y_l, z_l, 1],
                  [-x_l, y_l, z_l, 1],
                  [-x_l, y_l, -z_l, 1],
                  [x_l, -y_l, -z_l, 1],
                  [x_l, -y_l, z_l, 1],
                  [-x_l, -y_l, z_l, 1],
                  [-x_l, -y_l, -z_l, 1]]).T
    img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
    img_cor_points = img_cor_points.T
    img_cor_points[:, 0] /= img_cor_points[:, 2]
    img_cor_points[:, 1] /= img_cor_points[:, 2]
    x_img, y_img, z_img = img_cor_points[0]
    xc, yc, zc = x_img*z_img, y_img*z_img, z_img
    p_cam = np.array([xc, yc, zc])
    xw, yw, zw = np.dot(np.linalg.inv(camera_matrix), p_cam)
    img_cor_points = img_cor_points.astype(int)
    img_cor_points = img_cor_points[:, :2]

    if visFlag:
        plt.scatter(img_cor_points[:, 0], img_cor_points[:, 1])
        plt.imshow(img)
        plt.show()

    return img_cor_points

# Iterates once over all vehicles in all training images and stores the ground truth 9 points locally
def calculateAndSaveGroundTruth2DPoints():

    images = []
    correspondences = []

    # Load vehicle names (sorted by ID)
    with open("vehicleNames", "rb") as fp:
        vehicleNames = pickle.load(fp)

    PATH = 'data/pku-autonomous-driving/'
    train = pd.read_csv('data/pku-autonomous-driving/train.csv')
    test = pd.read_csv('data/pku-autonomous-driving/sample_submission.csv')

    # For all images
    for i in trange(0, len(train['ImageId'])):
        str = train['PredictionString'][i]
        imageID = train['ImageId'][i]
        coords = str_to_coords(str)
        # For all vehicles in the image
        for j in range(0, len(coords)):
            model_type, yaw, pitch, roll, x, y, z = list(coords[j].values())
            vehicleName = vehicleNames[model_type]
            img = cv2.imread('data/pku-autonomous-driving/' + "/train_images/" + imageID + '.jpg')
            vehicleFile = 'data/pku-autonomous-driving/car_models_json' + "/" + vehicleName + ".json"
            points = getGroundTruth2DProjections(img, vehicleFile, yaw, pitch, roll, x, y, z)
            images.append(imageID)
            correspondences.append(points)

    with open('image_IDS_train', 'wb') as fp:
        pickle.dump(images, fp)

    with open('correspondences_train', 'wb') as fp:
        pickle.dump(correspondences, fp)
