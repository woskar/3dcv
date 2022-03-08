import sys
from utils.functions import *
from encoder import Darknet
from decoder import Pose2DLayer
from CNNFunctions import Trainer
trainer = Trainer(config="configs/configuration.yml")
from correspondenceNet import correspondenceNet

# Just choose an example image and vehicle
# Create ground truth 2D Keypoints using vehicle model
# Instanciate CNN
# Forward non-sense image throughCNN
# Convert output into clusters
def example(testID = 0):

    # Load vehicle names (sorted by ID)
    with open("vehicleNames", "rb") as fp:
        vehicleNames = pickle.load(fp)

    # Choose one example image and vehicle
    # Extraxt ground truth 6DoF
    # Create ground truth 2D keypoints
    train = pd.read_csv('data/pku-autonomous-driving/train.csv')
    str = train['PredictionString'][testID]
    imageID = train['ImageId'][testID]
    coords = str_to_coords(str)
    model_type, yaw, pitch, roll, x, y, z = list(coords[0].values())
    vehicleName = vehicleNames[model_type]
    img = cv2.imread('data/pku-autonomous-driving/' + "/train_images/" + imageID + '.jpg')
    vehicleFile = 'data/pku-autonomous-driving/car_models_json' + "/" + vehicleName + ".json"
    points = getGroundTruth2DProjections(img, vehicleFile, yaw, pitch, roll, x, y, z)

    # Create new CNN
    encoder = Darknet()
    decoder = Pose2DLayer()
    CNN = correspondenceNet(encoder, decoder)

    # Create random image and forward it through CNN
    img = torch.zeros(5, 3, 100, 100).float()
    px, py, conf = CNN(img)

    # Convert CNN output into clean clusters
    predictions = convertOutputToClusters(px, py, conf)

#######################################################################

example(5)