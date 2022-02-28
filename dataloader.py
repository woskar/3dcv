from torch.utils.data import DataLoader

# Do we load the whole datase into the RAM? Or load one image at a time?

class DataSet():
    """ Dataset instance"""
    def __init__(self):
        a = None

    def __getitem__(self, index):
        return None

    def __len__(self):
        return len(self.imagePaths)

def getDataloader():
    """ Create train and test dataloader"""
    dataset_Train = DataSet()
    dataloader_Train = DataLoader(dataset=dataset_Train)
    dataloader_Test = DataLoader(dataset=dataset_Train)
    dataloader_Eval = DataLoader(dataset=dataset_Train)
    return dataloader_Train, dataloader_Test, dataloader_Eval
