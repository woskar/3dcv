import torch
from torch import nn

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

class Pose2DLayer(nn.Module):
    def __init__(self):
        super(Pose2DLayer, self).__init__()
        self.coord_norm_factor = 10
        self.num_keypoints = 9

    def forward(self, output):
        # output : BxAs*(1+2*num_vpoints+num_classes)*H*W
        nB = output.data.size(0)
        nA = 1
        nV = self.num_keypoints
        nH = output.data.size(2)
        nW = output.data.size(3)
        output = output.view(nB * nA, (3 * nV), nH * nW).transpose(0, 1). \
            contiguous().view((3 * nV), nB * nA * nH * nW)
        conf = torch.sigmoid(output[0:nV].transpose(0, 1).view(nB, nA, nH, nW, nV))
        x = output[nV:2*nV].transpose(0, 1).view(nB, nA, nH, nW, nV)
        y = output[2*nV:3*nV].transpose(0, 1).view(nB, nA, nH, nW, nV)
        grid_x = ((torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA * nV, 1, 1). \
            view(nB, nA, nV, nH, nW).type_as(output) + 0.5) / nW ) * self.coord_norm_factor
        grid_y = ((torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA * nV, 1, 1). \
            view(nB, nA, nV, nH, nW).type_as(output) + 0.5) / nH) * self.coord_norm_factor
        grid_x = grid_x.permute(0, 1, 3, 4, 2).contiguous()
        grid_y = grid_y.permute(0, 1, 3, 4, 2).contiguous()
        predx = x + grid_x
        predy = y + grid_y
        predx = predx.view(nB, nH, nW, nV) / self.coord_norm_factor
        predy = predy.view(nB, nH, nW, nV) / self.coord_norm_factor
        conf =conf.view(nB,nH,nW,nV)
        out_preds = [predx, predy, conf]
        return out_preds