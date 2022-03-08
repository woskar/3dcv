from torch import nn

class correspondenceNet(nn.Module):
    def __init__(self, encoder, decoder):
        super(correspondenceNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
