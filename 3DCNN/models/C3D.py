import torch.nn as nn
import torch

__all__ = [
    'C3D', 'c3d'
]

class C3D(nn.Module):
    """
    The C3D network as described in Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." Proceedings of the IEEE international conference on computer vision. 2015.
    Implementation based on the implementation by  David Abati: https://github.com/DavideA/c3d-pytorch
    """

    def __init__(self, num_classes = 1):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fConv6 = nn.Conv2d(512, 4096, kernel_size=(4,4), padding=(0,0))
        self.fConv7 = nn.Conv2d(4096, 4096, kernel_size=(1,1), padding=(0,0))
        self.fConv8 = nn.Conv2d(4096, num_classes, kernel_size=(1,1), padding=(0,0))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        if num_classes == 1:
            self.final = nn.Sigmoid()
        elif num_classes > 1:
            self.final = nn.LogSoftmax(dim=1)
        else:
            raise ValueError('{} is not a valid size for the last layer'.format(num_classes))
                
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = torch.squeeze(h, dim=2)
        h = self.relu(self.fConv6(h))
        h = self.dropout(h)
        h = self.relu(self.fConv7(h))
        h = self.dropout(h)
        h = self.fConv8(h)
        h = self.final(h)
        h = self.avgpool(h)
        h = torch.squeeze(h, dim=3)
        h = torch.squeeze(h, dim=2)

        return h



def c3d(**kwargs):
    model = C3D(**kwargs)
    return model
