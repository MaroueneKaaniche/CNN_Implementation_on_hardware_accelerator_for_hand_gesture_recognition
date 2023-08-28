from torch import nn
import torch
import ai8x


class CustoModel(nn.Module):
    """
    CNN model for gesture recognition based on EIT signal
    """
    def __init__(self, num_classes=10, num_channels=1, dimensions=(1, 40), bias=True, **kwargs):
        super().__init__()
        # self.conv1=ai8x.FusedMaxPoolConv1dBNReLU
        self.conv1=ai8x.FusedMaxPoolConv1dReLU(num_channels,25,3,stride=1,padding=0,bias=bias,**kwargs)
        self.drop1=nn.Dropout(0.2)
        self.linear1=ai8x.FusedLinearReLU(450,128,bias=bias,wide=False)
        self.linear2=ai8x.FusedLinearReLU(128,64,bias=bias,wide=False)
        self.linear3=ai8x.Linear(64,num_classes,bias=bias,wide=True)

    
    def forward(self,x):
        """
        Forward propagation
        """
        # print(x.size())
        x=self.conv1(x)
        x=self.drop1(x)
        # x=torch.flatten(x)
        # x=x.view(32,-1)
        x=x.view(x.size(0),-1)
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.linear3(x)
        return (x)


def custommodel2(pretrained=False,**kwargs):
    """
    construct EIT model
    """
    assert not pretrained
    return CustoModel(**kwargs)

models = [
    {
        'name': 'custommodel2',
        'min_input': 1,
        'dim': 1,
    },
]
