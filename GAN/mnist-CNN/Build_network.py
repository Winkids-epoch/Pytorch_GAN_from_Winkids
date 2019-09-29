'''
    sturcture of cnn
'''



import torch.nn as nn

class CNN(nn.Module):
    '''
    convoluntial neural network
    2 conv2d layers(conv2d——pooling——conv2d——pooling)
    1 fc layer
    '''
    def __init__(self):
        super(CNN ,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
            in_channels=1,                                # image(1,28,28)
            out_channels=16,                              # (16,28,28) n_filter(n_channels)
            kernel_size=5,                                # size of filter(5x5)
            stride=1,                                     # filter step
            padding=2                                     # padding = (kernel_size-1)/2 if want same image size of input
            ),
            nn.ReLU(),                                    # Activation Function
            nn.MaxPool2d(kernel_size=2)                   # 2x2 area to make image from(16,28,28) to (16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
              in_channels=16,                             # (16,14,14)
              out_channels=32,                            # (32,14,14)
              kernel_size=5,                              # (5x5)
              stride=1,                                   # step
              padding=2),                                 # unchanged size
            nn.MaxPool2d(kernel_size=2)                   # (32,7,7)
        )
        self.out = nn.Linear(32*7*7,10)                   # IMPORTANT:out_dim represent the classes you want(handwritten digitals has 10 numbers)

    def forward(self, x) :
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)                         # IMPORTANT FLATTEN : before enter full-connnected net, should flatten 4 dim to 2 dim
                                                          # (batch_size, channels, width, height)
                                                          # transforms (64, 32, 7, 7) to (64, 32*7*7)
        out = self.out(x)

        return out, x                                     # x for plot




'''

'''