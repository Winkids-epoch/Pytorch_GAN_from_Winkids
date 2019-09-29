'''
Learning from Movan Zhou(Using GAN for Handwritten Number Recognition(CNN-mnist))
Add some notes of my understand
At the bottom add the "torch.max" notes
'''



import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

torch.manual_seed(1)                                            # repeatable
EPOCH = 1                                                       # once train
BATCH_SIZE = 50                                                 # batch_size
LR = 0.02                                                       # learning rate
DOWNLOAD_DATA = True

# download dataset and set it able to train  # train_sample:6000, shape(28*28)
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),                # transform to tensor
    download=True
)

# dataloader
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,  # batch size
    shuffle=True,  # random shuffle
    num_workers=0
)

# [10000,28,28] is not used to train model
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)  

 # upsqueeze (28,28) to (1,28,28)，test set (#2000)，normalization to [2000,1,28,28]，value in [0,1]
test_x = torch.unsqueeze(train_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255  
test_y = test_data.targets[:2000]  # labels


# print(test_data.data.size())
# ERROR: MNIST object has no attribute 'size' ，_________using "test_data.data.size(); test_data.targets.size()" instead


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.output = torch.nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),
                   -1)  
        # flatten tensor to 1 dim ，x has 4 dims (batch_size, out_channels, height, weight)，x.size(0) is the first dim: batch-size, -1
        # Represents self-adaption, and automatically allocates the number of remaining columns when the number of rows is specified
        
        out = self.output(x)
        return out, x


cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)[0]  # CNNreturn 2 values ，we need the first one
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# predition(x is used to plot)
test_out, _ = cnn(test_x[:10])  # CNN returns out and x (10 samples 0-9)
pred = torch.max(test_out, 1)[1].data.numpy()
print(pred)

'''
1. torch.max(input, dim) :return the max value and its index of tensor(index begin with 0, indicate the location of max value). input must be tensor type，dim=0 represent col ，dim=1 represent row
Example: 
x: torch.rand(4,4)
 tensor([[0.5285, 0.1247, 0.8332, 0.5485],
        [0.7917, 0.6138, 0.5881, 0.3381],
        [0.4226, 0.6605, 0.8571, 0.0399],
        [0.1716, 0.0609, 0.9712, 0.4838]])

torch.max(x,1):
 (tensor([0.8332, 0.7917, 0.8571, 0.9712]), tensor([2, 0, 2, 2]))

torch.max(x,0):
 (tensor([0.7917, 0.6605, 0.9712, 0.5485]), tensor([1, 2, 3, 0]))

torch.max(x,1)[0]: [0] represent return the max value only!
 tensor([0.8332, 0.7917, 0.8571, 0.9712])

torch.max(x,1)[1]: [1] represent return the index value only!
 tensor([2, 0, 2, 2])

torch.max(x,1)[1].data:
 tensor([2, 0, 2, 2])

torch.max(x,1)[1].data.numpy():
 [2 0 2 2]

torch.max(x,1)[1].data.numpy().squeeze():
 [2 0 2 2]

torch.max(x,1)[0].data:
 tensor([0.8332, 0.7917, 0.8571, 0.9712])

torch.max(x,1)[0].data.numpy():
 [0.83318216 0.7917127  0.85708565 0.9711726 ]

torch.max(x,1)[0].data.numpy().squeeze():
 [0.83318216 0.7917127  0.85708565 0.9711726 ]

 2. x.view() seem like 'reshape', x.view(x.size(0),-1) turn [batch_size, out_channels, height, weight]，to [batch_size, out_channels*height*weight] where the #row is defined

                               1                 2              3      ...       out_channels*height*weight
 sample1
 sample2
 ...
 sample(batch_size)

'''
