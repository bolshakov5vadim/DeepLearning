import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from glob import glob
import os

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary


class NN(nn.Module):
    def __init__(self, dim, ch_num1, ch_num2):
        super(NN,self).__init__()
        self.ch_num2 = ch_num2
        self.dim = dim
        self.conv1 = nn.Conv2d(ch_num1, ch_num2, (2,2), stride=2)
        self.conv2 = nn.Conv2d(ch_num2, ch_num2*2, (2,2), stride=2)
        self.conv3 = nn.Conv2d(ch_num2*2, ch_num2*4, (2,2), stride=2)
        self.convt1 = nn.ConvTranspose2d(ch_num2*8, ch_num2*2, (2,2), stride=2)
        self.convt2 = nn.ConvTranspose2d(ch_num2*4, ch_num2, (2,2), stride=2)
        self.convt3 = nn.ConvTranspose2d(ch_num2*2, ch_num1, (2,2), stride=2)
       
        self.conv_stable1 = nn.Conv2d(ch_num2 * 2, ch_num2 * 2, 1, stride=1)
        self.conv_stable2 = nn.Conv2d(ch_num2, ch_num2, 1, stride=1)
        self.conv_stable3 = nn.Conv2d(ch_num1, ch_num1, 1, stride=1)


        # BatchNorm
        self.bn1 = nn.BatchNorm2d(ch_num2)
        self.bn2 = nn.BatchNorm2d(ch_num2 * 2)
        self.bn3 = nn.BatchNorm2d(ch_num2 * 4)
        self.bn4 = nn.BatchNorm2d(ch_num2 * 2)
        self.bn5 = nn.BatchNorm2d(ch_num2)
        self.bn6 = nn.BatchNorm2d(ch_num1)


        self.relu = nn.ReLU(inplace=True)

        # Channel Attention
        self.avg_pool1 = nn.AvgPool2d(self.dim // 8)
        self.avg_pool2 = nn.AvgPool2d(self.dim // 8)     
        self.lin_channel1 = nn.Linear(self.ch_num2*4, self.ch_num2*4, bias=False)
        self.lin_channel2 = nn.Linear(self.ch_num2*4, self.ch_num2*4, bias=False)
        self.lin_channel3 = nn.Linear(self.ch_num2*4, self.ch_num2*4, bias=False)
        self.lin_channel4 = nn.Linear(self.ch_num2*4, self.ch_num2*4, bias=False)
        self.sigmoid_chan = nn.Sigmoid()

    def forward(self,x):

        #ENCODER1
        n1 = self.conv1(x)
        n1 = self.bn1(n1)
        n1 = nn.ReLU()(n1)
        # print(f'Encoder 1 {n1.size()}')

        #ENCODER2
        n2 = self.conv2(n1)
        n2 = self.bn2(n2)
        n2 = nn.ReLU()(n2)
        # print(f'Encoder 2 {n2.size()}')

        #ENCODER3
        n3 = self.conv3(n2)
        n3 = self.bn3(n3)
        n3 = nn.ReLU()(n3)

        # Attention channel1
        a1 = self.avg_pool1(n3)[0].flatten()
        a1 = self.lin_channel1(a1)
        a1 = self.relu(a1)
        a1 = self.lin_channel2(a1)

        # Attention channel2
        a2 = self.avg_pool2(n3)[0].flatten()
        a2 = self.lin_channel3(a2)
        a2 = self.relu(a2)
        a2 = self.lin_channel4(a2)

        a2 = self.sigmoid_chan(a1+a2)
        a3 = n3 * a2.view(1, self.ch_num2 * 4, 1, 1)

        #DECODER1
        # cat увеличивает число каналов в 2 раза
        m1 = self.convt1(torch.cat([a3, n3], dim=1))
        m1 = self.conv_stable1(m1)
        m1 = self.bn4(m1)
        m1 = nn.ReLU()(m1)

        #DECODER2
        m2 = self.convt2(torch.cat([m1, n2], dim=1))
        m2 = self.conv_stable2(m2)
        m2 = self.bn5(m2)
        m2 = nn.ReLU()(m2)


        #DECODER3
        m3 = self.convt3(torch.cat([m2, n1], dim=1))
        m3 = self.conv_stable3(m3)
        m3 = self.bn6(m3)
        m3 = nn.ReLU()(m3)

        return nn.Tanh()(m3)


#DATASET UNIVERS
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_pairs,transform):
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        x_path, y_path = self.image_pairs[idx][0],  self.image_pairs[idx][1]
        x = Image.open(x_path).convert('RGB')
        y = Image.open(y_path).convert('RGB')
        x = self.transform(x)
        y = self.transform(y)
        return x, y


dim = 256
list_of_transformations = [
    transforms.Resize((dim,dim)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
]
transform = transforms.Compose(list_of_transformations)

model = NN(dim,1,64)
summary(model,((1,dim,dim))) # работает только с Conv2d+Pool


x_path = '/content/drive/MyDrive/x/*.png'
y_path = '/content/drive/MyDrive/y/*.png'
train_x = sorted(glob(x_path))# массив путей
train_y = sorted(glob(x_path))

image_pairs = np.vstack((train_x, train_y)).T

train_dataset = CustomDataset(image_pairs, transform=transform)
# print(train_dataset[0])

load_Train = torch.utils.data.DataLoader(train_dataset, batch_size=1,
             shuffle=True, num_workers=2)

optimizer = optim.Adam(model.parameters(), lr=2e-2, betas=(0.5, 0.999))
loss_mse = nn.MSELoss()
epochs = 3

for ep in range(epochs):
    for i, (x_data, y_data) in enumerate(load_Train):
        output = model(x_data)
        loss = loss_mse(output, y_data)

        loss_num = loss.item()
        psnr = 10 * torch.log10(1.0 / loss).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
                print('Train Epoch: {} [{}/{}], Loss {:.4f}, Accuracy: {:.2f}%'.format
                (ep+1, i, len(load_Train), loss_num, psnr))
torch.save(model.state_dict(), 'conv_net_BIG_model.pt')
