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

        # Параметры ML-модели
        self.ch_num2 = ch_num2
        self.dim = dim


        # 3 слоя энкодера
        self.conv1 = nn.Conv2d(ch_num1, ch_num2, (2,2), stride=2)
        self.conv2 = nn.Conv2d(ch_num2, ch_num2*2, (2,2), stride=2)
        self.conv3 = nn.Conv2d(ch_num2*2, ch_num2*4, (2,2), stride=2)


        # 3 слоя декодера

        # NOTE
        # Обратите внимание, размерность - ch_num2*8
        # На вход декодера идут и новые, и старые карты признаков.

        self.convt1 = nn.ConvTranspose2d(ch_num2*8, ch_num2*2, (2,2), stride=2)
        self.convt2 = nn.ConvTranspose2d(ch_num2*4, ch_num2, (2,2), stride=2)
        self.convt3 = nn.ConvTranspose2d(ch_num2*2, ch_num1, (2,2), stride=2)


        # Cлои для декодера
        self.conv_stable1 = nn.Conv2d(ch_num2 * 2, ch_num2 * 2, 1, stride=1)
        self.conv_stable2 = nn.Conv2d(ch_num2, ch_num2, 1, stride=1)
        self.conv_stable3 = nn.Conv2d(ch_num1, ch_num1, 1, stride=1)


        # 6 ReLU (3 энкодера + 3 декодера)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()


        # 6 BatchNorm (3 энкодера + 3 декодера)
        self.bn1 = nn.BatchNorm2d(ch_num2)
        self.bn2 = nn.BatchNorm2d(ch_num2 * 2)
        self.bn3 = nn.BatchNorm2d(ch_num2 * 4)
        self.bn4 = nn.BatchNorm2d(ch_num2 * 2)
        self.bn5 = nn.BatchNorm2d(ch_num2)
        self.bn6 = nn.BatchNorm2d(ch_num1)


        # ChannelAttention1
        self.avg_pool1 = nn.AvgPool2d(self.dim // 8)
        self.lin_channel1 = nn.Linear(self.ch_num2*4, self.ch_num2*4, bias=False)
        self.relu_a1 = nn.ReLU(inplace=True)
        self.lin_channel2 = nn.Linear(self.ch_num2*4, self.ch_num2*4, bias=False)


        # ChannelAttention2
        self.avg_pool2 = nn.AvgPool2d(self.dim // 8)     
        self.lin_channel3 = nn.Linear(self.ch_num2*4, self.ch_num2*4, bias=False)
        self.relu_a2 = nn.ReLU(inplace=True)
        self.lin_channel4 = nn.Linear(self.ch_num2*4, self.ch_num2*4, bias=False)


        # Attention output
        self.sigmoid_chan = nn.Sigmoid()


    def forward(self,x):

        #ENCODER1
        n1 = self.conv1(x)
        n1 = self.bn1(n1)
        n1 = self.relu1(n1)
        # print(f'Encoder 1 {n1.size()}')

        #ENCODER2
        n2 = self.conv2(n1)
        n2 = self.bn2(n2)
        n2 = self.relu2(n2)
        # print(f'Encoder 2 {n2.size()}')

        #ENCODER3
        n3 = self.conv3(n2)
        n3 = self.bn3(n3)
        n3 = self.relu3(n3)

        # ChannelAttention1
        a1 = self.avg_pool1(n3)[0].flatten()
        a1 = self.lin_channel1(a1)
        a1 = self.relu_a1(a1)
        a1 = self.lin_channel2(a1)

        # ChannelAttention2
        a2 = self.avg_pool2(n3)[0].flatten()
        a2 = self.lin_channel3(a2)
        a2 = self.relu_a2(a2)
        a2 = self.lin_channel4(a2)

        a2 = self.sigmoid_chan(a1+a2)
        a3 = n3 * a2.view(1, self.ch_num2 * 4, 1, 1)


        #DECODER1
        m1 = self.convt1(torch.cat([a3, n3], dim=1))
        m1 = self.conv_stable1(m1)
        m1 = self.bn4(m1)
        m1 = self.relu4(m1)

        #DECODER2
        m2 = self.convt2(torch.cat([m1, n2], dim=1))
        m2 = self.conv_stable2(m2)
        m2 = self.bn5(m2)
        m2 = self.relu5(m2)


        #DECODER3
        m3 = self.convt3(torch.cat([m2, n1], dim=1))
        m3 = self.conv_stable3(m3)
        m3 = self.bn6(m3)
        m3 = self.relu6(m3)

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
model = NN(dim,1,64)
summary(model,((1,dim,dim))) 


# Предобработка датасета
list_of_transformations = [
    transforms.Resize((dim,dim)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
]
transform = transforms.Compose(list_of_transformations)


# Массив путей к датасету
x_path = '/content/drive/MyDrive/x/*.png'
y_path = '/content/drive/MyDrive/y/*.png'
train_x = sorted(glob(x_path))
train_y = sorted(glob(x_path))

image_pairs = np.vstack((train_x, train_y)).T

train_dataset = CustomDataset(image_pairs, transform=transform)
# print(train_dataset[0])

load_Train = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=2
)


# Объекты для обучения
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




        # SpacialAttention
        # Такие слои встречаются в декодерах Image2Image
        # Они схлопывают все каналы в один. Затем изображение проходит пространственный анализ.
        # Реализуется тремя слоями 
        # (1 -> 32)(32 -> 16)(16 -> 1)

        # self.conv_sp4 = nn.Conv2d(1, 32, kernel_size=7, padding=3)
        # self.conv_sp5 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        # self.conv_sp6 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        # as1 = torch.mean(m1, dim=1, keepdim=True)
        # as1 = self.conv_sp1(as1)
        # as1 = self.conv_sp2(as1)
        # as1 = self.conv_sp3(as1)
        # as1 = self.sigmoid_chan(as1)
        # as1 = m1 * as1
