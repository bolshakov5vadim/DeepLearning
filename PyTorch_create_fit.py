# Трансформер-НС
#кодирование
#позиционное кодирование

#матрицы внимания1
#матрицы внимания2
#конкатенация
#forward_prop_A
#batch_norm

#forward_prop_1
#forward_prop_2
#batch_norm

#матрицы внимания1(эмбеддинг, Q*кодер)
#матрицы внимания2(эмбеддинг, Q*кодер)
#конкатенация
#forward_prop_A
#batch_norm

#матрицы внимания1(кодер Q*декодер1)
#матрицы внимания2(кодер Q*декодер1)
#конкатенация
#forward_prop_A
#batch_norm

#forward_prop_1
#forward_prop_2
#batch_norm

#forward_prop_3
#softmax


# Определение нейронной сети

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import random
# import math

# # Определение нейронной сети

# class Attention(nn.Module):
#    def __init__(self, vocab_size=10, dim=3):
#        super(Attention,self).__init__()
#        self.a_dim = dim
#        self.K = nn.Linear(vocab_size, dim)
#        self.V = nn.Linear(vocab_size, dim)
#        self.Q = nn.Linear(vocab_size, dim)
#        self.softmax = nn.Softmax(dim=0)
#    def forward(self,xo,x):
#        n1 = self.K(xo)
#        n2 = self.V(xo)
#        n3 = self.Q(x)
#        n4 = n3 * n1
#        n4 = n4 / math.sqrt(dim)
#        n4 = self.softmax(n4)
#        n4 = n4 * n2
#        return n4

# class NN(nn.Module):
#    def __init__(self, out=10, a_dim =3):
#        super(NN, self).__init__()
#        self.out = out
#        self.att1 = Attention(out, a_dim)
#        self.att2 = Attention(out, a_dim)
#        self.a_linear = nn.Linear(a_dim*2, out)
#        self.linear1 = nn.Linear(out, out//2)
#        self.linear2 = nn.Linear(out//2, out)
#        self.linear3 = nn.Linear(out, out)
#    def forward(self, x):
#     #ENCODER
#     a1 = self.att1(x,x)
#     a2 = self.att2(x,x)
#     a = torch.cat([a1, a2], dim=-1)
#     a = self.a_linear(a)
#     a = a + x
#     b1 = nn.LayerNorm(self.out)(a)

#     #BOTTLENECK
#     m = self.linear1(b1)
#     m = self.linear2(m)
#     m = m + b1
#     encoder = nn.LayerNorm(self.out)(m)

#     #DECODER1
#     a1 = self.att1(x, encoder)
#     a2 = self.att2(x, encoder)
#     a = torch.cat([a1, a2], dim=0)
#     decoder1 = self.a_linear(a)
#     decoder1 = decoder1 + encoder
#     decoder1 = nn.LayerNorm(self.out)(decoder1)

#     #DECODER2
#     a1 = self.att1(encoder,decoder1)
#     a2 = self.att2(encoder,decoder1)
#     a = torch.cat([a1, a2], dim=0)
#     decoder2 = self.a_linear(a)
#     decoder2 = decoder2 + decoder1
#     decoder2 = nn.LayerNorm(self.out)(decoder2)

#     #BOTTLENECK
#     m = self.linear1(decoder2)
#     m = self.linear2(m)
#     m = m + decoder2
#     output_n = nn.LayerNorm(self.out)(m)

#     output = self.linear3(output_n)
#     return nn.Sigmoid()(output)

# def totensor(s):
#   embedding = np.zeros(10, dtype=np.float32)
#   inputt = s.split(" ")
#   for word in inputt:
#         index = vocabulary.index(word)
#         embedding[index] = 1
#   tensor = torch.from_numpy(embedding)
#   return tensor

# def hamming_distance(input, output):
#   a = input
#   b = output.detach().cpu().numpy()
#   arr = np.array([[a], [b]])
#   hamming = np.diff(arr, axis=0).sum()
#   return hamming

# def mutate_weights(model):
#   random = np.random.uniform(0, 0.1, (20, 1))
#   with torch.no_grad():
# 	  for param in model.parameters():
#            noise = random[random.randint(1, 19)] * torch.randn_like(param)
#            if np.random.rand() < 0.1:param.add_(noise)

# dim = 10
# vocabulary = [
#     "hello",
#     "mundo",
#     "world",
#     "how",
#     "?",
#     "EOS",
#     "SOS",
#     "a",
#     "hola",
#     "c",
# ]


# model = NN(dim)
# reward = 0
# optimizer = torch.optim.Adam(model.parameters(), lr=0.7, betas=(0.5, 0.999))
# loss_c = nn.MSELoss()
# E_matrix = np.eye(dim)

# max_iterations = 15

# for iteration in range(max_iterations):
#  s = input()
#  tensor=totensor(s)
#  output = model(tensor)
#  print (f"ITERATION {iteration}: ")
#  for i in range (dim):
#   if output[i]>0.5:print(vocabulary[i])
#  print ("----")
#  #distance = hamming_distance(embedding, output)
#  #print(distance)
#  print("Оценка-")
#  grade = input()
#  if (grade == '?'):  reward -= 10
#  else: reward += 10
#  if (reward < 0):
#        #y_data = tensor
#        y_raw=vocabulary[random.randint(0, dim-1)]
#        y_data=totensor(y_raw)
#        loss = loss_c(output, y_data)
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#  print ("--------------")


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, ch_num1, ch_num2):
        super(NN,self).__init__()
        self.ch_num2 = ch_num2
        self.conv1 = nn.Conv2d(ch_num1, ch_num2, (3,3), stride=3) 
        self.conv2 = nn.Conv2d(ch_num2, ch_num2*2, (3,3), stride=3)
        self.conv3 = nn.Conv2d(ch_num2*2, ch_num2*4, (3,3), stride=3)
        self.conv4 = nn.Conv2d(ch_num2*4, ch_num2*8, (3,3), stride=3)
        self.convt1 = nn.ConvTranspose2d(ch_num2*8, ch_num2*4, (3,3), stride=3)
        self.convt2 = nn.ConvTranspose2d(ch_num2*4, ch_num2*2, (3,3), stride=3)
        self.convt3 = nn.ConvTranspose2d(ch_num2*2, ch_num2, (3,3), stride=3)
        self.convt4 = nn.ConvTranspose2d(ch_num2, ch_num1, (3,3), stride=3)
        self.conv0 = nn.Conv2d(ch_num2*2, 1, (3,3), stride=3) 
        self.pool_1 = nn.MaxPool2d((2, 2), stride=2)
        self.uppool_1 = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self,x):
        n1 = self.conv1(x)
        n1 = nn.BatchNorm2d(self.ch_num2, affine=False, track_running_stats=False)(n1)
        n1 = nn.ReLU(n1)
        n1 = self.conv1(n1)
        n1 = self.pool_1(n1)

        n2 = self.conv2(n1)
        n2 = nn.BatchNorm2d(self.ch_num2*2, affine=False, track_running_stats=False)(n2)
        n2 = nn.ReLU(n2)
        n2 = self.conv2(n2)
        n2 = self.pool_1(n2)

        n3 = self.conv3(n2)
        n3 = nn.BatchNorm2d(self.ch_num2*4, affine=False, track_running_stats=False)(n3)
        n3 = nn.ReLU(n3)
        n3 = self.conv3(n3)
        n3 = self.pool_1(n3)

        n4 = self.conv4(n3)
        n4 = nn.BatchNorm2d(self.ch_num2*8, affine=False, track_running_stats=False)(n4)
        n4 = nn.ReLU(n4)
        n4 = self.conv4(n4)
        n4 = self.pool_1(n4)

        xx = self.conv7(n4)
        xx = self.conv7(xx)

        m1 = self.convt1(xx)
        m1 = torch.cat([m1, n4], 1)
        m1 = self.conv4(m1)
        m1 = nn.BatchNorm2d(self.ch_num2*8, affine=False, track_running_stats=False)(m1)
        m1 = nn.ReLU(m1)
        m1 = self.conv4(m1)
        m1 = self.uppool_1(m1)

        m2 = self.convt2(m1)
        m2 = torch.cat([m2, n3], 1)
        m2 = self.conv3(m2)
        m2 = nn.ReLU(m2)
        m2 = self.conv3(m2)
        m2 = self.uppool_1(m2)

        m3 = self.convt3(m2)
        m3 = torch.cat([m3, n2], 1)
        m3 = self.conv2(m3)
        m3 = nn.ReLU(m3)
        m3 = self.conv2(m3)
        m3 = self.uppool_1(m3)

        m4 = self.convt4(m3)
        m4 = torch.cat([m4, n1], 1)
        m4 = self.conv1(m4)
        m4= nn.ReLU(m4)
        m4 = self.conv1(m4)
        m4 = self.uppool_1(m4)

        end = self.conv0(m4)
        return nn.Tanh(end)


list_of_transformations = [
    transforms.Resize((256,256)),
    transforms.ToTensor(),
]
transform = transforms.Compose(list_of_transformations)

dataset_path = "new_data"
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "test")

#DATASET1
#L__train
#¦   L__ Cat
#¦   L__ Dog

train_dataset = ImageFolder(root=train_path, transform=transform)
val_dataset = ImageFolder(root=valid_path, transform=transform)
batch_size = 50
num_workers = 2
load_Train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
             shuffle=True, num_workers=num_workers)
load_Test = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
            shuffle = False, num_workers=num_workers)



#DATASET UNIVERS
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_pairs,transform):
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        img1 = Image.open(img1_path).convert('RGB')  
        img2 = Image.open(img2_path).convert('RGB')
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2
train_x = sorted(glob(os.path.join(train_path, "image", "*png")))#возвращает 2 массива путей
train_y = sorted(glob(os.path.join(train_path, "mask", "*png")))
train_x, train_y = shuffle(train_x, train_y, random_state=42) # перемешивание
image_pairs = np.hstack((train_x, train_y))

train_dataset = CustomDataset(image_pairs, transform=transform)
val_dataset = CustomDataset(image_pairs, transform=transform)

load_Train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
             shuffle=True, num_workers=num_workers)
load_Test = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
            shuffle = False, num_workers=num_workers)


NN = NN(64, 128)
print(NN)
optimizer = optim.Adam(Gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
epochs = 5

for ep in range(epochs):
    for i, (x_data, y_data) in enumerate(load_Train):
        output = NN(x_data)
        loss = nn.NLLLoss(ouptut, y_data)

        total = y_data.size(0)
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == y_data).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
                print('Train Epoch: [{}/{}], Loss {:.4f}, Accuracy: {:.2f}%'.format
                (ep+1, epochs, loss.data[0], correct / total * 100)
torch.save(NN.state_dict(), 'conv_net_model.ckpt')
