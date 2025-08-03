# Трансформер-НС
#кодирование

#матрицы внимания1
#матрицы внимания2
#конкатенация
#forward_prop_A
#batch_norm

#forward_prop_1
#forward_prop_2
#batch_norm

#матрицы внимания1
#матрицы внимания2
#конкатенация
#forward_prop_A
#batch_norm

#матрицы внимания1(кодер)
#матрицы внимания2(кодер)
#конкатенация
#forward_prop_A
#batch_norm

#forward_prop_1
#forward_prop_2
#batch_norm

#forward_prop_3
#softmax

	int a = 5;
	int* a1 = &a;
	std::cout << *a1 << '\t' << a1;

# Word2Vec — это метод представления слов в виде векторов, который позволяет 
# захватывать семантические и синтаксические особенности слов.
# Состоит из матриц длиной vocab x embed_dim.
# Существует два способа обучения Word2Vec:
# CBOW (Continuous Bag of Words) — предсказывает одно слово по многим.
# Skip-gram — предсказывает многие слова по одному.


# Метрики безопасности-среднее время реагирования на инцидент, процент покрытия систем,
# скорость восстановления, срок и точку восстановления данных
# Конференция 2023 (уязвимости перехвата СМС, ультразвуковой взлом Умных док-станций)


# Определение нейронной сети
import torch
import torch.nn as nn
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)//2

	def __getitem__(self, idx):
		y = self.data[idx]
		y_int = to_pretensor(y)
		yt = to_tensor(y_int)
		xt1,xt2=yt.split(yt.size(0)//2)
		zeros=torch.zeros(yt.size(0)//2, dtype=torch.int64)
		xt = torch.cat([xt1, zeros],dim=0)
		return xt,xt2
    #return torch.tensor(self.data[idx*2], dtype=torch.long), torch.tensor(self.data[idx+1], dtype=torch.long)

class Attention(nn.Module):
   def __init__(self, vocab_size, dim):
       super(Attention,self).__init__()
       self.a_dim = dim
       self.K = nn.Linear(vocab_size, dim)
       self.V = nn.Linear(vocab_size, dim)
       self.Q = nn.Linear(vocab_size, dim)
       self.softmax = nn.Softmax(dim=0)
   def forward(self,x,xo):
       n1 = self.K(xo)
       n2 = self.V(xo)
       n3 = self.Q(x)
       n4 = n3 * n1
       n4 = n4 / math.sqrt(dim)
# МАСКА ИСПОЛЬЗУЕТСЯ ПРИ ОБУЧЕНИИ
       n4 = self.softmax(n4)
       n4 = n4 * n2
       #output = scatter(n4.unsqueeze(-1) * n2, dim=0, dim_size=n2.size(0), reduce='sum')#РАЗРЕЖЕННОЕ ВНИМАНИЕ (РАСПАРАЛЛЕЛ И ОТБРОС)
       return n4

class NN(nn.Module):
   def __init__(self, dim):
       super(NN, self).__init__()
       self.dim = dim
       self.a_dim = dim // 2
       self.max_context=max_context
       self.w2v = nn.Embedding(self.dim, self.a_dimembedding_dim)
       self.att1 = Attention(self.dim, self.a_dim)
       self.att2 = Attention(self.dim, self.a_dim)
       self.a_linear = nn.Linear(self.a_dim, self.dim)
       self.linear1 = nn.Linear(self.dim, self.dim//2)
       self.linear2 = nn.Linear(self.dim//2, self.dim)
       self.linear3 = nn.Linear(self.dim, self.dim)
       self.att = nn.MultiheadAttention(self.dim, 1)
   def forward(self, x):
    current_sequence = x.clone()#ИСПРАВЛЕНИЕ!
    outputs =  torch.empty(dim)
    for i in range(max_context//2-1):#!!!ОСНОВНОЙ ПРИКОЛ ТРАНСФОРМЕРА - ЦИКЛ
      #ENCODER
      y=self.w2v(x)
      # a1 = self.att1(y,y)
      # a2 = self.att2(y,y)
      # a = torch.cat([a1, a2], dim=-1)#dim -1 для w2v
      # a = self.a_linear(a)
      a,b = self.att(y,y,y)
      a = a + y
      b1 = nn.LayerNorm(self.dim)(a)

      #BOTTLENECK
      m = self.linear1(b1)
      m = self.linear2(m)
      m = m + b1
      encod = nn.LayerNorm(self.dim)(m)

      #DECODER1
      a,b = self.att(encod,encod,encod) # Декодер=Энкодер+Энкодер с инпутом старого
      decoder1 = a + encod
      decoder1 = nn.LayerNorm(self.dim)(decoder1)

      #DECODER2
      a,b = self.att(decoder1,decoder1,encod)
      decoder2 = a + decoder1
      decoder2 = nn.LayerNorm(self.dim)(decoder2)

      #BOTTLENECK
      b = self.linear1(decoder2)
      b = self.linear2(b)
      b = b + decoder2
      b = nn.LayerNorm(self.dim)(b)

      output = self.linear3(b)
      output = nn.Sigmoid()(output)

      outputs = torch.vstack((outputs, output[0][0]))

      next_token = torch.argmax(output[0][0], dim=-1)
      addr=(current_sequence == 0).nonzero(as_tuple=True)[-1][0]#адрес первого нуля
      current_sequence[0][addr] = next_token.unsqueeze(0).unsqueeze(0)
      #выделить токен и прибавить к тексту

      if next_token==vocabulary.index("EOS"): break
      if current_sequence[0][-1]!=0.: break

    return current_sequence,outputs;


def to_pretensor(s):
  not_tensor = []
  inputt=s.split(" ")
  for word in inputt:
    if len(not_tensor)==max_context: return not_tensor
    try:not_tensor.append(vocabulary.index(word))
    except: print (word+" not in list")
  return not_tensor


def to_tensor(not_tensor):

  embedding=np.array(not_tensor, dtype=np.int64)#int64/float32
  tensor = torch.from_numpy(embedding)
  example=torch.zeros(max_context)
  tensor=nn.utils.rnn.pad_sequence([tensor,example]).T[0]
  return tensor


def to_probs(int_tensor):
  probs = torch.zeros((max_context//2, dim))
  i=0
  for j in int_tensor:
    probs[i][j] = 1
    i=+1
  return probs


def traindata():
  myfile = open('train.txt', 'r')
  data = myfile.read()
  raw_data=data.lower()
  raw_data="".join(ch for ch in raw_data if (ch.isalpha() or ch==" " or ch== ":"))
  clean_data = raw_data.split("nperson" or ":")
  vocabulary = raw_data.split(" " or "nperson" or ":")
  vocabulary = list(set(vocabulary))
  vocabulary.append("EOS")
  return vocabulary, clean_data



vocabulary = [
    "N",
    "hello",
    "mundo",
    "world",
    "how",
    "EOS",
    "SOS",
    "a",
    "hola",
    "c",
]


reward = 0
loss_c = nn.CrossEntropyLoss()
epochs = 10





#ОСНОВНОЕ ОБУЧЕНИЕ
vocabulary, clean_data = traindata()

dim = len(vocabulary)
max_context = 10
model = NN(dim)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

dataset = CustomDataset(clean_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=5)


for i, (x, y) in enumerate(dataloader):#ОБУЧЕНИЕ
  output,outputs=model(x)
  y_probs=to_probs(y)
  print('ВЫВОД GPT')
  for word in output[0]: print(f'{vocabulary[word]} {word}')
  loss = loss_c(outputs, y_probs)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if i % 100 ==0:
    print('Train Epoch: [{}/{}], Loss {:.4f}'.format
   (i+1, len(dataloader), loss))
      input()


# ВВОД
for iteration in range(epochs):
 s = input()
 pretensor = to_pretensor(s)
 tensor=to_tensor(pretensor)
 print(f'original {tensor}')
 input()
 output = model(tensor)
 print (f"ITERATION {iteration}: ")
 print(f'original {tensor}')
 print(f'modif {output}')
 print ("----")
















# Модель image-to-image

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from glob import glob
import os

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary

class NN(nn.Module):
    def __init__(self, ch_num1, ch_num2):
        super(NN,self).__init__()
        self.ch_num2 = ch_num2
        self.conv1 = nn.Conv2d(ch_num1, ch_num2, (2,2), stride=2)
        self.conv2 = nn.Conv2d(ch_num2, ch_num2*2, (2,2), stride=2)
        self.conv3 = nn.Conv2d(ch_num2*2, ch_num2*4, (2,2), stride=2)
        self.convt1 = nn.ConvTranspose2d(ch_num2*4, ch_num2*2, (2,2), stride=2)
        self.convt2 = nn.ConvTranspose2d(ch_num2*2, ch_num2, (2,2), stride=2)
        self.convt3 = nn.ConvTranspose2d(ch_num2, ch_num1, (2,2), stride=2)
        self.pool_1 = nn.MaxPool2d((2, 2), stride=2)
        self.uppool_1 = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self,x):
        #ENCODER
        print(f'Start {x.size()}')
        n1 = self.conv1(x)
        n1 = nn.BatchNorm2d(self.ch_num2, affine=False, track_running_stats=False)(n1)
        n1 = nn.ReLU()(n1)
        n1 = self.pool_1(n1)
        print(f'Encoder 1 {n1.size()}')
        #здесь иногда Attention

        n2 = self.conv2(n1)
        n2 = nn.BatchNorm2d(self.ch_num2*2, affine=False, track_running_stats=False)(n2)
        n2 = nn.ReLU()(n2)
        n2 = self.pool_1(n2)
        print(f'Encoder 2 {n2.size()}')

        #DECODER
        m1 = self.convt2(n2)
        #m2 = torch.cat([m2, n3], 1)
        #m2 = self.conv2(m2)
        m1 = nn.BatchNorm2d(self.ch_num2*2, affine=False, track_running_stats=False)(m1)
        m1 = nn.ReLU()(m1)
        m1 = self.uppool_1(m1)
        print(f'Decoder 1 {m1.size()}')

        m1 = self.convt3(m1)
        m1 = nn.BatchNorm2d(self.ch_num2*2, affine=False, track_running_stats=False)(m1)
        m1 = nn.ReLU()(m1)
        m1 = self.uppool_1(m1)
        print(f'Decoder 2 {m1.size()}')
        return nn.Tanh()(m1)


list_of_transformations = [
    transforms.Resize((256,256)),
    transforms.ToTensor(),
]
transform = transforms.Compose(list_of_transformations)

model=NN(1,2)
summary(model,((1,256,256)))

#DATASET UNIVERS
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_pairs,transform):
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        x_path, y_path = self.image_pairs[0][idx],  self.image_pairs[1][idx]
        print(x_path, y_path)
        x = Image.open(x_path).convert('RGB')  
        y = Image.open(y_path).convert('RGB')
        x = self.transform(x)
        y = self.transform(y)
        return x, y

train_path = ''
train_x = sorted(glob(os.path.join(train_path, "image*")))#возвращает 2 массива путей
train_y = sorted(glob(os.path.join(train_path, "mask*")))

image_pairs = np.vstack((train_x, train_y))

train_dataset = CustomDataset(image_pairs, transform=transform)
print(train_dataset[0])
val_dataset = CustomDataset(image_pairs, transform=transform)

load_Train = torch.utils.data.DataLoader(train_dataset, batch_size=1, 
             shuffle=True, num_workers=2)
load_Test = torch.utils.data.DataLoader(val_dataset, batch_size=1, 
            shuffle = False, num_workers=2)


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
                (ep+1, epochs, loss.data[0], correct / total * 100))
torch.save(NN.state_dict(), 'conv_net_model.ckpt')








# Reinforcement агента с многопараметровым действием (движение руки)

# def select_action(state, epsilon=0.6, model):
#    if random.random() < epsilon:
#        return random.randint(0, num_actions - 1)  # Случайное действие
#    else:
#        with torch.no_grad():
#            return model(state).argmax().item()  # Жадное действие

# def select_multiple_action(x, model):
#    with torch.no_grad():
#        y = model(x)
#        action = torch.multinomial(y, 1).item()
#    return action



# y=select_action(state=x, model)# выбор случайного действия
# y_probs=to_probs(y)

# y, x_after=model(x) #робот возвращает вид с камеры
# grade=reward_model(x_after) # оценка вида с камеры

# epsilon=0.2
# ratio = torch.exp(y - old_y) #PPO, чтобы отношение не выходило за пределы [1-e, 1+e]
# clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

# loss = -torch.min(ratio * grade, clipped_ratio * grade).mean()
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
