
import torch
import torch.nn as nn
from torchsummary import summary
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
		xt1,xt2=y.split(y.size(0)//2)
		zeros=torch.zeros(y.size(0)//2, dtype=torch.int64)
		xt = torch.cat([xt1, zeros],dim=0)
		return xt,xt2


class NN(nn.Module):
    def __init__(self, dim, max_context):
        super(NN, self).__init__()
        self.dim = dim
        self.a_dim = dim // 2
        self.max_context=max_context
        self.embeddings = nn.Embedding(self.dim, self.a_dim)
        self.att = nn.MultiheadAttention(self.a_dim, 1)
        self.linear1 = nn.Linear(self.a_dim, self.a_dim)
        self.linear2 = nn.Linear(self.a_dim, self.a_dim)
        self.linear3 = nn.Linear(self.a_dim, self.dim)
        self.layer_norm1 = nn.LayerNorm(self.a_dim)
        self.layer_norm2 = nn.LayerNorm(self.a_dim)
        self.layer_norm3 = nn.LayerNorm(self.a_dim)
        self.layer_norm4 = nn.LayerNorm(self.a_dim)
        self.layer_norm5 = nn.LayerNorm(self.a_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, generation_mode=False):
      if not generation_mode:
          return self._forward_train(x)
      else:
          return self._forward_generate(x)

    def _forward_train(self, x):
        #ENCODER
        y=self.embeddings(x)
        a,b = self.att(y,y,y)
        a = a + y
        b1 = self.layer_norm1(a)

        #BOTTLENECK
        m = self.linear1(b1)
        m = self.linear2(m)
        m = m + b1
        encod = self.layer_norm2(m)

        #DECODER1
        a,b = self.att(encod,encod,encod) # Декодер=Энкодер+Энкодер с инпутом старого
        decoder1 = a + encod
        decoder1 = self.layer_norm3(decoder1) # Вычитание среднего+обучаемые параметры

        #DECODER2
        a,b = self.att(decoder1,decoder1,encod)
        decoder2 = a + decoder1
        decoder2 = self.layer_norm4(decoder2)

        #BOTTLENECK
        b = self.linear1(decoder2)
        b = self.linear2(b)
        b = b + decoder2
        b = self.layer_norm5(b)

        output = self.linear3(b)
        output =self.sigmoid(output)
        return output

    def _forward_generate(self, x):

        for i in range(self.max_context):#!!!ОСНОВНОЙ ПРИКОЛ ТРАНСФОРМЕРА - ЦИКЛ
          output = self._forward_train(x)

          next_token = torch.argmax(output[0], dim=-1)
          zero_position=(x == 0).nonzero(as_tuple=True)[0]#адрес первого нуля
          if len(zero_position) > 0:
            x[zero_position[0]] = next_token
          #выделить токен и прибавить к тексту

          if next_token==vocabulary.index("EOS"): break
          if x[-1]!=0: break

        return x;


def to_pretensor(s): # Получение int
  not_tensor = []
  inputt=s.split(" ")
  for word in inputt:
    if len(not_tensor)==max_context: return not_tensor # Обрезка ввода до max_context//2
    try:not_tensor.append(vocabulary.index(word)) # Обработка неизвестных слов
    except: print (word+" not in list")
  return not_tensor


def to_tensor(not_tensor): # Получение int64

  tensor = torch.tensor(not_tensor, dtype=torch.int64)

  example=torch.zeros(max_context) # Заполнение нулями
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


max_context = 10
epochs = 3
tests = 10


vocabulary, clean_data = traindata()
int_data = []
for elem in clean_data:
  int_data.append ( to_tensor(to_pretensor(elem)) )
dim = len(vocabulary)


model = NN(dim, max_context)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_c = nn.CrossEntropyLoss()

dataset = CustomDataset(int_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=5)

for j in range(epochs): 
 for i, (x, y) in enumerate(dataloader):#ОБУЧЕНИЕ
  output=model(x)
  y_probs=to_probs(y)
  #print(f'Output {output[0][0]}') # Вывод-матрица. Берем первую строку
  #print(f'Target {y_probs[0]}')
  #print(f'{vocabulary[torch.argmax(output[0][0], dim=-1)]} {output}')
  loss = loss_c(output[0][0], y_probs[0]) # Вывод-матрица. Берем первую строку
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if i % 100 ==0:
    print('Train Epoch {}: [{}/{}], Loss {:.4f}'.format
   (j, i+1, len(dataloader), loss))
    

# ВВОД
for test in range(tests):
    s = input()
    pretensor = to_pretensor(s)
    tensor=to_tensor(pretensor)
    print (f"TEST {test}/{tests}: ")
    print(f'Input {tensor}')

    output = model(tensor, generation_mode=True)
    print(f'Output {output}')
    for word in output: print(f'{vocabulary[word]} {word}')
    print ("----")


torch.save(model, "model.pt")
