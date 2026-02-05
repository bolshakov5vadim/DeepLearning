
import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
import random
import math
from collections import Counter
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)-6

	def __getitem__(self, idx):
		x_t = self.data[idx:(idx+5)]
		pad=torch.zeros(max_context-x_t.size(0), dtype=torch.int64)
		x = torch.cat([x_t, pad],dim=0)
		y = self.data[idx+6]
		return x, y


class NN(nn.Module):
    def __init__(self, dim, max_context):
        super(NN, self).__init__()
        self.dim = dim
        self.e_dim = 512
        self.max_context=max_context
        self.embeddings = nn.Embedding(self.dim, self.e_dim)
        self.att = nn.MultiheadAttention(self.e_dim, 1)
        self.linear1 = nn.Linear(self.e_dim, self.e_dim)
        self.linear2 = nn.Linear(self.e_dim, self.e_dim)
        self.linear3 = nn.Linear(self.e_dim, self.dim)
        self.layer_norm1 = nn.LayerNorm(self.e_dim)
        self.layer_norm2 = nn.LayerNorm(self.e_dim)
        self.layer_norm3 = nn.LayerNorm(self.e_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, generation_mode=False):
      if not generation_mode:
          return self._forward_train(x)
      else:
          return self._forward_generate(x)

    def _forward_train(self, x):
        y=self.embeddings(x)

        #ENCODER1
        a,b = self.att(y,y,y)
        a = a + y
        encod1 = self.layer_norm1(a)

        #ENCODER2
        a,b = self.att(encod1,encod1,encod1) # Декодер=Энкодер+Энкодер с инпутом старого
        decoder1 = a + encod1
        decoder1 = self.layer_norm2(decoder1) # Вычитание среднего+обучаемые параметры

        #BOTTLENECK
        b = self.linear1(decoder1)
        b = self.linear2(b)
        b = b + decoder1
        b = self.layer_norm3(b)

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
  for word in s:
    #if len(not_tensor)==max_context: return not_tensor # Обрезка ввода до max_context//2
    try:not_tensor.append(vocabulary.index(word)) # Обработка неизвестных слов
    except: print (word+" not in list")
  return not_tensor


def to_probs(int_tensor):
  probs = torch.zeros(dim)
  for j in int_tensor:
    probs[j] = 1
  return probs


def traindata():
  myfile = open('train.txt', 'r')
  data = myfile.read()
  raw_data=data.lower()
  raw_data="".join(ch for ch in raw_data if (ch.isalpha() or ch==" " or ch== ":"))
  clean_data = raw_data.split(" " or "nperson" or ":")
  vocabulary = list(set(clean_data))
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
tests = 3

vocabulary, clean_data = traindata() # Текст -> List<string> 
print(clean_data)

# List<string> -> List<int>
int_data = to_pretensor(clean_data)
print(int_data)

 # List<int> -> Датасет torch.int64
data = torch.tensor(int_data, dtype=torch.int64)
print(data)
dim = len(vocabulary)


model = NN(dim, max_context)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_c = nn.CrossEntropyLoss()

dataset = CustomDataset(data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=5)

for j in range(epochs): 
 for i, (x, y) in enumerate(dataloader):#ОБУЧЕНИЕ
  output=model(x)
  y_probs=to_probs(y)
  #print(f'Output {output[0][0]}') # Убираем кавычки. Берем первую строку матрицы.
  #print(f'Target {y_probs[0]}')
  #print(f'{vocabulary[torch.argmax(output[0][0], dim=-1)]} {output}')
  
  # output[0][0]
  # Означает: 
  # -убираем кавычки матрицы
  # -берем первую строку матрицы.
  
  loss = loss_c(output[0][0], y_probs) # Вывод-матрица. Берем первую строку
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if i % 20 ==0:
    print('Train Epoch {}: [{}/{}], Loss {:.4f}'.format
   (j, i+1, len(dataloader), loss))
    

# ВВОД
for test in range(tests):
    s = input()
    pretensor = to_pretensor(s)
    tensor=torch.tensor(pretensor, dtype=torch.int64)
    print (f"TEST {test}/{tests}: ")
    print(f'Input {tensor}')

    output = model(tensor, generation_mode=True)
    print(f'Output {output}')
    for word in output: print(f'{vocabulary[word]} {word}')
    print ("----")


torch.save(model, "model.pt")
