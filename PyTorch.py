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
       self.embeddings = nn.Embedding(self.dim, self.a_dim)
       self.att = nn.MultiheadAttention(self.dim, 1)
       # self.a_linear = nn.Linear(self.a_dim, self.dim)
       self.linear1 = nn.Linear(self.dim, self.dim//2)
       self.linear2 = nn.Linear(self.dim//2, self.dim)
       self.linear3 = nn.Linear(self.dim, self.dim)

   def forward(self, x):
    current_sequence = x.clone()#ИСПРАВЛЕНИЕ!
    outputs =  torch.empty(dim)
    for i in range(max_context//2-1):#!!!ОСНОВНОЙ ПРИКОЛ ТРАНСФОРМЕРА - ЦИКЛ
      #ENCODER
      y=self.embeddings(x)
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
 print(f'input {tensor}')
 print(f'output {output}')
 print ("----")
