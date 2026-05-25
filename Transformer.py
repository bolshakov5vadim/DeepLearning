
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class CustomDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return self.data.size(0) - (self.data.size(0) % 7)
    # Умная формула, чтобы датасет делился на 7.
	def __getitem__(self, idx):
		if idx+6>=self.data.size(0): raise IndexError(f"{idx} больше размера датасета {self.data.size(0)}")
		x = self.data[idx:idx+5]
		x = F.pad(x, (0, 5), 'constant')
    # Метод pad() позволяет заполнить тензор нулями до нужного размера.
    # Второй агрумент pad() содержит 4 числа -
    # числа слева, числа справа, числа сверху, числа снизу.
		return x,self.data[idx+6]


def to_probs(int_tensor):
  probs = torch.zeros(src_vocab_size)
  probs[int_tensor] = 1
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, key_padding_mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size) # [64,10,256?] -> [64,10,vocab_size]
        self.dropout = nn.Dropout(dropout)


    # Маски превращают токены паддинга [0] в [-inf] чтобы они не влияли на результат
    # Маски нужно транспонировать, иначе они не ложатся на оригиналы
    # Треугольная матрица нужна из-за батч-обучения. Чтобы скрыть будущие токены.
    # Если в маске будет условие (src != 0), то все значения true, модель выдаст NaN

    def generate_mask(self, src, tgt):
        src_mask = (src == 0).T
        tgt_mask = (tgt == 0).T
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(seq_length, 64), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # EMBEDDING + СОЗДАНИЕ МАСОК
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, None)
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, None, None)
        output = self.fc(dec_output)
        return output


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


# СОЗДАНИЕ ДАТАСЕТА
# Должно быть перед созданием модели, т. к. задаётся размер словаря

# Чтение из файла, нормализация
vocabulary, clean_data = traindata()

# Создание torch-вектора
int_data = torch.empty(len(clean_data), dtype=torch.int64)

# Запись в torch-вектор
for i, elem in enumerate(clean_data):
    try:int_data[i] = vocabulary.index(elem) # Обработка неизвестных слов
    except: print (elem+" not in list")
print(f'Words in dataset{int_data.size()}')


dataset = CustomDataset(int_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=5)


dim = len(vocabulary)


src_vocab_size = dim
tgt_vocab_size = dim
d_model = 128 # 128-256 mid-512
num_heads = 1 # 4
num_layers = 1 # 2
d_ff = 512 # 512-1024 mid-2048
max_seq_length = 10 # 100
dropout = 0.1


model = Transformer(
    src_vocab_size = src_vocab_size,
    tgt_vocab_size = tgt_vocab_size,
    d_model = d_model,
    num_heads = num_heads,
    num_layers = num_layers,
    d_ff = d_ff,
    max_seq_length = max_seq_length,
    dropout = dropout)


src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
#tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.zeros([64, max_seq_length], dtype = torch.int64)

output = model(src_data, tgt_data)
print(output[0]) # Первый батч из 64. Длина - src_vocab_len


# ОБУЧЕНИЕ
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
epochs = 1

model.train()


for e in range(epochs):
 for i, (x, y) in enumerate(dataloader):

  output=model(x, tgt_data)
  y_probs=to_probs(y)
  loss = criterion(output[0][0], y_probs)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  print('Train Epoch {}: [{}/{}], Loss {:.4f}'.format
   (e+1, i, len(dataloader), loss))

torch.save(model, "model.pt")
print(vocabulary)
