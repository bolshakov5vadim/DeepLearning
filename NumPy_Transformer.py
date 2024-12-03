import numpy as np

def attention(x, xo, WQ, WK, WV):
    d = len(WQ) # размерность внимания
    K = xo @ WK    # Отличие, передаём output
    V = xo @ WV
    Q = x @ WQ   # То же, что и для самовнимания
# QVK обучаются вместе. Задачи распределяются между ними из-за особого порядка умножения
    scores = Q * K
    scores = scores / np.sqrt(d)
    scores = softmax(scores)
    scores = scores * V
    return scores

def pos_embedd(inputt, vocabulary, embedding):
#    inputt[pos] += np.sin(pos / (10000 ** (2 * i / dim)))
    for word in inputt:
      index = vocabulary.index(word)
      embedding[index] = 1
    return embedding

def layer_norm(x, epsilon=1e-6): # layer_norm нужен для защиты от переполнения
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

def softmax(x):

    return np.exp(x) / np.sum(np.exp(x))

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def forward_prop(Z, W, b):
    layer_raw = Z.dot(W) + b
    layer = 1 / (1 + np.exp(-layer_raw))
    return layer


vocabulary = [
    "hello",
    "mundo",
    "world",
    "how",
    "?",
    "eos",
    "sos",
    "a",
    "hola",
    "c",
]

W1 = np.random.randn(10, 4) # вход декод внимание * сжатие 
W2 = np.random.randn(4, 10)
W3 = np.random.randn(10, 10) # длина ввода * длина словаря
b1 = np.random.randn(4)
b2 = np.random.randn(10)
b3 = np.random.randn(10) # длина словаря
# матрицы 

WK1 = np.random.randn(10, 3) # размерность головы х размер запроса
WV1 = np.random.randn(10, 3)
WQ1 = np.random.randn(10, 3)

WK2 = np.random.randn(10, 3)
WV2 = np.random.randn(10, 3)
WQ2 = np.random.randn(10, 3)

W_attent = np.random.randn(6, 10) 
b_attent = np.random.randn(10)
# матрицы внимания

s = input()
s = s.lower()
inputt = s.split(" ")

embedding = pos_embedd(inputt, vocabulary, np.zeros(10))# вход
print(f'Embedding-{embedding}')

# Начинается кодер.

attention1 = attention(embedding, embedding, WQ1, WK1, WV1)
attention2 = attention(embedding, embedding, WQ2, WK2, WV2)
attentions = np.concatenate([attention1, attention2], axis=0)

Z = forward_prop(attentions, W_attent, b_attent)
Z = layer_norm(Z + embedding)

encoder = forward_prop(Z, W1, b1)
encoder = forward_prop(encoder, W2, b2)
encoder = layer_norm(encoder + Z)

print(f'Attentions(encode)-{encoder}')
input()

# Здесь добавить обучение

# Начинается декодер. Здесь max_iters = 5 for i in range(max_iters):

attention1 = attention(encoder, embedding, WQ1, WK1, WV1)
attention2 = attention(encoder, embedding, WQ2, WK2, WV2)
attentions = np.concatenate([attention1, attention2], axis=0)

decoder1 = forward_prop(attentions, W_attent, b_attent)
decoder1 = layer_norm(decoder1)

print(f'Attentions(decoder1)-{decoder1}')
input()

attention1 = attention(encoder, decoder1, WQ1, WK1, WV1)
attention2 = attention(encoder, decoder1, WQ2, WK2, WV2)
attentions = np.concatenate([attention1, attention2], axis=0)

decoder1 = forward_prop(attentions, W_attent, b_attent)
decoder2 = layer_norm(decoder2 + decoder1)

print(f'Attentions(decoder2)-{Z}')
input()

output = forward_prop(decoder2, W1, b1)
output = forward_prop(output, W2, b2)
output = layer_norm(output + decoder2)

# Здесь добавить обучение

logits = forward_prop(output, W3, b3)# Используем для предсказания только последние выходные данные
#logits = softmax([logits])
logits = sigmoid([logits])

out = []
out += vocabulary[np.argmax(logits)]
print(out)
input()
