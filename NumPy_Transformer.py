import numpy as np


def attention(x, WQ, WK, WV):
    d_key = 3 # размерность внимания
    K = x @ WK
    V = x @ WV
    Q = x @ WQ

    scores = Q @ K.T 
    scores = scores / np.sqrt(d_key)  # we just changed this
    scores = softmax(scores)
    scores = scores @ V
    return scores

#def pos_embedd(input, dim, pos):
#    input[pos] += np.sin(pos / (10000 ** (2 * i / dim)))

def decoder_attention(output, x, WQ, WK, WV):
    d_key = 3 # размерность внимания
    K = output @ WK    # Отличие, передаём output
    V = output @ WV 
    Q = x @ WQ   # То же, что и для самовнимания

    scores = Q @ K.T # остаточное соединение умножением
    scores = scores / np.sqrt(d_key)
    scores = softmax(scores)
    scores = scores @ V # остаточное соединение умножением
    return scores

def layer_norm(x, epsilon=1e-6): # layer_norm нужен для защиты от переполнения
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def forward_prop(Z, W, b):
    layer_raw = Z.dot(W) + b
    layer = 1 / (1 + np.exp(-hidden_raw))
    return layer


vocabulary = [
    "hello",
    "mundo",
    "world",
    "how",
    "?",
    "EOS",
    "SOS",
    "a",
    "hola",
    "c",
]
W1 = np.random.randn(4, 10)
W2 = np.random.randn(10, 4)
b1 = np.random.randn(10)
b2 = np.random.randn(4)
W_attent = np.random.randn(4, 6) 
b_attent = np.random.randn(6)
# матрицы + матрицы внимания

embedding = np.zeros(10)# вход
inn = input()
inn = inn.lower
inn_array = inn.split(inn)
i = 0
for word in inn_array:
	index = inn_array.index(word)
	embedding[index] = 1

WK1 = np.random.randn(3, 10) # размерность головы х размер запроса
WV1 = np.random.randn(3, 10)
WQ1 = np.random.randn(3, 10)

WK2 = np.random.randn(3, 10)
WV2 = np.random.randn(3, 10)
WQ2 = np.random.randn(3, 10)

W_linear = np.random.randn(10, 10) # длина ввода * длина словаря
b_linear = np.random.randn(10) # длина словаря

# Начинается кодер.

attention1 = attention(embedding, WQ1, WK1, WV1)

attention2 = attention(embedding, WQ2, WK2, WV2)

attentions = np.concatenate([attention1, attention2], axis=1)
Z = attentions @ W_attent
Z = layer_norm(Z + embedding)

output = forward_prop(Z, W1, b1)
output = forward_prop(Z, W2, b2)
output = layer_norm(output + Z)

# Здесь добавить обучение

# Начинается декодер. Здесь max_iters = 5 for i in range(max_iters):  if next_token == "EOS":return output

attention1 = attention(output, embedding, WQ1, WK1, WV1)

attention2 = attention(output, embedding, WQ2, WK2, WV2)

attentions = np.concatenate([attention1, attention2], axis=1)
Z = forward_prop(attentions, W_attent, b_attent)
Z = layer_norm(Z_decoder + [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) #здесь символ SOS

attention1 = decoder_attention(output, Z, WQ1, WK1, WV1)

attention2 = decoder_attention(output, Z, WQ2, WK2, WV2)

attentions = np.concatenate([attention1, attention2], axis=1)
Z_decoder = attentions @ W_attent
Z_decoder = layer_norm(Z_decoder + Z)


output = forward_prop(Z_decoder, W1, b1)
output = forward_prop(Z_decoder, W2, b2)
output = layer_norm(output + Z_decoder)

# Здесь добавить обучение

logits = forward_prop(decoder_output[-1], W_linear, b_linear)# Используем для предсказания только последние выходные данные
probs = softmax([logits])

print(inn)
print(" EOS " + vocabulary[np.argmax(probs)])
input()
