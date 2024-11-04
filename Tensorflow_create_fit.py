import os
import cv2
import tensorflow as tf
import numpy as np
from glob import glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow import dice_loss, dice_coef, Adam, iou, CSVLogger, ReduceLROnPlateau
from tensorflow import ModelCheckpoint, Recall, Precision, TensorBoard, EarlyStopping
from tensorflow import Model

from sklearn.utils import shuffle
from tensorflow import ResNet50





def function_for_map(x, y):
  x = x.decode()
  x = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
  x = cv2.resize(x, (512, 512))
  x = x/255.0
  x = x.astype(np.float32)
  x.set_shape([512, 512, 1])


  y = y.decode()
  y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
  y = cv2.resize(y, (512, 512))
  y = y/255.0
  y = y.astype(np.float32)
  y.set_shape([512, 512, 1])
  #ЕСЛИ картинка-текст
  # y = np.loadtxt(y, usecols=range(5))
  return x, y



class Model(keras.layers.Layer):
    def __init__(self):
	super().__init__()
        sef.inp = layers.Input(shape=(512, 512, 3),batch_size=3)
        self.conv_64 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
# (3 3) размер кисти
# 64 число новых каналов 
        self.pool_1 = layers.MaxPooling2D((2, 2))#уменьшение в 2 раза
        self.conv_128 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv_256 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv_512 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv_t256 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
        self.conv_t128 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        self.conv_t64 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        self.conv_1 = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')

    def call(self, inputs):
        c1 = self.inp(inputs)
        c1 = self.conv_64(c1)
        c1 = layers.BatchNormalization()(c1)
        с1 = layers.Activation("relu")(c1)
        c1 = self.conv_64(c1)
        p1 = self.pool_1(c1)

        c2 = self.conv_128(p1)
        c2 = layers.BatchNormalization()(c2)
        с2 = layers.Activation("relu")(c2)
        с2 = self.relu(с2)
        c2 = self.conv_128(c2)
        с2 = self.batch_norm(с2)
        p2 = self.pool_1(c2)

        c3 = self.conv_256(p2)
        c3 = layers.BatchNormalization()(c3)
        с3 = layers.Activation("relu")(c3)
        c3 = self.conv_256(c3)
        p3 = self.pool_1(c3)

        c4 = self.conv_512(p3)#ботл-нек
        c4 = self.conv_512(c4)

        c5 = self.conv_t256(c4)
        c5 = layers.concatenate([c5, c3])
        c5 = self.conv_256(c5)
        c5 = layers.BatchNormalization()(c5)#вычитает из входов среднее чтобы не учить заново красные машины
        с5 = layers.Activation("relu")(c5)
        c5 = self.conv_256(c5)

        c5 = self.conv_t128(c4)
        c5 = layers.concatenate([c5, c2])
        c5 = self.conv_128(c5)
        c5 = layers.BatchNormalization()(c5)
        с5 = layers.Activation("relu")(c5)
        c5 = self.conv_128(c5)

        c5 = self.conv_t64(c5)
        c5 = layers.concatenate([c5, c1])
        c5 = self.conv_64(c5)
        c5 = layers.BatchNormalization()(c5)
        с5 = layers.Activation("relu")(c5)
        c5 = self.conv_64(c5)

        c5 = self.conv_1(c5)
        return c5
# вверху вид сверточной сети U-net/Diffusion
# внизу сверточная сеть
#model = tf.keras.models.Sequential([Conv2D_32(relu), MaxPool_2, Conv2D_64(relu), MaxPool_2, Conv2D_64(relu), MaxPool_2, Flatten, Dense_128, Dense_256, Dense_10])





#self.batch_norm_2 = layers.BatchNormalization()
#self.in_layer = model.add(layers.Flatten())
#self.hid_layer1 = layers.Dense(64, activation='relu')
#self.hid_layer2 = layers.Dense(10, activation='softmax')

#Готовая нс-трансформер с тысячей классов
#encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
#image_features = get_layer(encoder,"conv4_block6_out",-1)

H=512
W=512


# DATASET картинка-каринка
dataset_path = "new_data"
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "test")

batch_size = 2
lr = 1e-4
num_epochs = 20
model_path = os.path.join("files", "model.h5")#куда сохранять НС
csv_path = os.path.join("files", "data.csv")

train_x = sorted(glob(os.path.join(train_path, "image", "*png")))#возвращает 2 массива путей
train_y = sorted(glob(os.path.join(train_path, "mask", "*png")))
train_x, train_y = shuffle(train_x, train_y, random_state=42) # перемешивание

valid_x = sorted(glob(os.path.join(valid_path, "image", "*png")))#возвращает 2 массива путей
valid_y = sorted(glob(os.path.join(valid_path, "mask", "*png")))

print(f"Train: {len(train_x)} - {len(train_y)}")
print(f"Valid: {len(valid_x)} - {len(valid_y)}")


train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))

train_dataset = train_dataset.map(function_for_map)
#мап принимает функцию и отрабатывает по всему датасету
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(10)



valid_dataset = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))

valid_dataset = valid_dataset.map(function_for_map)
#функция мап принимает функцию и отрабатывает ее всему датасету
valid_dataset = valid_dataset.batch(batch_size)
valid_dataset = valid_dataset.prefetch(10)

# DATASET картинка-текст
# image_dataset = tf.keras.utils.image_dataset_from_directory(
#    'data/images',
#    image_size=(224, 224),
#    batch_size=32
#)

# DATASET текст-класс
# дата/
#    класс1/
#        файл1.txt
#        файл2.txt
# raw_train_ds = utils.text_dataset_from_directory(
#    train_dir,
#    batch_size=batch_size,
#    validation_split=0.2,
#    subset='training',
#    seed=seed)
#
#raw_val_ds = utils.text_dataset_from_directory(
#    train_dir,
#    batch_size=batch_size,
#    validation_split=0.2,
#    subset='validation',
#    seed=seed)
#
# train_ds = raw_train_ds.map(vectorize_text)  (f(x_train, y_train): x_train = tf.expand_dims(x_train, -1))
# val_ds = raw_val_ds.map(binary_vectorize_text)
# train_ds = train_ds.batch(batch_size)
# val_ds = train_ds.batch(batch_size)
# train_ds = train_ds.prefetch(10)
# val_ds = val_ds.prefetch(10)


# Загрузить модель в ОЗУ
# model = tf.keras.models.load_model('my_model.h5')
# model.summary() # Напечатать свойства

# Fine-tune
#for layer in model.layers:
#    if layer.name == 'block5_conv1':
#    	layer.trainable = False

model = Model() #если создана классом
model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision()])

callbacks = [
ModelCheckpoint(model_path, verbose=1, save_best_only=True),
ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
CSVLogger(csv_path),
TensorBoard(),
EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
]

model.fit(
train_dataset,
epochs=num_epochs,
validation_data=valid_dataset,
callbacks=callbacks
)

model.save("model.h5")
