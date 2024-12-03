import os
import cv2
from tensorflow.python.keras import layers
import numpy as np
from glob import glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow.python.keras.callbacks as callbacks
import tensorflow.python.keras.metrics as metrics
from tensorflow.python.keras.models import Model
import tensorflow.python.keras.losses as losses

from sklearn.utils import shuffle
import tensorflow as tf





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



class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.inp = layers.Input(shape=(512, 512, 3),batch_size=3)
        self.conv_64 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
# (3 3) размер кисти
# 64 число новых каналов 
        self.conv_64 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv_128 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv_256 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv_512 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv_t256 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
        self.conv_t128 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        self.conv_t64 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        self.conv_1 = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')
        self.pool_1 = layers.MaxPooling2D((2, 2))#2Х уменьш. + пофиг на повороты, размеры, координаты
        self.uppool_1 = layers.UpSampling2D((2, 2)) # обратная операция

    def call(self, inputs):
        c1 = self.inp(inputs)
        c1 = self.conv_64(c1)
        c1 = layers.BatchNormalization()(c1) # чтобы не переобучался на красные машины-вычитание среднего из данных, нормализация, максимизация ошибки. Затруднит мухлеж НС
        c1 = self.conv_64(c1)
        p1 = self.pool_1(c1)

        c2 = self.conv_128(p1)
        c2 = layers.BatchNormalization()(c2)
        c2 = self.conv_128(c2)
        p2 = self.pool_1(c2)

        c3 = self.conv_256(p2)
        c3 = layers.BatchNormalization()(c3)
        c3 = self.conv_256(c3)
        p3 = self.pool_1(c3)

        c4 = self.conv_512(p3)#ботл-нек
        c4 = self.conv_512(c4)

        c5 = self.conv_t256(c4)
        c5 = layers.concatenate([c5, c3])
        c5 = self.conv_256(c5)
        c5 = layers.BatchNormalization()(c5)
        c5 = self.conv_256(c5)
        c5 = self.uppool_1(c5)

        c5 = self.conv_t128(c4)
        c5 = layers.concatenate([c5, c2])
        c5 = self.conv_128(c5)
        c5 = layers.BatchNormalization()(c5)
        c5 = self.conv_128(c5)
        c5 = self.uppool_1(c5)

        c5 = self.conv_t64(c5)
        c5 = layers.concatenate([c5, c1])
        c5 = self.conv_64(c5)
        c5 = layers.BatchNormalization()(c5)
        c5 = self.conv_64(c5)
        c5 = self.uppool_1(c5)

        c5 = self.conv_1(c5)
        return c5
# вверху вид сверточной сети U-net/Diffusion
# слой внимания можно добавить в кодере перед пулингом
# в декодере после конкатенации
# внизу сверточная сеть
#model = tf.keras.models.Sequential([Conv2D_32(relu), Batch, Conv2D_32(relu), MaxPool_2, Conv2D_64(relu), Batch, Conv2D_64(relu), MaxPool_2, Flatten, Dense_128, Dense_256, Dense_10])


#Готовая нс-трансформер с тысячей классов
#encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
#image_features = get_layer(encoder,"conv4_block6_out",-1)

H=512
W=512


ds_path = "new_data"
csv_path = ds_path
train_path = os.path.join(ds_path, "train")
valid_path = os.path.join(ds_path, "test")

batch_size = 2
lr = 1e-4
num_epochs = 20
model_path = os.path.join("files", "model.h5")#куда сохранять НС
csv_path = os.path.join("files", "data.csv")

# DATASET универсальный
train_x = sorted(glob(os.path.join(train_path, "image", "*png")))#возвращает 2 массива путей
train_y = sorted(glob(os.path.join(train_path, "mask", "*png")))
train_x, train_y = shuffle(train_x, train_y, random_state=42) # перемешивание

valid_x = sorted(glob(os.path.join(valid_path, "image", "*png")))#возвращает 2 массива путей
valid_y = sorted(glob(os.path.join(valid_path, "mask", "*png")))

print(f"Train: {len(train_x)} - {len(train_y)}")
print(f"Valid: {len(valid_x)} - {len(valid_y)}")


train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_ds = train_ds.map(lambda x, y: tf.py_function(function_for_map, [x, y], [tf.float32, tf.int32]))
#мап принимает функцию и отрабатывает по всему датасету
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(10)#предзагрузка в озу

valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
valid_ds = valid_ds.map(lambda x, y: tf.py_function(function_for_map, [x, y], [tf.float32, tf.int32]))
#функция мап принимает функцию и отрабатывает ее всему датасету
valid_ds = valid_ds.batch(batch_size)
valid_ds = valid_ds.prefetch(10)

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
# train_ds = train_ds.batch(batch_size)
# train_ds = train_ds.prefetch(10) # предзагрузка датасета в ОЗУ
# val_ds = raw_val_ds.map(binary_vectorize_text)
# val_ds = train_ds.batch(batch_size)
# val_ds = val_ds.prefetch(10)


# Загрузить модель с файла
# model = tf.keras.models.load_model('my_model.h5')
# model.summary() # Напечатать свойства

# Fine-tune
#for layer in model.layers:
#    if layer.name == 'block5_conv1':
#    	layer.trainable = False

model = MyModel() #если создана классом
model.compile(loss=losses.SparseCategoricalCrossentropy, optimizer=tf.keras.optimizers.Adam(lr), metrics=[metrics.Recall(), metrics.Precision()])

callbacks = [
callbacks.ModelCheckpoint(model_path, verbose=1, save_best_only=True),
callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
callbacks.CSVLogger(csv_path),
callbacks.TensorBoard(),
callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
]

model.fit(
train_ds,
epochs=num_epochs,
validation_data=valid_ds,
callbacks=callbacks
)

model.save("model.h5")
