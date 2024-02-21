import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import os

train_path_fake = []

train_fake_base_1 = r".\ai-photos\train\FAKE"

print(f'Length train_fake_base_1 {len(os.listdir(train_fake_base_1))}')
length_train_path_fake = len(os.listdir(train_fake_base_1))
for i in range(0, length_train_path_fake):
    i = i + 1
    a = '{}{}.jpg'.format(train_fake_base_1 + "/", "image " + "(" + str(i) + ")")
    print(f'i: {i} a: {a}')
    train_path_fake.append(a)
print(f'Length appended {len(train_path_fake)}')




# for i in range(1000, 6000):
#     a = '{}{}.jpg'.format(train_fake_base_1 + "/", i)
#     print(f'i: {i} a: {a}')
#     train_path_fake.append(a)
#
# for i in range(1000, 6000):
#     for j in range(2, 11):
#         b = '{}{} ({}).jpg'.format(train_fake_base_1 + "/", i, j)
#         print(f'j: {j} b: {b}')
#         train_path_fake.append(b)
# len(train_path_fake)

df_train_fake = pd.DataFrame(train_path_fake)
df_train_fake.columns = ['path']
df_train_fake['label'] = 0
df_train_fake.head()

train_path_real = []
train_real_base_1 = r".\ai-photos\train\REAL"

print(f'Length train_real_base_1 {len(os.listdir(train_real_base_1))}')
length_train_real_base = len(os.listdir(train_real_base_1))
for j in range(0, length_train_real_base):
    j = j + 1
    b = '{}{}.jpg'.format(train_real_base_1 + "/", "image " + "(" + str(j) + ")")
    # print(f'j: {j} b: {b}')
    train_path_real.append(b)

print(f'Length appended {len(train_path_real)}')


# for i in range(0, 10):
#     c = '{}000{}.jpg'.format(train_real_base_1 + "/", i)
#     print(f'i: {i} c: {c}')
#     train_path_real.append(c)
#
# for i in range(0, 10):
#     for j in range(2, 11):
#         d = '{}000{} ({}).jpg'.format(train_real_base_1 + "/", i, j)
#         # print(f'i: {i} j: {j} d: {d}')
#         train_path_real.append(d)
# len(train_path_real)
#
# for i in range(10, 100):
#     e = '{}00{}.jpg'.format(train_real_base_1 + "/", i)
#     train_path_real.append(e)
#
# for i in range(10, 100):
#     for j in range(2, 11):
#         f = '{}00{} ({}).jpg'.format(train_real_base_1 + "/", i, j)
#         train_path_real.append(f)
# len(train_path_real)
#
# for i in range(100, 1000):
#     g = '{}0{}.jpg'.format(train_real_base_1 + "/", i)
#     train_path_real.append(g)
#
# for i in range(100, 1000):
#     for j in range(2, 11):
#         h = '{}0{} ({}).jpg'.format(train_real_base_1 + "/", i, j)
#         train_path_real.append(h)
# len(train_path_real)
#
# for i in range(1000, 5000):
#     k = '{}{}.jpg'.format(train_real_base_1 + "/", i)
#     train_path_real.append(k)
#
# for i in range(1000, 5000):
#     for j in range(2, 11):
#         l = '{}{} ({}).jpg'.format(train_real_base_1 + "/", i, j)
#         train_path_real.append(l)
# len(train_path_real)

df_train_real = pd.DataFrame(train_path_real)
df_train_real.columns = ['path']
df_train_real['label'] = 1
df_train_real.head()

df_train = pd.concat((df_train_fake, df_train_real), axis=0)
print(df_train.shape)
df_train.sample(5)

image_exts = ['jpeg', 'jpg', 'bmp', 'png']
# USING CV2 TO CREATE X_TRAIN
image_df = []
for image in df_train['path']:
    img = cv2.imread(image)
    resized = cv2.resize(img, (32, 32))
    image_df.append(resized)
image_array = np.array(image_df)
x_train = image_array / 255
x_train.ndim

y_train = df_train['label']
y_train.head()

test_path_fake = []
test_fake_base_1 = r".\ai-photos\test\FAKE"

print(f'Length test_fake_base_1 {len(os.listdir(test_fake_base_1))}')
length_test_fake_base = len(os.listdir(test_fake_base_1))
for k in range(0, length_test_fake_base):
    k = k + 1
    c = '{}{}.jpg'.format(test_fake_base_1 + "/", "image " + "(" + str(k) + ")")
    print(f'k: {k} c: {c}')
    test_path_fake.append(c)

print(f'Length appended {len(test_path_fake)}')


# for i in range(0, 1000):
#     m = '{}{}.jpg'.format(test_fake_base_1 + "/", i)
#     test_path_fake.append(m)
#
# for i in range(0, 1000):
#     for j in range(2, 11):
#         n = '{}{} ({}).jpg'.format(test_fake_base_1 + "/", i, j)
#         test_path_fake.append(n)
# len(test_path_fake)

df_test_fake = pd.DataFrame(test_path_fake)
df_test_fake.columns = ['path']
df_test_fake['label'] = 0
df_test_fake.head()

test_path_real = []

test_real_base_1 = r".\ai-photos\test\REAL"

print(f'Length test_real_base_1 {len(os.listdir(test_real_base_1))}')
length_test_real_base = len(os.listdir(test_real_base_1))
for l in range(0, length_test_real_base):
    l = l + 1
    d = '{}{}.jpg'.format(test_real_base_1 + "/", "image " + "(" + str(l) + ")")
    print(f'l: {l} d: {d}')
    test_path_real.append(d)

print(f'Length appended {len(test_path_real)}')

# for i in range(0, 10):
#     o = '{}000{}.jpg'.format(test_real_base_1 + "/", i)
#     test_path_real.append(o)
#
# for i in range(0, 10):
#     for j in range(2, 11):
#         p = '{}000{} ({}).jpg'.format(test_real_base_1 + "/", i, j)
#         test_path_real.append(p)
# len(test_path_real)
#
# for i in range(10, 100):
#     q = '{}00{}.jpg'.format(test_real_base_1 + "/", i)
#     test_path_real.append(q)
#
# for i in range(10, 100):
#     for j in range(2, 11):
#         r = '{}00{} ({}).jpg'.format(test_real_base_1 + "/", i, j)
#         test_path_real.append(r)
# len(test_path_real)
#
# for i in range(100, 1000):
#     s = '{}0{}.jpg'.format(test_real_base_1 + "/", i)
#     test_path_real.append(s)
#
# for i in range(100, 1000):
#     for j in range(2, 11):
#         t = '{}0{} ({}).jpg'.format(test_real_base_1 + "/", i, j)
#         test_path_real.append(t)
# len(test_path_real)

df_test_real = pd.DataFrame(test_path_real)
df_test_real.columns = ['path']
df_test_real['label'] = 1
df_test_real.head()

# USING CONCAT TO CREATE DF TEST
df_test = pd.concat((df_test_fake, df_test_real), axis=0)
print(df_test.shape)
df_test.sample(5)

# USING CV2 TO CREATE X_TEST
image_ds = []
for image in df_test['path']:
    imge = cv2.imread(image)
    resize = cv2.resize(imge, (32, 32))
    image_ds.append(resize)
image_arry = np.array(image_ds)
X_test = image_arry / 255
X_test.ndim

y_test = df_test['label']
y_test.head()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=80, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(filters=40, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid'),

])
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.1, nesterov=False)
optimizer = tf.keras.optimizers.Adamax(learning_rate=0.00001)

# Adam learning rate defaults to 0.001 | SDG to 0.01
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)

model.fit(
    x=x_train,
    y=y_train,
    epochs=18,
    batch_size=10
)

model.save("model29.h5")
