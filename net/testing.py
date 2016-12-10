import matplotlib.pyplot as plt
import os, datetime
import numpy as np
import tensorflow as tf
from DylanDataLoader import *

# Dataset Parameters
batch_size = 1
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097, 0.44674252445, 0.41352266842])

# Construct dataloader
opt_data_train = {
    'data_h5': 'miniplaces_256_train.h5',
    # 'data_root': 'YOURPATH/images/',   # MODIFY PATH ACCORDINGLY
    # 'data_list': 'YOURPATH/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
}
loader_train = DataLoaderH5(**opt_data_train)

# Load Data
images_batch, labels_batch = loader_train.next_batch(batch_size)

# # Color filter
# filt = [[0, 0, 0],
#         [0, 1, 0],
#         [0, 0, 0]]
# const = tf.constant([[filt for i in range(1)] for j in range(1)], dtype=np.float32)

# # Normal filter
filt = [[0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]]
kernel = np.zeros([1, 3, 3, 3])
for i in range(3):
    for j in range(3):
        kernel[:, i, j, :] = filt[i][j]
print kernel
const = tf.constant(kernel, dtype=np.float32)

# tf Graph input
img_tf = tf.Variable(images_batch, [None, fine_size, fine_size, c], dtype=np.float32)

# # Change data
# out = tf.image.rgb_to_grayscale(img_tf, name=None)

# Apply blur
out = tf.nn.conv2d(img_tf, const, strides=[1, 1, 1, 1], padding='SAME')
# out = tf.nn.avg_pool(img_tf, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")



init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
filtered_batch = sess.run(out)

plt.imshow(images_batch[0])
plt.show()
print filtered_batch[0].shape
plt.imshow(filtered_batch[0])
plt.show()
