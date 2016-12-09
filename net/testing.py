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


# filt = [[1.0/16, 2.0/16, 1.0/16],
#         [2.0/16, 4.0/16, 2.0/16],
#         [1.0/16, 2.0/16, 1.0/16]]
filt = [[1.0/16, 2.0/16, 1.0/16],
        [2.0/16, 4.0/16, 2.0/16],
        [1.0/16, 2.0/16, 1.0/16]]
const = tf.constant([[filt for i in range(3)] for j in range(3)], dtype=np.float64)
# norm = tf.random_normal([3, 3, 3, 3], stddev=np.sqrt(2. / (11 * 11 * 3)))




# tf Graph input
img_tf = tf.Variable(images_batch, [None, fine_size, fine_size, c], dtype=np.float64)

# Apply blur
out = tf.nn.conv2d(img_tf, const, strides=[1, 1, 1, 1], padding='SAME')
# out = tf.nn.avg_pool(img_tf, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")



init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
new_imgs_batch = sess.run(out)

kernel = sess.run(const)
print kernel


plt.imshow(images_batch[0])
plt.show()
plt.imshow(new_imgs_batch[0])
plt.show()
