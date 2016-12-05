import os, datetime
import numpy as np
import tensorflow as tf
from DataLoader import *

# Dataset Parameters
batch_size = 200
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 100000
step_display = 1
step_save = 10000
path_save = 'alexnet'
start_from = ''

def alexnet(x, keep_dropout):
    weights = {
        'wc1_t': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2_t': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3_t': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4_t': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5_t': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
        
        'wc1_b': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2_b': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3_b': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4_b': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5_b': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),

        'wf6_tt': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf6_tb': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf6_bt': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf6_bb': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf7_tt': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wf7_tb': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wf7_bt': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wf7_bb': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        
        'wo_t': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096))),
        'wo_b': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bc1_t': tf.Variable(tf.zeros(96)),
        'bc2_t': tf.Variable(tf.zeros(256)),
        'bc3_t': tf.Variable(tf.zeros(384)),
        'bc4_t': tf.Variable(tf.zeros(256)),
        'bc5_t': tf.Variable(tf.zeros(256)),
        
        'bc1_b': tf.Variable(tf.zeros(96)),
        'bc2_b': tf.Variable(tf.zeros(256)),
        'bc3_b': tf.Variable(tf.zeros(384)),
        'bc4_b': tf.Variable(tf.zeros(256)),
        'bc5_b': tf.Variable(tf.zeros(256)),

        'bf6_t': tf.Variable(tf.zeros(4096)),
        'bf6_b': tf.Variable(tf.zeros(4096)),
        'bf7_t': tf.Variable(tf.zeros(4096)),
        'bf7_b': tf.Variable(tf.zeros(4096)),
        
        'bo': tf.Variable(tf.zeros(100))
    }

    ###############################################################################################
    #### BRANCH 1
    
    # Conv + ReLU + LRN + Pool, 224->55->27
    conv1_t = tf.nn.conv2d(x, weights['wc1_t'], strides=[1, 4, 4, 1], padding='SAME')
    conv1_t = tf.nn.relu(tf.nn.bias_add(conv1_t, biases['bc1_t']))
    lrn1_t = tf.nn.local_response_normalization(conv1_t, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool1_t = tf.nn.max_pool(lrn1_t, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU + LRN + Pool, 27-> 13
    conv2_t = tf.nn.conv2d(pool1_t, weights['wc2_t'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_t = tf.nn.relu(tf.nn.bias_add(conv2_t, biases['bc2_t']))
    lrn2_t = tf.nn.local_response_normalization(conv2_t, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool2_t = tf.nn.max_pool(lrn2_t, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3_t = tf.nn.conv2d(pool2_t, weights['wc3_t'], strides=[1, 1, 1, 1], padding='SAME')
    conv3_t = tf.nn.relu(tf.nn.bias_add(conv3_t, biases['bc3_t']))

    # Conv + ReLU, 13-> 13
    conv4_t = tf.nn.conv2d(conv3_t, weights['wc4_t'], strides=[1, 1, 1, 1], padding='SAME')
    conv4_t = tf.nn.relu(tf.nn.bias_add(conv4_t, biases['bc4_t']))

    # Conv + ReLU + Pool, 13->6
    conv5_t = tf.nn.conv2d(conv4_t, weights['wc5_t'], strides=[1, 1, 1, 1], padding='SAME')
    conv5_t = tf.nn.relu(tf.nn.bias_add(conv5_t, biases['bc5_t']))
    pool5_t = tf.nn.max_pool(conv5_t, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    ###############################################################################################
    #### BRANCH 2
    
    # Conv + ReLU + LRN + Pool, 224->55->27
    conv1_b = tf.nn.conv2d(x, weights['wc1_b'], strides=[1, 4, 4, 1], padding='SAME')
    conv1_b = tf.nn.relu(tf.nn.bias_add(conv1_b, biases['bc1_b']))
    lrn1_b = tf.nn.local_response_normalization(conv1_b, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool1_b = tf.nn.max_pool(lrn1_b, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU + LRN + Pool, 27-> 13
    conv2_b = tf.nn.conv2d(pool1_b, weights['wc2_b'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_b = tf.nn.relu(tf.nn.bias_add(conv2_b, biases['bc2_b']))
    lrn2_b = tf.nn.local_response_normalization(conv2_b, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool2_b = tf.nn.max_pool(lrn2_b, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3_b = tf.nn.conv2d(pool2_b, weights['wc3_b'], strides=[1, 1, 1, 1], padding='SAME')
    conv3_b = tf.nn.relu(tf.nn.bias_add(conv3_b, biases['bc3_b']))

    # Conv + ReLU, 13-> 13
    conv4_b = tf.nn.conv2d(conv3_b, weights['wc4_b'], strides=[1, 1, 1, 1], padding='SAME')
    conv4_b = tf.nn.relu(tf.nn.bias_add(conv4_b, biases['bc4_b']))

    # Conv + ReLU + Pool, 13->6
    conv5_b = tf.nn.conv2d(conv4_b, weights['wc5_b'], strides=[1, 1, 1, 1], padding='SAME')
    conv5_b = tf.nn.relu(tf.nn.bias_add(conv5_b, biases['bc5_b']))
    pool5_b = tf.nn.max_pool(conv5_b, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    ###############################################################################################
    #### Interconnecting branch
    
    pool5_t_temp = tf.reshape(pool5_t, [-1, weights['wf6_tt'].get_shape().as_list()[0]])
    pool5_b_temp = tf.reshape(pool5_b, [-1, weights['wf6_bt'].get_shape().as_list()[0]])
    
    # FC + ReLU + Dropout
    fc6_t = tf.add(tf.matmul(pool5_t_temp, weights['wf6_tb']) + tf.matmul(pool5_b_temp, weights['wf6_tb']), biases['bf6_t'])
    fc6_t = tf.nn.relu(fc6_t)
    fc6_t = tf.nn.dropout(fc6_t, keep_dropout)
    
    # FC + ReLU + Dropout
    fc6_b = tf.add(tf.matmul(pool5_b_temp, weights['wf6_bt']) + tf.matmul(pool5_b_temp, weights['wf6_bb']), biases['bf6_b'])
    fc6_b = tf.nn.relu(fc6_b)
    fc6_b = tf.nn.dropout(fc6_b, keep_dropout)
    
    # FC + ReLU + Dropout
    fc7_t = tf.add(tf.matmul(fc6_t, weights['wf7_tt']) + tf.matmul(fc6_b, weights['wf7_tb']), biases['bf7_t'])
    fc7_t = tf.nn.relu(fc7_t)
    fc7_t = tf.nn.dropout(fc7_t, keep_dropout)
    
    # FC + ReLU + Dropout
    fc7_b = tf.add(tf.matmul(fc6_t, weights['wf7_bt']) + tf.matmul(fc6_b, weights['wf7_bb']), biases['bf7_b'])
    fc7_b = tf.nn.relu(fc7_b)
    fc7_b = tf.nn.dropout(fc7_b, keep_dropout)

    ###############################################################################################
    #### Singleton branch
    
    # Output FC
    out = tf.add(tf.matmul(fc7_t, weights['wo_t']) + tf.matmul(fc7_b, weights['wo_b']), biases['bo'])
    
    return out

# Construct dataloader
opt_data_train = {
    'data_h5': 'miniplaces_256_train.h5',
    #'data_root': 'YOURPATH/images/',   # MODIFY PATH ACCORDINGLY
    #'data_list': 'YOURPATH/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    'data_h5': 'miniplaces_256_val.h5',
    #'data_root': 'YOURPATH/images/',   # MODIFY PATH ACCORDINGLY
    #'data_list': 'YOURPATH/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

# loader_train = DataLoaderDisk(**opt_data_train)
# loader_val = DataLoaderDisk(**opt_data_val)
loader_train = DataLoaderH5(**opt_data_train)
loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)

# Construct model
logits = alexnet(x, keep_dropout)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

# define initialization
init = tf.initialize_all_variables()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

# Launch the graph
with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)
    
    step = 0

    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)
        
        if step % step_display == 0:
            print '[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # Calculate batch loss and accuracy on training set
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.}) 
            print "-Iter " + str(step) + ", Training Loss= " + \
            "{:.4f}".format(l) + ", Accuracy Top1 = " + \
            "{:.2f}".format(acc1) + ", Top5 = " + \
            "{:.2f}".format(acc5)

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1.}) 
            print "-Iter " + str(step) + ", Validation Loss= " + \
            "{:.4f}".format(l) + ", Accuracy Top1 = " + \
            "{:.2f}".format(acc1) + ", Top5 = " + \
            "{:.2f}".format(acc5)
        
        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout})
        
        step += 1
        
        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print "Model saved at Iter %d !" %(step)
        
    print "Optimization Finished!"


    # Evaluate on the whole validation set
    print 'Evaludation on the whole validation set...'
    num_batch = loader_val.size()/batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)    
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.})
        acc1_total += acc1
        acc5_total += acc5
        print "Validation Accuracy Top1 = " + \
            "{:.2f}".format(acc1) + ", Top5 = " + \
            "{:.2f}".format(acc5)

    acc1_total /= num_batch
    acc5_total /= num_batch
    print 'Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total)
