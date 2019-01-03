import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mnist = input_data.read_data_sets('E:/MNIST', one_hot=True)

lr = 0.4
num_steps = 200
batch_size = 128
temperature = 10

X = tf.placeholder(tf.float32, shape=[None, 784])
X_img = tf.reshape(X, shape=[-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob_conv = tf.placeholder(tf.float32)
keep_prob_hidden = tf.placeholder(tf.float32)

def TeacherNetwork(inputs, keep_prob_conv, keep_prob_hidden, scope='Mnist', reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], stride=[1, 1], biases_initializer=tf.constant_initializer(0.0), activation_fn=tf.nn.relu):

            net = slim.conv2d(inputs, 32, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
            net = tf.nn.dropout(net, keep_prob_conv)

            net = slim.conv2d(net, 64, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
            net = tf.nn.dropout(net, keep_prob_conv)

            net = slim.conv2d(net, 128, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
            net = tf.nn.dropout(net, keep_prob_conv)

            net = slim.flatten(net)
            
        with slim.arg_scope([slim.fully_connected], biases_initializer=tf.constant_initializer(0.0), activation_fn=tf.nn.relu):
            
            net = slim.fully_connected(net, 625, scope='fc1')
            net = slim.fully_connected(net, 10, activation_fn=None, scope='fc2')
            
            return net
            
def StudentNetwork(inputs, scope='Mnist', reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as sc:
        with slim.arg_scope([slim.fully_connected], biases_initializer=tf.constant_initializer(0.0), activation_fn=tf.nn.sigmoid):
            
            net = slim.fully_connected(inputs, 800, scope='fc1')
            net = slim.fully_connected(net, 800, scope='fc2')
            net = slim.fully_connected(net, 10, activation_fn=None, scope='fc3')
            
            return net

def loss_accuracy(prediction, label):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=label))
    is_correction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correction, tf.float32))
    
    return cross_entropy, accuracy
    
logit_teacher = TeacherNetwork(X_img, keep_prob_conv, keep_prob_hidden, scope='teacher')
logit_student = StudentNetwork(X, scope='student')

logit_teacher_tem = logit_teacher / temperature
logit_student_tem = logit_student / temperature

output_student_tem = tf.nn.softmax(logit_student / temperature)
output_student = tf.nn.softmax(logit_student)

cross_entropy_teacher, accuracy_teacher = loss_accuracy(logit_teacher, Y)
cross_entropy1_student, _ = loss_accuracy(logit_student_tem, tf.nn.softmax(logit_teacher_tem))

loss_teacher_summ = tf.summary.scalar('loss_teacher', cross_entropy_teacher)

cross_entropy2_student, accuracy_student = loss_accuracy(logit_student, Y)

cross_entropy_student = cross_entropy1_student + cross_entropy2_student
loss_teacher_summ = tf.summary.scalar('loss_student', cross_entropy_student)

model_variables = tf.trainable_variables()
var_teacher = [var for var in model_variables if 'teacher' in var.name]
var_student = [var for var in model_variables if 'student' in var.name]

gradient_teacher = tf.gradients(cross_entropy_teacher, var_teacher)
gradient_student = tf.gradients(cross_entropy_student, var_student)

optimizer_teacher = tf.train.GradientDescentOptimizer(learning_rate=lr)
optimizer_student = tf.train.GradientDescentOptimizer(learning_rate=lr)

train_teacher = optimizer_teacher.apply_gradients(zip(gradient_teacher, var_teacher))
train_student = optimizer_student.apply_gradients(zip(gradient_student, var_student))

merged_summary = tf.summary.merge_all()

with tf.Session() as sess:

    writer = tf.summary.FileWriter('./logs')
    writer.add_graph(sess.graph)
    
    sess.run(tf.global_variables_initializer())
    saver1 = tf.train.Saver(var_teacher)
    saver2 = tf.train.Saver(var_student)
    
    for i in range(num_steps):
        mnist_batch = mnist.train.next_batch(batch_size)
        batch_xs, batch_ys = mnist_batch[0], mnist_batch[1]
        
        if i % 50 == 0:
            train_teacher_accuracy = sess.run(accuracy_teacher, feed_dict={X: batch_xs, Y: batch_ys, keep_prob_conv: 1.0, keep_prob_hidden: 1.0})
            print('Step %d, teacher train_accuracy %s'%(i, train_teacher_accuracy))
        _, summary = sess.run([train_teacher, merged_summary], feed_dict={X: batch_xs, Y: batch_ys, keep_prob_conv: 0.8, keep_prob_hidden: 0.5})
        writer.add_summary(summary, global_step=i)
        
    saver1.save(sess, './models/teacher.ckpt')
    
    
    for i in range(num_steps):
        mnist_batch = mnist.train.next_batch(batch_size)
        batch_xs, batch_ys = mnist_batch[0], mnist_batch[1]
       
        if i % 50 == 0:
            train_student_accuracy = sess.run(accuracy_student, feed_dict={X: batch_xs, Y: batch_ys, keep_prob_conv: 1.0, keep_prob_hidden: 1.0})
            print('Step %d, student train_accuracy %s'%(i, train_student_accuracy))
        _ = sess.run(train_student, feed_dict={X: batch_xs, Y: batch_ys, keep_prob_conv: 1.0, keep_prob_hidden: 1.0})
        
    saver2.save(sess, './models/student.ckpt')