import tensorflow as tf
import config
import data
import network
import os
import numpy as np
import matplotlib.pyplot as plt
tf.reset_default_graph()
def img_show(X):
    #print(X.shape)
    for x in X:
        x = (x+1)/2
        #print(x.shape,x.max(),x.min())
        plt.imshow(x, cmap='Greys_r')
        break
    plt.show()

def get_samples(sess,networks,output):
    tf.local_variables_initializer()
    noise = np.random.uniform(-1,1,size = (config.batch_size, config.noise_size))
    
    samples = sess.run(output,feed_dict = {networks.y:noise})
    return samples
def train():
    networks = network.Network()
    datas = data.cifar_data()
    output = networks.generator(config.img_depth,1)
    g_loss, d_loss = networks.loss(config.img_depth)
    g_opt, d_opt = networks.optimizer(g_loss, d_loss)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoh in range(config.epoh):
            for step in range(500):
                X,Y = datas.next_batch()
                X = X * 2 - 1
                #img_show(X)
                noise = np.random.uniform(-1,1,size = (config.batch_size, config.noise_size))
                _ = sess.run(g_opt, feed_dict = {networks.x:X,
                                                        networks.y:noise})
                _ = sess.run(d_opt, feed_dict = {networks.x:X,
                                                       networks.y:noise})
                if step % 10 == 0:
                    samples = get_samples(sess,networks,output)
                    img_show(samples)
                loss_d = d_loss.eval({networks.x:X,
                                      networks.y:noise})
                loss_g = g_loss.eval({networks.x:X,
                                      networks.y:noise})
                print('\repoh: {}, steps: {}, d_loss: {}, g_loss: {}'.format(epoh,step,loss_d,loss_g), end="")

if __name__ == '__main__':
    train()