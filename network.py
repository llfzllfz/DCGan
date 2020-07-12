import tensorflow as tf
import config

class Network:
    def __init__(self):
        tf.reset_default_graph()
        self.batch_size = config.batch_size
        self.noise_size = config.noise_size
        self.img_height = config.img_height
        self.img_width = config.img_width
        self.img_depth = config.img_depth
        self.alpha = config.alpha
        self.learning_rate = config.learning_rate
        self.x = tf.placeholder('float', [self.batch_size,self.img_height,self.img_width,self.img_depth])
        self.y = tf.placeholder('float', [self.batch_size,self.noise_size])
    
    def link_relu(self,inputs):
        return tf.maximum(self.alpha * inputs, inputs)
    
    def generator(self,output_dim,flg = 0):
        with tf.variable_scope('generator', reuse = tf.AUTO_REUSE):
            dense1 = tf.layers.dense(self.y, 4*4*512)
            dense1 = tf.reshape(dense1, [-1,4,4,512])
            
            layer1 = tf.layers.conv2d_transpose(dense1, 256, kernel_size = 4,strides = 2,padding = 'same')
            bn1 = tf.layers.batch_normalization(layer1)
            relu1 = self.link_relu(bn1)
            
            layer2 = tf.layers.conv2d_transpose(relu1, 128, kernel_size = 3,strides = 2,padding = 'same')
            bn2 = tf.layers.batch_normalization(layer2)
            relu2 = self.link_relu(bn2)
            
            logits = tf.layers.conv2d_transpose(relu2, output_dim, kernel_size = 3,strides = 2,padding = 'same')
            output = tf.tanh(logits)
            return output
        
    def discriminator(self,inputs, reuse = False):
        with tf.variable_scope('discriminator', reuse = reuse):
            layers1 = tf.layers.conv2d(inputs, 128, 3, strides = 2, padding = 'same')
            bn1 = tf.layers.batch_normalization(layers1)
            relu1 = self.link_relu(bn1)
            
            layers2 = tf.layers.conv2d(relu1, 256, 3, 2, 'same')
            bn2 = tf.layers.batch_normalization(layers2)
            relu2 = self.link_relu(bn2)
            
            layers3 = tf.layers.conv2d(relu2, 512, 4, 2, 'same')
            bn3 = tf.layers.batch_normalization(layers3)
            relu3 = self.link_relu(bn3)
            
            flatten = tf.reshape(relu3, (-1,4*4*512))
            logits = tf.layers.dense(flatten, 1)
            outputs = tf.sigmoid(logits)
            return logits, outputs
    
    def loss(self, img_depth):
        smooth = 0.1
        g_outputs = self.generator(img_depth)
        d_logits_real, d_outputs_real = self.discriminator(self.x)
        d_logits_fake, d_outputs_fake = self.discriminator(g_outputs, reuse=True)
        
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, labels = tf.ones_like(d_outputs_fake)*(1-smooth)))
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real, labels = tf.ones_like(d_outputs_real)*(1-smooth)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, labels = tf.ones_like(d_outputs_fake)))
        d_loss = tf.add(d_loss_real,d_loss_fake)
        return g_loss, d_loss
        
    def optimizer(self,g_loss,d_loss):
        beta1 = 0.4
        train_vars = tf.trainable_variables()
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        g_opt = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1 = beta1).minimize(g_loss, var_list=gen_vars)
        d_opt = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1 = beta1).minimize(d_loss, var_list=dis_vars)
        return g_opt, d_opt
    
        
        