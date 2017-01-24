import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import time

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Define encoder network
#data -> (mu, sigma)

config = tf.ConfigProto( device_count = {'GPU':1}, log_device_placement=True)
sess = tf.Session(config = config)
# sess = tf.Session()
#Weight initialization
n_hidden1 = 200
n_hidden2 = 200

#Weights for encoder network
with tf.variable_scope("enc"):
	enc_w1 = tf.get_variable('w1', shape=[784, n_hidden1], initializer=xavier_initializer())
	enc_b1 = tf.get_variable('b1', shape = [n_hidden1], initializer=xavier_initializer())
	enc_w2 = tf.get_variable('w2', shape=[n_hidden1,n_hidden2], initializer=xavier_initializer())
	enc_b2 = tf.get_variable('b2', shape = [n_hidden2], initializer=xavier_initializer())
	enc_w_mean = tf.get_variable('w_mean', shape=[n_hidden2,2], initializer=xavier_initializer())
	enc_b_mean = tf.get_variable('b_mean', shape=[2], initializer=xavier_initializer())
	enc_w_sigma = tf.get_variable('w_sigma', shape=[n_hidden2,2], initializer=tf.random_uniform_initializer(1e-5, 1e-3))
	enc_b_sigma = tf.get_variable('b_sigma', shape=[2], initializer=tf.random_uniform_initializer(1e-5, 1e-3))

#Weights for decoder network
with tf.variable_scope("dec"):
	dec_w1 = tf.get_variable('w1', shape=[2, n_hidden2], initializer=xavier_initializer())
	dec_b1 = tf.get_variable('b1', shape=[n_hidden2], initializer=xavier_initializer())
	dec_w2 = tf.get_variable('w2', shape=[n_hidden2, n_hidden1], initializer=xavier_initializer())
	dec_b2 = tf.get_variable('b2', shape=[n_hidden1], initializer=xavier_initializer())
	dec_w3 = tf.get_variable('w3', shape=[n_hidden1, 784], initializer=xavier_initializer())
	dec_b3 = tf.get_variable('b3', shape=[784], initializer=xavier_initializer())

x = tf.placeholder(tf.float32, shape=[None, 784]) #Data


def encoder_network(x):
	with tf.variable_scope("enc"):

		layer1 = tf.nn.relu(tf.add(tf.matmul(x, enc_w1), enc_b1))
		layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, enc_w2), enc_b2))

		z_mean = tf.add(tf.matmul(layer2, enc_w_mean), enc_b_mean)
		z_log_sigma_sq = tf.add(tf.matmul(layer2, enc_w_sigma), enc_b_sigma)

		return (z_mean, z_log_sigma_sq)

def decoder_network(z):
	with tf.variable_scope("dec"):
		layer1 = tf.nn.relu(tf.add(tf.matmul(z, dec_w1), dec_b1))
		layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, dec_w2), dec_b2))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, dec_w3), dec_b3))

		return layer3

def train():
	e = tf.random_normal([2]) #Reparameterization trick

	z_mean, z_log_sigma_sq = encoder_network(x)
	z_sample = z_mean + z_log_sigma_sq * e

	x0 = decoder_network(z_sample)

	# log_likelihood = -tf.reduce_sum(x * tf.log(1e-10 + x0) + (1-x) * tf.log(1e-10 + 1 - x0), 1)
	# negative_kl_divergence = -0.5 * tf.reduce_sum(1 + tf.log(1e-10 + tf.square(z_sample)) - tf.square(z_mean) - tf.square(z_sample))

	log_likelihood = -tf.reduce_sum(x * tf.log(1e-10 + x0) + (1 - x) * tf.log(1e-10 + 1 - x0), 1)
	negative_kl_divergence = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
	neg_variational_bound = tf.reduce_mean(log_likelihood + negative_kl_divergence)
	# neg_variational_bound = tf.reduce_mean(-(log_likelihood))


	optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(neg_variational_bound)



	return (optimizer, neg_variational_bound)

optimizer, bound = train()
training_cost = tf.scalar_summary('cost', bound)
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('logs', sess.graph)
sess.run(tf.initialize_all_variables())


def fit(X):
	opt, cost = sess.run((optimizer, training_cost), feed_dict={x:X})
	return cost

def from_latent(Z): # Should be 2 dimensionsal
	# print("Sampling from: %s" % Z)
	# z_input = tf.Variable(z, dtype=tf.float32)
	z = tf.placeholder(tf.float32, shape=[None, 2]) #Latent

	out = sess.run(decoder_network(z), feed_dict={z:[Z]})
	img = np.reshape(out, (28,28))

	return img

def to_latent(X):
	x = tf.placeholder(tf.float32, shape=[None, 784])
	z_mean, z_sigma = sess.run(encoder_network(x), feed_dict={x:[X]})

	return (z_mean, z_sigma)


def reconstruction(X): # x is a numpy object.
	z_mean, z_sigma = to_latent(X)
	print("mean: %s, sigma: %s" %(z_mean, z_sigma))

	img = from_latent(z_mean)

	plt.matshow(img, cmap='Greys')
	plt.show()
	plt.clf()

	plt.matshow(np.reshape(out, (28,28)), cmap='Greys')
	plt.show()

def latent_manifold():
	X = np.linspace(-1.5, 1.5, 20)
	Y = np.linspace(-1.5, 1.5, 20)

	def getRow(y, xlin):
		row = from_latent([xlin[0], y])
		for x in xlin[1:]:
			row = np.hstack((row, from_latent([x, y])))
		return row

	print("Row")
	full = getRow(Y[0], X)
	for y in Y[1:]:
		print("Row")
		full = np.vstack((getRow(y, X), full))

	plt.matshow(full, cmap='Greys')
	# plt.show()

mnist_batch = mnist.train.next_batch(100)

def project_to_manifold(mnist_batch):
	X, labels = mnist_batch
	raw = list(map(lambda x:to_latent(x)[0][0], X))
	data = list(zip(*raw))
	x = list(data[0])
	y = list(data[1])
	labels = list(map(lambda a:list(a).index(1), labels))

	plt.scatter(x, y)
	for i, txt in enumerate(labels):
		plt.annotate(str(txt), (x[i], y[i]))

	# plt.show()


start_time = time.time()

print("Training Begin")

for i in range(20000):
	batch = mnist.train.next_batch(100)
	cost = fit(batch[0])
	# if i % 50 == 0:
	# 	project_to_manifold(mnist_batch)

	if i % 100 == 0:
		writer.add_summary(cost, i)
		# merged = tf.merge_all_summaries()
		# train_writer = tf.train.SummaryWriter('logs', sess.graph)

		print("Epoch: %s" % i)
		# print(cost)

print("Final cost:")
print (cost)

print(time.time() - start_time)
# project_to_manifold(mnist_batch)


# print("Latent:")
# vis(np.array([[0.023, 0.31]]))
# X = mnist.train.next_batch(1)[0]
# reconstruction(X)

# latent_manifold()
