import numpy as np
import tensorflow as tf

LABELS = 21
BATCHES = 100
EPOCHS = 1000;
HIDDEN_NODES = 20;

TRAIN_FILE = file("dataset_train.csv")
EVAL_FILE = file("dataset_eval.csv")

def onehot(label, length):
	arr = [0] * length
	arr[label] = 1

	return arr


def parse(input_file):
	labels = []
	inputs = []

	for line in input_file:
		row = line.split(",")

		labels.append(int(row[0]))
		inputs.append([float(x) for x in row[1:]])

	np_labels_pre = []
	np_inputs = np.array(inputs).astype(dtype=np.uint8)

	for label in labels:
		np_labels_pre.append(onehot(label, LABELS))

	np_labels = np.matrix(np_labels_pre).astype(np.float32)

	return np_inputs, np_labels


train_data, train_labels = parse(TRAIN_FILE);
eval_data, eval_labels = parse(EVAL_FILE);

train_size, features = train_data.shape

x = tf.placeholder("float", shape=[None, features])
y_ = tf.placeholder("float", shape=[None, LABELS])

eval_data_node = tf.constant(eval_data)

w_hidden = tf.Variable(tf.random_normal([features, HIDDEN_NODES], stddev=0.01, dtype=tf.float32))
b_hidden = tf.Variable(tf.zeros([1, HIDDEN_NODES], dtype=tf.float32))
hidden = tf.nn.tanh(tf.matmul(x, w_hidden) + b_hidden)

w_hidden_2 = tf.Variable(tf.random_normal([HIDDEN_NODES, HIDDEN_NODES], stddev=0.01, dtype=tf.float32))
b_hidden_2 = tf.Variable(tf.zeros([1, HIDDEN_NODES], dtype=tf.float32))
hidden_2 = tf.nn.tanh(tf.matmul(hidden, w_hidden_2) + b_hidden_2)

w_hidden_3 = tf.Variable(tf.random_normal([HIDDEN_NODES, HIDDEN_NODES], stddev=0.01, dtype=tf.float32))
b_hidden_3 = tf.Variable(tf.zeros([1, HIDDEN_NODES], dtype=tf.float32))
hidden_3 = tf.nn.tanh(tf.matmul(hidden_2, w_hidden_3) + b_hidden_3)

w_output = tf.Variable(tf.random_normal([HIDDEN_NODES, LABELS], stddev=0.01, dtype=tf.float32))
b_output = tf.Variable(tf.zeros([1, LABELS], dtype=tf.float32))
y = tf.nn.relu(tf.matmul(hidden_3, w_output) + b_output)


cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


with tf.Session() as session:
	tf.initialize_all_variables().run()
	for step in xrange(EPOCHS * train_size // BATCHES):
	    offset = (step * BATCHES) % train_size

	    batch_data = train_data[offset:(offset + BATCHES), :]
	    batch_labels = train_labels[offset:(offset + BATCHES)]

	    train_step.run(feed_dict={x: batch_data, y_: batch_labels})

	np.savetxt("w_hidden.csv", session.run(w_hidden), delimiter=",")
	np.savetxt("b_hidden.csv", session.run(b_hidden), delimiter=",")

	np.savetxt("w_hidden_2.csv", session.run(w_hidden_2), delimiter=",")
	np.savetxt("b_hidden_2.csv", session.run(b_hidden_2), delimiter=",")

	np.savetxt("w_hidden_3.csv", session.run(w_hidden_3), delimiter=",")
	np.savetxt("b_hidden_3.csv", session.run(b_hidden_3), delimiter=",")

	np.savetxt("w_output.csv", session.run(w_output), delimiter=",")
	np.savetxt("b_output.csv", session.run(b_output), delimiter=",")

	print "Accuracy:", accuracy.eval(feed_dict={x: eval_data, y_: eval_labels})

