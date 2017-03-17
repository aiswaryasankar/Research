# Placeholders for data
# X matrix is a 100x3 matrix
# y matrix is a 100x1 column

# Thus if I have a neural net, I have w0 = 3x2, w1 = 2x1

X1 = np.linspace(1, 10, 100)
X2 = np.linspace(-5, 24, 100)
X3 = np.linspace(134, 136, 100)
X = np.vstack((X1, X2))
X = np.vstack((X, X3)).T

Y = [np.random.randn() * 5 * i for i in range(100)]
for i in range(5):
    print(X[i])

# Initialize placeholders 
X = tf.placeholder('float', [100, 3])
Y = tf.placeholder('float', [100, 1])

# Initialize weights
def initialize_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Create model
def model(X, w1, w2):
    first_layer_output = tf.nn.sigmoid(tf.matmul(X, w1))
    print(first_layer_output)
    second_layer_output = tf.matmul(first_layer_output, w2)
    print(second_layer_output)
    return second_layer_output

# Initialize a model
w1 = initialize_weights((3, 2))
w2 = initialize_weights((2, 1))
pred_model = model(X, w1, w2)

# Write a cost function
cost = tf.square(Y - pred_model)
print('cost is')
print(cost)

# Optimizer Stochastic Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost)


# Activate session
with tf.Session() as sess:
    # Initialize all variables
    tf.initialize_all_variables().run()
    
    for i in range(100):
        # Run the optimizer on the data points
        sess.run(optimizer, {X:X[i], Y:Y[i]})

    print(sess.run(w1, w2))

