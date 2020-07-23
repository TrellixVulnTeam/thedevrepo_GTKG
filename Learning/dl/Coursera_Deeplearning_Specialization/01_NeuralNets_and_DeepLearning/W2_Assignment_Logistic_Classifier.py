import numpy as np

"""
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
"""


def sigmoid(z):
    s = 1. / (1 + np.exp(-z))
    return s


print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))

"""
This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

Argument:
dim -- size of the w vector we want (or number of parameters in this case)

Returns:
w -- initialized vector of shape (dim, 1)
b -- initialized scalar (corresponds to the bias)
"""


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    return w, b


dim = 2
w, b = initialize_with_zeros(dim)
print("w = " + str(w))
print("b = " + str(b))

"""
Implement the cost function and its gradient for the propagation explained above

Arguments:
w -- weights, a numpy array of size (num_px * num_px * 3, 1)
b -- bias, a scalar
X -- data of size (num_px * num_px * 3, number of examples)
y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

Return:
J -- negative log-likelihood cost for logistic regression
dw -- gradient of the loss with respect to w, thus same shape as w
db -- gradient of the loss with respect to b, thus same shape as b

Tips:
- Write your code step by step for the propagation. np.log(), np.dot()
"""


def propagate(w, b, X, y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    J = (-1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A), axis=1, keepdims=True)
    dw = (1 / m) * np.dot(X, (A - y).T)
    db = (1 / m) * np.sum(A - y, axis=1, keepdims=True)

    gradients = {'dw': dw, 'db': db}
    return gradients, J


w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))

"""
This function optimizes w and b by running a gradient descent algorithm

Arguments:
w -- weights, a numpy array of size (num_px * num_px * 3, 1)
b -- bias, a scalar
X -- data of shape (num_px * num_px * 3, number of examples)
y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
n_iter -- number of iterations of the optimization loop
lr -- learning rate of the gradient descent update rule
verbose -- True to print the loss every 100 steps

Returns:
params -- dictionary containing the weights w and bias b
grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

Tips:
You basically need to write down two steps and iterate through them:
    1) Calculate the cost and the gradient for the current parameters. Use propagate().
    2) Update the parameters using gradient descent rule for w and b.
"""


def optimize(w, b, X, y, n_iter=500, lr=0.005, verbose=False):
    costs = []

    for i in range(n_iter):
        grads, J = propagate(w, b, X, y)

        dw = grads['dw']
        db = grads['db']

        w = w - lr * dw
        b = b - lr * db

        if i % 100 == 0:
            if verbose:
                print("Cost after #{}: {}".format(i, J))
            costs.append(J)

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}

    return params, grads, costs


params, grads, costs = optimize(w, b, X, Y, n_iter=200, lr=0.009, verbose=True)

print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))

'''
Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

Arguments:
w -- weights, a numpy array of size (num_px * num_px * 3, 1)
b -- bias, a scalar
X -- data of size (num_px * num_px * 3, number of examples)

Returns:
y_pred -- a numpy array (vector) containing all predictions (0/1) for the examples in X
'''


def predict(w, b, X):
    m = X.shape[1]
    w = w.reshape(X.shape[0], 1)
    y_pred = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        y_pred[0, i] = np.round(A[0, i])

    return y_pred


w = np.array([[0.1124579], [0.23106775]])
b = -0.3
X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
print("predictions = " + str(predict(w, b, X)))
