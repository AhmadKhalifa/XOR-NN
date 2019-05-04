import numpy as np

INPUT_NEURONS_NUMBER = 2
HIDDEN_NEURONS_NUMBER = 2
OUTPUT_NEURONS_NUMBER = 1
ITERATIONS_COUNT = 1000
LEARNING_RATE = 0.3


def sigmoid_activation_function(x):
    return 1 / (1 + np.exp(-x))


def init_params(n_x, n_h, n_y):
    return {
        "w1": np.random.randn(n_h, n_x),
        "b1": np.zeros((n_h, 1)),
        "w2": np.random.randn(n_y, n_h),
        "b2": np.zeros((n_y, 1))
    }


def forward_prop(x, parameters):
    a1 = np.tanh(np.dot(parameters["w1"], x) + parameters["b1"])
    a2 = sigmoid_activation_function(np.dot(parameters["w2"], a1) + parameters["b2"])
    return a2, {
        "a1": a1,
        "a2": a2
    }


def calculate_cost(a2, y):
    return np.squeeze(-np.sum(np.multiply(y, np.log(a2)) + np.multiply(1 - y, np.log(1 - a2))) / m)


def backward_prop(x, y, cache, parameters):
    return {
        "dw1": np.dot(np.multiply(np.dot(parameters["w2"].T, cache["a2"] - y), 1 - np.power(cache["a1"], 2)), x.T) / m,
        "db1": np.sum(np.multiply(np.dot(parameters["w2"].T, cache["a2"] - y), 1 - np.power(cache["a1"], 2)), axis=1,
                      keepdims=True) / m,
        "dw2": np.dot(cache["a2"] - y, cache["a1"].T) / m,
        "db2": np.sum(cache["a2"] - y, axis=1, keepdims=True) / m
    }


def update_parameters(parameters, grads, learning_rate):
    return {
        "w1": parameters["w1"] - learning_rate * grads["dw1"],
        "w2": parameters["w2"] - learning_rate * grads["dw2"],
        "b1": parameters["b1"] - learning_rate * grads["db1"],
        "b2": parameters["b2"] - learning_rate * grads["db2"]
    }


def model(x, y):
    parameters = init_params(INPUT_NEURONS_NUMBER, HIDDEN_NEURONS_NUMBER, OUTPUT_NEURONS_NUMBER)
    for i in range(0, ITERATIONS_COUNT + 1):
        a2, cache = forward_prop(x, parameters)
        parameters = update_parameters(parameters, backward_prop(x, y, cache, parameters), LEARNING_RATE)
        print('Cost in iteration {:d} = {:f}'.format(i, calculate_cost(a2, y)))
    return parameters


def predict(x, parameters):
    a2, cache = forward_prop(x, parameters)
    return 1 if np.squeeze(a2) >= 0.5 else 0


np.random.seed(2)
input = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
output = np.array([[0, 1, 1, 0]])
m = input.shape[1]
trained_parameters = model(input, output)
x_test = np.array([[1], [1]])
y_predict = predict(x_test, trained_parameters)
print("Successfully trained!")
print('{:d} XOR {:d} = {:d}'.format(x_test[1][0], y_predict, x_test[0][0]))
