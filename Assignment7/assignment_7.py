import numpy as np


def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test

def sigmoid(x):
    #As defined on p. 803
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def loss_function(y_pred, y_actual) -> float:
    return 0.5*np.sum((y_pred-y_actual)**2)

def mean_squared_error(y_pred, y_actual) -> float:
    return np.mean((y_pred-y_actual)**2)

class NeuralNetwork:
    def __init__(self, learning_rate: float, iterations: int):
        np.random.seed(0)
        #Weights for the link between input and the two hidden nodes
        self.input_weights = np.random.randn(2, 2)
        #Weights for the link between the hidden nodes and the output node
        self.output_weights = np.random.randn(2, 1)
        self.learning_rate = learning_rate
        self.iterations = iterations

    def forward_hidden(self, X):
        return sigmoid(np.dot(X, self.input_weights))
    
    def forward_output(self, X):
        return np.dot(X, self.output_weights)
    
    def loss_gradient_hidden(self, y_pred, y_actual, hidden_output, X):
        #Need the gradient for each input weight
        gradient = np.zeros(self.input_weights.shape)
        #Iterate over each row in the 2x2 input weight matrix
        for i in range(self.input_weights.shape[0]):
            #Iterate over each weight in the row
            for j in range(self.input_weights.shape[1]):
                #from the second derivative term on p. 806. activation is linear for output layer and therefore its derivative is 1
                gradient[i, j] = np.sum(-2*(y_actual - y_pred) * self.output_weights[j] * sigmoid_derivative(hidden_output[:, j]) * X[:, i])
        return gradient


    def loss_gradient_output(self, y_pred, y_actual, hidden_output):
        #Need the gradient for each output weight
        gradient = np.zeros(self.output_weights.shape)
        for (i, w) in enumerate(self.output_weights):
            #from the first derivative term on p. 806
            gradient[i] = np.sum((-2*(y_actual - y_pred) * hidden_output[:, i]))
        return gradient
    

    def train(self, X, y):

        for i in range(self.iterations+1):
            
            # Forward pass

            #From input to hidden layer
            hidden_output = self.forward_hidden(X)
            #From hidden layer to output
            output = self.forward_output(hidden_output)

            #Calculate loss and print it every 10 iterations for monitoring
            loss = loss_function(output.flatten(), y)
            if i % 10 == 0:
                print(f"Iteration {i}, loss: {loss}")

            #Backward pass/Gradient descent

            #Calculate gradients for the input and output weights using the chain rule and the loss function
            input_gradient = self.loss_gradient_hidden(output.flatten(), y, hidden_output, X)
            output_gradient = self.loss_gradient_output(output.flatten(), y, hidden_output)

            #Update weights using the gradients and the learning rate in accordance with the gradient descent algorithm
            self.input_weights -= self.learning_rate * input_gradient
            self.output_weights -= self.learning_rate * output_gradient
                    
    #Predict the output for the given input - i.e. complete forward pass     
    def predict(self, X):
        hidden_output = self.forward_hidden(X)
        return self.forward_output(hidden_output).flatten()
        


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
        

    # TODO: Your code goes here.

    #Create a neural network with a learning rate of 0.007 and 250 iterations
    nn = NeuralNetwork(0.007, 250)
    #Train the neural network with the training data
    nn.train(X_train, y_train)
    
    
    # Make predictions on training and test sets
    y_train_pred = nn.predict(X_train)
    y_test_pred = nn.predict(X_test)

    #Calculate the mean squared error for the training and test sets
    mean_squared_error_train = mean_squared_error(y_train_pred, y_train)
    mean_squared_error_test = mean_squared_error(y_test_pred, y_test)

    print(f"Mean squared error on training set: {mean_squared_error_train:.4f}")
    print(f"Mean squared error on test set: {mean_squared_error_test:.4f}")


    






