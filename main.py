import numpy as np
inputs = [(0,0), (1,0), (0,1), (1,1)]
outputs = [0, 1, 1, 0]

inputs_outputs = {inputs[i]: outputs[i] for i in range(len(outputs))} # Create a dictionary. Keys: Inputs, Values: Correct Outputs


class Neuron: # This is for each neuron in the hidden layer. Can also be used for final output neuron too. 
    def __init__(self):
        self.weights = np.random.rand(2) # (Each neuron has 2 weights to connect to each input)
        self.bias = 0 


class Network: # Will be responsible for forward and backward pass.
    def __init__(self, l1_weights, l1_biases, l2_weights, l2_bias): # L1 and L2 just mean hidden layer 1 and hidden layer 2 
        self.l1_weights = l1_weights
        self.l1_biases = l1_biases
        self.l2_weights = l2_weights
        self.l2_bias = l2_bias

    def forward(self,inputs):
        l1_neurons = matrix_mult(self.l1_weights, inputs) + self.l1_biases # (Matrix of all weights * input vector) + bias vector
        l1_neurons = np.array(list(map(sigmoid, l1_neurons))) # Applies sigmoid on each of the neurons final outputs. 
        output = np.dot(self.l2_weights, l1_neurons) + self.l2_bias # Should be a scalar
        output = sigmoid(output)
        return output
    
    def backward(self,input): 
        pass



def matrix_mult(matrix, vector):
    new_vector = np.empty(0)
    for row in matrix: 
        value = 0
        for ind in range(len(row)):
            value += row[ind] * vector[ind]
        new_vector = np.append(new_vector, value)
    return new_vector

def sigmoid(scalar): 
    return 1/(1+np.e**(-scalar))

#----------------------------Main logic--------------------------------

hidden_1 = Neuron() # First neuron of hidden layer. Contains 2 weights and one bias
hidden_2 = Neuron() # Second neuron of hidden layer. Contains 2 weights and one bias
out = Neuron() # Output neuron. 1 connection between each of the 2 previous neuron (2 weights) and a final bias. 

all_hidden_weights = np.stack((hidden_1.weights, hidden_2.weights), axis=0) # 2x2 numpy matrix. 1st row has all weights for 1st neuron. 2nd row has all weights for 2nd neuron.  
all_hidden_biases = np.stack((hidden_1.bias, hidden_2.bias), axis=0) # Should be 1x2

net = Network(all_hidden_weights, all_hidden_biases, out.weights, out.bias)


print(net.forward([1,0]))










