import numpy as np


inputs = [(0, 0), (1, 0), (0, 1), (1, 1)] # All possible pairs of inputs
outputs = [0, 1, 1, 0] # Corresponding correct outputs
inputs_outputs = {inputs[i]: outputs[i] for i in range(len(outputs))} # Create a dictionary


'''
class Neuron: 
    def __init__():
        self.weights
'''



def matrix_mult(matrix, vector):
    new_vector = np.empty(2)
    for row in matrix: 
        value = 0
        for ind in range(len(row)):
            value += row[ind] * vector[ind]
        new_vector.append(value)
    return new_vector

mat = np.array([[3, 4], [5, 6], [9,10]])
vect = np.array([1, 2])
a = matrix_mult(mat, vect)
print(a)



