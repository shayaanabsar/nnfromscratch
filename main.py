from dataclasses import dataclass, field
from random import randint
from math import e
from time import sleep

@dataclass
class LayerSpecification:
	num_nodes: int
	activation: str

@dataclass
class Node:
	activation: str
	value: float = 0.0
	bias: int = 0

	def apply_activation(self):
		match self.activation:
			case 'sigmoid': self.value = (1 / (1 + e ** -(self.value + self.bias)))
			case 'relu'   : self.value = max(0, self.value + self.bias)
			
	
@dataclass
class Network:
	layers : list  = field(default_factory=list)
	weights: list = field(default_factory=list)

	def setup(self, specifications):
		# Setup the neural network

		final_spec = len(specifications) - 1

		for i, spec in enumerate(specifications):
			self.layers.append([Node(activation=spec.activation) for _ in range(spec.num_nodes)])
	
			if i != final_spec:
				self.weights.append([])
				for _ in range(spec.num_nodes):
					self.weights[-1].append([randint(0, 5) for _ in range(specifications[i+1].num_nodes)])

	def reset_network(self):
		# Reset the node values

		for i in range(len(self.layers)):
			for j in range(len(self.layers[i])):
				self.layers[i][j].value = 0

	def traverse(self, input) -> list:
		# Pass the input through the network and return the output

		self.reset_network()

		if len(input) != len(self.layers[0]):
			raise Exception('Shape of the Input provided and of the Input Layer do not match.')

		for i, val in enumerate(input):
			self.layers[0][i].value = val
		
		for i, layer in enumerate(self.layers[:-1]):
			for j, node in enumerate(layer):
				
				for a, weight in enumerate(self.weights[i][j]):
					self.layers[i+1][a].value += (node.value * weight)

			for a, node in enumerate(self.layers[i+1]):
				self.layers[i+1][a].apply_activation()

		return [node.value for node in self.layers[-1]]

	def calculate_loss(self, x, y) -> int:
		# Calculate the means squared error for a given pair of data

		predicted = sum(self.traverse([x]))
		actual = sum([y])

		return (predicted - actual) ** 2

	def calculate_cost(self, x_data, y_data) -> int:
		# Calculate the sum of mse for all training data

		cost = 0
		for i, x in enumerate(x_data):
			cost += self.calculate_loss(x, y_data[i])
		return cost

	def derivative(self, x_data, y_data, h=0.000001, weight=None, bias=None) -> int:
		# Calculate the partial derivative of a given weight or bias with respect to the cost

		a = self.calculate_cost(x_data, y_data)
		
		if bias is None:
			layer, node_number, connection = weight
			self.weights[layer][node_number][connection] += h
		else:
			layer, node_number = bias
			self.layers[layer][node_number].bias += h

		b = self.calculate_cost(x_data, y_data)

		if bias is None:
			self.weights[layer][node_number][connection] -= h
		else:
			self.layers[layer][node_number].bias -= h

		return (b - a) / h

	def train(self, x_data, y_data, epochs, learning_rate=0.01):
		for _ in range(epochs):
			for x, layer in reversed(list(enumerate(self.weights[::-1]))):
				for y, node in enumerate(layer):
					for z, connection in enumerate(node):
						negative_derivative = -self.derivative(x_data, y_data, weight=(x, y, z))
						self.weights[x][y][z] += (learning_rate * negative_derivative)
			
#nn = Network()
#nn.setup([LayerSpecification(1, None), LayerSpecification(1, 'relu')])
#nn.train([2, 5], [10, 25], 10)
#print(nn)
#print(nn.traverse([9]))
