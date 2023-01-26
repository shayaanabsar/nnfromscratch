from dataclasses import dataclass, field
from random import randint, random
from math import e

def reverse_enumerate(l):
	return list(enumerate(l))[::-1]

@dataclass
class Layer:
	num_nodes: int
	activation: str = None

@dataclass
class Node:
	activation: str
	value: float = 0.0
	bias: float = randint(1, 5)

	def apply_activation(self):
		match self.activation:
			case 'sigmoid': self.value = (1 / (1 + e ** -(self.value + self.bias)))
			case 'relu'   : self.value = max(0, self.value + self.bias)
			
	
@dataclass
class Network:
	layers : list  = field(default_factory=list)
	weights: list = field(default_factory=list)

	x_data: list = None
	y_data: list = None
	epochs: int = None
	learning_rate: int = 0.02

	def setup(self, specifications):
		# Setup the neural network

		final_spec = len(specifications) - 1

		for i, spec in enumerate(specifications):
			self.layers.append([Node(activation=spec.activation) for _ in range(spec.num_nodes)])
	
			if i != final_spec:
				self.weights.append([])
				for _ in range(spec.num_nodes):
					self.weights[-1].append([randint(1,5) for _ in range(specifications[i+1].num_nodes)])

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

		predicted, actual = self.traverse(x), y
		loss = 0

		for i, val in enumerate(predicted):
			loss += (val - actual[i]) ** 2

		return loss

	def calculate_cost(self) -> int:
		# Calculate the sum of mse for all training data

		cost = 0
		
		for i, x in enumerate(self.x_data):
			cost += self.calculate_loss(x, self.y_data[i])
		return cost / len(self.x_data)

	def gradient(self, h, weight=None, bias=None) -> int:
		# Calculate the derivative

		a = self.calculate_cost()

		if bias is None:
			layer, node_number, connection = weight
			self.weights[layer][node_number][connection] += h
		else:
			layer, node_number = bias
			self.layers[layer][node_number].bias += h
		b = self.calculate_cost()

		if bias is None:
			self.weights[layer][node_number][connection] -= h
		else:
			self.layers[layer][node_number].bias -= h

		return (b - a) / h

	def derivative(self, weight=None, bias=None):
		h, previous, current = 1, None, None

		while True:
			h, previous, current = h/10, current, self.gradient(weight=weight, bias=bias, h=h)

			if ((None not in (previous, current)) and (round(previous, 2) == round(current, 2))) or (h <= 10e-10):
				return current

	def train(self):
		if None in (self.x_data, self.y_data, self.epochs):
			raise Exception('x_data, y_data and epochs must be provided to train.')

		for _ in range(self.epochs):
			weight_derivatives, bias_derivatives = {}, {}
			
			for x, layer in reverse_enumerate(self.weights):
				for y, node in enumerate(layer):
					bias_derivatives[(x+1, y)] = -self.derivative(bias=(x+1, y))
					for z, connection in enumerate(node):
						weight_derivatives[(x, y, z)] = -self.derivative(weight=(x, y, z))

			for weight in weight_derivatives:
				x, y, z = weight
				self.weights[x][y][z] += (self.learning_rate * weight_derivatives[(x, y, z)])

			for node in bias_derivatives:
				x, y = node
				self.layers[x][y].bias += (self.learning_rate * bias_derivatives[(x, y)])

test_network = Network()

test_network.setup([
	Layer(1),
	Layer(1, 'relu')
])

gradient, y_intercept = randint(1, 10), randint(1, 10)
print(f'f(x) = {gradient}x + {y_intercept}')

x_train, y_train = [[i] for i in range(10)], [[i*gradient+y_intercept] for i in range(10)]

test_network.x_data, test_network.y_data, test_network.epochs = x_train, y_train, 500

test_network.train()

print(f'n(x) = {test_network.weights[0][0][0]:.4f}x + {test_network.layers[1][0].bias:.4f}')