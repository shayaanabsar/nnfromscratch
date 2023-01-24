from dataclasses import dataclass, field
from random import randint, random
from math import e

def reverse_enumerate(l):
	return list(enumerate(l))[::-1]

@dataclass
class LayerSpecification:
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

		predicted = sum(self.traverse(x))
		actual = sum(y)

		return (predicted - actual) ** 2

	def calculate_cost(self, x_data, y_data) -> int:
		# Calculate the sum of mse for all training data

		cost = 0
		for i, x in enumerate(x_data):
			cost += self.calculate_loss(x, y_data[i])
		return cost

	def derivative(self, x_data, y_data, h=0.00001, weight=None, bias=None) -> int:
		# Calculate the derivative

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

	def train(self, x_data, y_data, epochs, weight_learning_rate=0.001, bias_learning_rate=0.05):
		for _ in range(epochs):
			weight_derivatives, bias_derivatives = {}, {}
			
			for x, layer in reverse_enumerate(self.weights):
				for y, node in enumerate(layer):
					bias_derivatives[(x+1, y)] = -self.derivative(x_data, y_data, bias=(x+1, y))
					for z, connection in enumerate(node):
						weight_derivatives[(x, y, z)] = -self.derivative(x_data, y_data, weight=(x, y, z))

			for weight in weight_derivatives:
				x, y, z = weight
				self.weights[x][y][z] += (weight_learning_rate * weight_derivatives[(x, y, z)])

			for node in bias_derivatives:
				x, y = node
				self.layers[x][y].bias += (bias_learning_rate * bias_derivatives[(x, y)])

test_network = Network()

test_network.setup([
	LayerSpecification(1),
	LayerSpecification(1, 'relu')
])

gradient, y_intercept = randint(1, 10), randint(1, 10)
print(f'f(x) = {gradient}x + {y_intercept}')

x_train, y_train = [[i] for i in range(10)], [[i*gradient+y_intercept] for i in range(10)]
test_network.train(x_data=x_train, y_data=y_train, epochs=50)

print(f'n(x) = {test_network.weights[0][0][0]:.4f}x + {test_network.layers[1][0].bias:.4f}')