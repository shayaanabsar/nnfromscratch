from dataclasses import dataclass, field
from random import randint
from math import e

@dataclass
class LayerSpecification:
	num_nodes: int
	activation: str

@dataclass
class Node:
	activation: str
	value: float = 0.0
	bias: int = randint(0, 5)

	def apply_activation(self):
		match self.activation:
			case 'sigmoid': self.value = (1 / (1 + e ** -(self.value + self.bias)))
			case 'relu'   : self.value = max(0, self.value + self.bias)
			
	
@dataclass
class Network:
	layers : list  = field(default_factory=list)
	weights: list = field(default_factory=list)

	def setup(self, specifications):
		final_spec = len(specifications) - 1

		for i, spec in enumerate(specifications):
			self.layers.append([Node(activation=spec.activation) for _ in range(spec.num_nodes)])
	
			if i != final_spec:
				self.weights.append([])
				for _ in range(spec.num_nodes):
					self.weights[-1].append([randint(0, 5) for _ in range(specifications[i+1].num_nodes)])

	def reset_network(self):
		for i in range(len(self.layers)):
			for j in range(len(self.layers[i])):
				self.layers[i][j].value = 0

	def traverse(self, input) -> list:
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
		predicted = sum(self.traverse(x))
		actual = sum(y)

		return (predicted - actual) ** 2

	def calculate_cost(self, x_data, y_data) -> int:
		cost = 0
		for i, x in enumerate(x_data):
			cost += self.calculate_loss(x, y_data[i])
		return cost

	def derivative(x_data, y_data, self, h=0.00001, weight=None, bias=None) -> int:
		a = self.calculate_cost(x_data, y_data)

		layer, node_number, connection = weight
		self.weights[layer][node_number][connection] += h

		b = self.calculate_cost(x_data, y_data)

		self.weights[layer][node_number][connection] -= h

		return (a + b) / h

	def train(x_data, y_data, epochs, learning_rate=0.0001):
		pass


nn = Network()
nn.setup([LayerSpecification(3, None), 
		LayerSpecification(2, 'sigmoid'), 
		LayerSpecification(1, 'relu')])
