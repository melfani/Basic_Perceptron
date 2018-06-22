import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as pyplot
%matplotlib inline

class Operation():
    def __init__(self,input_nodes=[]):
        self.input_nodes=input_nodes
        self.output_nodes=[]

        for node in input_nodes:
            node.output_nodes.append(self)

           # if hasattr(node, 'input_nodes'):
                #print(node.input_nodes)

        _default_graph.operations.append(self)

    def compute(self):
        pass

class add(Operation):
    def __init__(self, x,y):
        super().__init__([x,y])

    def compute(self, x, y):
        self.inputs=[x,y]
        return x+ y

class multiply(Operation):
    def __init__(self, x,y):
        super().__init__([x,y])

    def compute(self, x, y):
        self.inputs=[x,y]
        return x* y

class matmul(Operation):
    def __init__(self, x,y):
        super().__init__([x,y])

    def compute(self, x, y):
        self.inputs=[x,y]
        return x.dot(y)

class sigmmoid(operation):
    def __init__(self,z):
        super().__init__([z])

    def compute(self, z):
        return 1/ (1 + np.exp(-z))


class Placeholder():
    def __init__(self):
        self.output_nodes=[]

        _default_graph.placeholders.append(self)

class Variable():
    def __init__(self, initial_value=None):
        self.value=initial_value
        self.output_nodes=[]
        _default_graph.variables.append(self)

class Graph():

    def __init__(self):
        self.operations= []
        self.variables= []
        self.placeholders= []

    def set_as_default(self):
        global _default_graph
        _default_graph = self

def traverse_postorder(operation):
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    recurse(operation)
    return nodes_postorder

class Session():

    def run(self,operation, feed_dict={}):
        nodes_postorder=traverse_postorder(operation)

        for node in nodes_postorder:
            if type(node)== Placeholder:
                node.output=feed_dict[node]

            elif type(node) == Variable:
                node.output = node.value

            else:

                node.inputs= [ input_nodes.output for input_nodes in node.input_nodes]
                node.output= node.compute(*node.inputs) #args

            if type(node.output) == list:
                node.output=np.array(node.output)
        return operation.output




g = Graph()
g.set_as_default()

A= Variable(10)
b = Variable(1)
x = Placeholder()
y= multiply(A,x)
z = add(y,b)

sess=Session()
sess.run(operation=z, feed_dict={x:10})

## Classification


# Plot sigmoid function
sample_x = np.linspace(-10,10,100)
sample_y= sigmoid(sample_x)

plt.plot(sample_x, sample_y)


data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=42)
features= data[0]
label = data[1]
plt.scatter(features[:,0],features[:1], c=labels, cmap ='coolwarm')

x =np.linspace(0,11,10)
y = -x +5
plt.plot(x,y)
