import json
from os import readlink
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import var
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
from torch.nn.modules.activation import ReLU
import torch.optim as optim
from torch.autograd import Variable, backward
from sklearn.preprocessing import MinMaxScaler

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    print("Cuda avaialble")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("Cuda not avaliable")


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
 
f = open('correct_device.json',)
 
data = json.load(f)

# initialize directed graph
G = nx.DiGraph()

# features -> name, layers, id, port in components, and source, sink, component, entity in connection for topology.
components = data['components']
connections = data['connections'] 

component_name_list = []
layer_list = data['layers']

# initialize {name: id} dict
name_lookup = {}

# store positions
xtag_to_ix = {}
ytag_to_ix = {}
device_to_ix = {}

max_x = 0
max_y = 0

position_sorted_list = {}

# add node (component) to the graph
for component in components:
    name = component['name']
    component_id = component['id']
    params = component['params']

    G.add_node(name)
    name_lookup[component_id] = name
    component_name_list.append(name)
    x_position = params['position'][0]
    y_position = params['position'][1]
    xtag_to_ix[name] = x_position
    ytag_to_ix[name] = y_position
    position_sorted_list[name] = [x_position, y_position]
    
    max_x = x_position if x_position > max_x else max_x
    max_y = y_position if y_position > max_y else max_y

    if name not in device_to_ix:
        device_to_ix[name] = len(device_to_ix)

# add connection to the graph
for connection in connections:
    source_name = name_lookup[connection['source']['component']]
    
    for sink in connection['sinks']:
        sink_name = name_lookup[sink['component']]

        G.add_edge(source_name, sink_name)


topological_list = list(nx.topological_sort(G))

device_in = prepare_sequence(topological_list, device_to_ix)
position_in = prepare_sequence(topological_list, position_sorted_list)

keep_prob = 1

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=2, stride=1, padding=1), # padding is 0
            nn.ReLU(), # activation function, output = each value of output layer
            nn.MaxPool1d(kernel_size=2, stride=2), # take max val
            nn.Dropout(p=1 - keep_prob) # from 0 and 1, 
        )
        # self.layer2 = torch.nn.Sequential(
        #         torch.nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0),
        #         torch.nn.ReLU(),
        #         torch.nn.MaxPool1d(kernel_size=2, stride=2),
        #         torch.nn.Dropout(p=1 - keep_prob))

        # self.layer3 = torch.nn.Sequential(
        #     torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=0),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        #     torch.nn.Dropout(p=1 - keep_prob))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(1 * 32, 625, bias=True)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 5 outputs
        self.fc2 = torch.nn.Linear(625, 5, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters additonla step

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        # out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out

xmodel = CNN()
ymodel = CNN()

learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.
loss_function = nn.MSELoss()
xoptimizer = torch.optim.Adam(params=xmodel.parameters(), lr=learning_rate)
yoptimizer = torch.optim.Adam(params=ymodel.parameters(), lr=learning_rate)

training_epochs = 2000

data_loader = torch.utils.data.DataLoader(dataset=device_in,
                                          batch_size=32,
                                          shuffle=True)

xtargets = prepare_sequence(topological_list, xtag_to_ix)
ytargets = prepare_sequence(topological_list, ytag_to_ix)

for epoch in range(training_epochs):
    inputs = Variable(device_in.type(torch.FloatTensor))
    xlabels = Variable(xtargets.type(torch.FloatTensor))
    ylabels  = Variable(ytargets.type(torch.FloatTensor))


    in_width = int(inputs.shape[0])
    x_width = int(xlabels.shape[0])

    input_1d = inputs.reshape([1, in_width, 1])
    x_1d = xlabels.reshape([1, in_width, 1])

    # for i, batch_in in enumerate(data_loader):

    xoptimizer.zero_grad() # <= initialization of the gradients
    yoptimizer.zero_grad()
    
    # forward propagation
    xhypothesis = xmodel(input_1d)
    yhyopthesis = ymodel(input_1d)

    xcost = loss_function(xhypothesis, xlabels) # <= compute the loss function
    ycost = loss_function(yhyopthesis, ylabels)
    
    # Backward propagation
    xcost.backward() # <= compute the gradient of the loss/cost function    
    ycost.backward()

    xoptimizer.step() # <= Update the gradients
    yoptimizer.step()


xmodel.eval()
ymodel.eval()

xprediction = xmodel(input_1d)
yprediction = ymodel(input_1d)

print("\n---- x prediciton ----")
print(xprediction.data)
print(xlabels)

print("\n---- y prediciton ----")
print(yprediction.data)
print(ylabels)