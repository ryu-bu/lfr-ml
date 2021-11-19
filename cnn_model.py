import json
from os import readlink
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.similarity import optimize_edit_paths
import numpy as np
from numpy.core.fromnumeric import var
from sklearn import preprocessing
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
    return torch.tensor(idxs, dtype=torch.float)
 
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
rtag_to_ix = {}
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
    rotation = params['rotation'] if "rotation" in params else 0
    xtag_to_ix[name] = x_position
    ytag_to_ix[name] = y_position
    rtag_to_ix[name] = rotation
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
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1), # padding is 0
            nn.ReLU(), # activation function, output = each value of output layer
            nn.MaxPool1d(kernel_size=2, stride=2), # take max val
            nn.Dropout(p=1 - keep_prob) # from 0 and 1, 
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1x = torch.nn.Linear(1 * 4 * 32, 625, bias=True)
        torch.nn.init.xavier_uniform(self.fc1x.weight)
        self.fc1y = torch.nn.Linear(1 * 4 * 32, 625, bias=True)
        torch.nn.init.xavier_uniform(self.fc1y.weight)
        self.fc1r = torch.nn.Linear(1 * 4 * 32, 625, bias=True)
        torch.nn.init.xavier_uniform(self.fc1r.weight)
        self.layer4x = torch.nn.Sequential(
            self.fc1x,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))

        self.layer4y = torch.nn.Sequential(
            self.fc1y,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        
        self.layer4r = torch.nn.Sequential(
            self.fc1r,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 5 outputs
        self.fc2x = torch.nn.Linear(625, 5, bias=True)
        self.fc2y = torch.nn.Linear(625, 5, bias=True)
        self.fc2r = torch.nn.Linear(625, 5, bias=True)

        torch.nn.init.xavier_uniform_(self.fc2x.weight) # initialize parameters additonla step
        torch.nn.init.xavier_uniform_(self.fc2y.weight) # initialize parameters additonla step
        torch.nn.init.xavier_uniform_(self.fc2r.weight) # initialize parameters additonla step

    def forward(self, x):
        xout = self.layer1(x)
        xout = self.layer2(xout)
        xout = self.layer3(xout)
        xout = xout.view(xout.size(0), -1)   # Flatten them for FC
        xout = self.fc1x(xout)
        xout = self.fc2x(xout)


        yout = self.layer1(x)
        yout = self.layer2(yout)
        yout = self.layer3(yout)
        yout = yout.view(yout.size(0), -1)
        yout = self.fc1y(yout)
        yout = self.fc2y(yout)

        rout = self.layer1(x)
        rout = self.layer2(rout)
        rout = self.layer3(rout)
        rout = rout.view(rout.size(0), -1)
        rout = self.fc1r(rout)
        rout = self.fc2r(rout)

        return xout, yout, rout

model = CNN()

learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()    # Softmax is internally computed. for classification
loss_function = nn.MSELoss() #TODO: try more loss functions
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

training_epochs = 1100

data_loader = torch.utils.data.DataLoader(dataset=device_in,
                                          batch_size=32,
                                          shuffle=True)

xtargets = prepare_sequence(topological_list, xtag_to_ix)
ytargets = prepare_sequence(topological_list, ytag_to_ix)
rtargets = prepare_sequence(topological_list, rtag_to_ix)

in_width = int(device_in.shape[0])
x_width = int(xtargets.shape[0])

input_1d = device_in.reshape([1, 1, in_width])
# x_1d = xtargets.reshape([1, 1, in_width])

for epoch in range(training_epochs):
    

    # for i, batch_in in enumerate(data_loader):
    optimizer.zero_grad()
    
    # forward propagation
    xhypothesis, yhypothesis, rhypothesis = model(input_1d)

    xcost = loss_function(xhypothesis, xtargets) # <= compute the loss function
    ycost = loss_function(yhypothesis, ytargets)
    rcost = loss_function(rhypothesis, rtargets)

    cost = xcost + ycost + rcost
    
    # Backward propagation
    cost.backward(retain_graph=True) # <= compute the gradient of the loss/cost function    

    optimizer.step()

model.eval()

xprediction, yprediction, rprediction = model(input_1d)

print("\n---- x prediciton ----")
print(xprediction.data)
print(xtargets)

print("\n---- y prediciton ----")
print(yprediction.data)
print(ytargets)

print("\n---- r prediciton ----")
print(rprediction.data)
print(rtargets)