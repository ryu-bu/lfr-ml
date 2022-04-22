import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
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
 
f = open('train_files/correct_device.json',)
 
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

print(topological_list)

device_in = prepare_sequence(topological_list, device_to_ix)
position_in = prepare_sequence(topological_list, position_sorted_list)

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        outx = self.fc(h_out)
        outy = self.fc(h_out)

        return outx, outy

# training

sc = MinMaxScaler()
print(position_in[:, 0].shape)
x = sc.fit_transform(device_in.reshape(len(device_in), 1))
y = sc.fit_transform(position_in.reshape(len(position_in), 2))

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

num_epochs = 1000
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1

num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = lstm(device_in)
    optimizer.zero_grad()

    loss = criterion(outputs, position_in)

