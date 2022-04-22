import json
import glob, os
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.similarity import optimize_edit_paths
import numpy as np
from numpy.core.fromnumeric import var
from sklearn import preprocessing
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
from torch.nn.modules.activation import ReLU
import torch.optim as optim
from torch.autograd import Variable, backward
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from lstm_model import LSTMTagger

DEVICE_COUNT = 5
EMBEDDING_DIM = 1
HIDDEN_DIM = 5

training_epochs = 1100

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
 
# f = open('train_files/correct_device.json',)
 
# data = json.load(f)

component_name_list = []
device_to_ix = {}

def deg_classifer(data):
    classified_list = []

    for deg in data:
        min_deg = 360
        deg_item = 0
        for i in [0, 90, 180, 270]:
            temp = abs(i - deg)
            if temp < min_deg:
                min_deg = temp
                deg_item = i

        classified_list.append(deg_item)

    return classified_list    

def analyze_single_file(data, component_name_list, device_to_ix):

    # initialize directed graph
    G = nx.DiGraph()

    # features -> name, layers, id, port in components, and source, sink, component, entity in connection for topology.
    components = data['components']
    connections = data['connections'] 

    layer_list = data['layers']

    name_lookup = {}

    # store positions
    xtag_to_ix = {}
    ytag_to_ix = {}
    rtag_to_ix = {}

    max_x = 0
    max_y = 0

    # add node (component) to the graph
    for component in components:
        name = component['name']
        component_id = component['id']
        params = component['params']

        G.add_node(name)
        name_lookup[component_id] = name
        if name not in component_name_list:
            component_name_list.append(name)
        x_position = params['position'][0]
        y_position = params['position'][1]
        rotation = params['rotation'] if "rotation" in params else 0
        xtag_to_ix[name] = x_position
        ytag_to_ix[name] = y_position
        rtag_to_ix[name] = rotation
        
        max_x = x_position if x_position > max_x else max_x
        max_y = y_position if y_position > max_y else max_y

        if name not in device_to_ix:
            device_to_ix[name] = len(device_to_ix)
        
    # add connection to the graph
    for connection in connections:
        source_name = name_lookup[connection['source']['component']]
        
        if connection['sinks']:
            for sink in connection['sinks']:
                sink_name = name_lookup[sink['component']]

                G.add_edge(source_name, sink_name)


    topological_list = list(nx.topological_sort(G))

    device_in = prepare_sequence(topological_list, device_to_ix)

    return topological_list, device_in, xtag_to_ix, ytag_to_ix, rtag_to_ix, component_name_list, device_to_ix

os.chdir("train_files")
files = glob.glob("*.json")
f = []
input_1d_list = []
xtargets_list = []
ytargets_list = []
rtargets_list = []

for file in files:
    print(file)
    f = open(file,)
    # print(file)
    data = json.load(f)
    
    topological_list, device_in, xtag_to_ix, ytag_to_ix, rtag_to_ix, component_name_list, device_to_ix = analyze_single_file(data, component_name_list, device_to_ix)


    data_loader = torch.utils.data.DataLoader(dataset=device_in,
                                            batch_size=32,
                                            shuffle=True)

    xtargets = prepare_sequence(topological_list, xtag_to_ix)
    ytargets = prepare_sequence(topological_list, ytag_to_ix)
    rtargets = prepare_sequence(topological_list, rtag_to_ix)

    in_width = int(device_in.shape[0])
    x_width = int(xtargets.shape[0])

    input_1d = device_in.reshape([1, 1, in_width])

    xtargets_list.append([xtargets, input_1d, file])
    ytargets_list.append([ytargets, rtargets])

xtargets_list = np.asarray(xtargets_list)
ytargets_list = np.asarray(ytargets_list)

# X_train, X_test, Y_train, Y_test = train_test_split(xtargets_list, ytargets_list, test_size=0.33, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(xtargets_list, ytargets_list, test_size=0.1, shuffle=False)


print(X_test)

model_x = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, DEVICE_COUNT, DEVICE_COUNT)
model_y = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, DEVICE_COUNT, DEVICE_COUNT)
model_r = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, DEVICE_COUNT, DEVICE_COUNT)
loss_function = nn.MSELoss() #TODO: change this for linear regression, use mean square error etc...
optimizer_x = optim.SGD(model_x.parameters(), lr=0.1)
optimizer_y = optim.SGD(model_y.parameters(), lr=0.1)
optimizer_r = optim.SGD(model_r.parameters(), lr=0.1)

for epoch in range(training_epochs):
    for i, X in enumerate(X_train, 0):

        x = X[0]
        device_num = X[1]

        y = Y_train[i, 0]
        r = Y_train[i, 1]

        # for i, batch_in in enumerate(data_loader):
        optimizer_x.zero_grad()
        optimizer_y.zero_grad()
        optimizer_r.zero_grad()
        
        # forward propagation
        # xhypothesis, yhypothesis, rhypothesis = model(device_num)
        xhypothesis = model_x(device_num)
        yhypothesis = model_y(device_num)
        rhypothesis = model_r(device_num)

        xcost = loss_function(xhypothesis, x) # <= compute the loss function
        ycost = loss_function(yhypothesis, y)
        rcost = loss_function(rhypothesis, r)
        
        # Backward propagation
        xcost.backward(retain_graph=True) # <= compute the gradient of the loss/cost function    
        ycost.backward(retain_graph=True)
        rcost.backward(retain_graph=True)

        optimizer_x.step()
        optimizer_y.step()
        optimizer_r.step()

for i, X in enumerate(X_test, 0):
    xprediction = model_x(X[1])
    yprediction = model_y(X[1])
    rprediction = model_r(X[1])

    print("\n\ncount: ", i)
    print("file name: ", X[2])

    print("\n---- x prediciton ----")
    print("prediction: ", xprediction.data)
    print(X[0])

    # ce = calculate_error(xprediction.data, xtargets)

    # print("error: ", ce)

    print("\n---- y prediciton ----")
    print("prediction: ", yprediction.data)
    print(Y_test[i][0])

    # make classifier for rotation: 0, 90, 180, 270
    print("\n---- r prediciton ----")
    # print(rprediction.data)
    print("prediction: ", deg_classifer(rprediction.data.tolist()[0][0]))
    print(Y_test[i][1])