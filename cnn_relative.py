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

DEVICE_COUNT = 5
X_BASE = 45000
Y_BASE = 30000
MAG = 2300

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

    # component_name_list = x[]
    layer_list = data['layers']

    # initialize {name: id} dict
    name_lookup = {}

    # store positions
    xtag_to_ix = {}
    ytag_to_ix = {}
    rtag_to_ix = {}
    # device_to_ix = {}

    max_x = 0
    max_y = 0

    # position_sorted_list = {}

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
        # position_sorted_list[name] = [x_position, y_position]
        
        max_x = x_position if x_position > max_x else max_x
        max_y = y_position if y_position > max_y else max_y

        if name not in device_to_ix:
            device_to_ix[name] = len(device_to_ix)
        
        # print(device_to_ix)

    # add connection to the graph
    for connection in connections:
        source_name = name_lookup[connection['source']['component']]
        
        if connection['sinks']:
            for sink in connection['sinks']:
                sink_name = name_lookup[sink['component']]

                G.add_edge(source_name, sink_name)


    topological_list = list(nx.topological_sort(G))

    device_in = prepare_sequence(topological_list, device_to_ix)
    # position_in = prepare_sequence(topological_list, position_sorted_list)

    return topological_list, device_in, xtag_to_ix, ytag_to_ix, rtag_to_ix, component_name_list, device_to_ix

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

        # L4 FC 128 inputs -> 625 outputs
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
        # L5 Final FC 625 inputs -> 4 outputs
        self.fc2x = torch.nn.Linear(625, 4, bias=True)
        self.fc2y = torch.nn.Linear(625, 4, bias=True)
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
        # make it 1x625 array
        # xout = xout.view(5, -1)
        xout = self.fc2x(xout)

        yout = self.layer1(x)
        yout = self.layer2(yout)
        yout = self.layer3(yout)
        yout = yout.view(yout.size(0), -1)
        yout = self.fc1y(yout)
        # yout = yout.view(5, -1)
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

# load data
os.chdir("train_files")
files = glob.glob("*.json")
f = []
input_1d_list = []
xtargets_list = []
ytargets_list = []
rtargets_list = []

for file in files:
    f = open(file,)
    print(file)
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
    # put ytargets and rtargets together
    ytargets_list.append([ytargets, rtargets])
    # rtargets_list.append(rtargets)
    # input_1d_list.append(input_1d)

    f.close()
# x_1d = xtargets.reshape([1, 1, in_width])

# print(xtargets_list)

xtargets_list = np.asarray(xtargets_list)
ytargets_list = np.asarray(ytargets_list)

# X_train, X_test, Y_train, Y_test = train_test_split(xtargets_list, ytargets_list, test_size=0.33, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(xtargets_list, ytargets_list, test_size=0.1, shuffle=False)


# print(component_name_list)

def compute_distance(x):
    x = np.array(x)
    #initialize array
    diff_mat = np.zeros((5, 4))
    
    #calculate diff
    for i, val in enumerate(x, 0):
        diff_array = x - val

        # erase base element from the arary
        diff_array = np.delete(diff_array, i)

        diff_mat[i] = diff_array

    return diff_mat

def compute_relative_distance(x_array, y_array):
    x_array = np.array(x_array)
    y_array = np.array(y_array)

    # calc distance
    x_diff = x_array - x_array[0]
    y_diff = y_array - y_array[0]

    # delete first element because its always 0
    x_diff = np.delete(x_diff, 0)
    y_diff = np.delete(y_diff, 0)

    base_val = float('inf')

    # find base val
    for x, y in zip(x_diff, y_diff):
        base_val = abs(x) if abs(x) < base_val and x != 0 else base_val
        base_val = abs(y) if abs(y) < base_val and y != 0 else base_val

    x_diff /= base_val
    y_diff /= base_val

    return torch.tensor(x_diff), torch.tensor(y_diff)


# reverse compute_distance
def put_back(x, y):
    x = np.array(x)
    y = np.array(y)

    x_dist = x * MAG;
    y_dist = y * MAG;

    x_dist = np.insert(x_dist, 0, 0)
    y_dist = np.insert(y_dist, 0, 0)

    x_dist += X_BASE;
    y_dist += Y_BASE;

    return x_dist, y_dist



# # test a file
# ftest = open("tdroplet9.json")
# test_data = json.load(ftest)

# topological_list, device_in_test, xtag_to_ix, ytag_to_ix, rtag_to_ix, component_name_list, device_to_ix = analyze_single_file(test_data, component_name_list, device_to_ix)
# input_test = device_in_test.reshape([1, 1, in_width])

# xtargets = prepare_sequence(topological_list, xtag_to_ix)
# ytargets = prepare_sequence(topological_list, ytag_to_ix)
# rtargets = prepare_sequence(topological_list, rtag_to_ix)

# for epoch in range(training_epochs):
#     optimizer.zero_grad()

#     xprediction, yprediction, rprediction = model(input_test)


#     # x = compute_distance(xtargets)
#     # y = compute_distance(ytargets)

#     x, y = compute_relative_distance(xtargets, ytargets)

#     # x = torch.tensor(x, dtype=torch.float)
#     # y = torch.tensor(y, dtype=torch.float)

#     xcost = loss_function(xprediction, x) # <= compute the loss function
#     ycost = loss_function(yprediction, y)
#     rcost = loss_function(rprediction, rtargets)

#     cost = xcost + ycost + rcost
    
#     # Backward propagation
#     cost.backward(retain_graph=True) # <= compute the gradient of the loss/cost function    

#     optimizer.step()

# model.eval()

# xprediction, yprediction, rprediction = model(input_test)
# xpred_abs, ypred_abs = put_back(xprediction.data, yprediction.data)

# x_actual, y_actual = compute_relative_distance(xtargets, ytargets)

# print("\n---- x prediciton ----")
# print("prediction ratio: ", xprediction)
# print("prediction abs: ", xpred_abs)

# print("actual ratio: ", x_actual)
# print("actual abs: ", xtargets)

# ce = calculate_error(xprediction.data, xtargets)

# print("error: ", ce)

# print("\n---- y prediciton ----")
# print("prediction: ", yprediction)
# print("prediction abs: ", ypred_abs)

# print("actual ratio: ", y_actual)
# print("actual abs: ", ytargets)

# # make classifier for rotation: 0, 90, 180, 270
# print("\n---- r prediciton ----")
# # print(rprediction.data)
# print("prediction: ", deg_classifer(rprediction.data.tolist()[0]))
# print(rtargets)


for epoch in range(training_epochs):
    for i, X in enumerate(X_train, 0):

        x = X[0]
        device_num = X[1]

        y = Y_train[i, 0]
        r = Y_train[i, 1]

        # for i, batch_in in enumerate(data_loader):
        optimizer.zero_grad()
        
        # forward propagation
        xhypothesis, yhypothesis, rhypothesis = model(device_num)

        x, y = compute_relative_distance(x, y)

        xcost = loss_function(xhypothesis, x) # <= compute the loss function
        ycost = loss_function(yhypothesis, y)
        rcost = loss_function(rhypothesis, r)

        cost = xcost + ycost + rcost
        
        # Backward propagation
        cost.backward(retain_graph=True) # <= compute the gradient of the loss/cost function    

        optimizer.step()

model.eval()

def calculate_error(predictions, targets):
    predictions = predictions.numpy()[0]
    targets = targets.numpy()

    diff = 0

    for i in range(DEVICE_COUNT):
        pred_rel = predictions - predictions[i]
        target_rel = targets - targets[i]

        diff += abs(pred_rel - target_rel)

        # print("error diff: ", diff)

    return diff.sum() / DEVICE_COUNT

for i, X in enumerate(X_test, 0):
    xprediction, yprediction, rprediction = model(X[1])

    x_actual, y_actual = compute_relative_distance(X[0], Y_test[i][0])

    xpred_abs, ypred_abs = put_back(xprediction.data, yprediction.data)

    print("\n\ncount: ", i)
    print("file name: ", X[2])

    print("\n---- x prediciton ----")
    print("prediction ratio: ", xprediction)
    print("prediction abs: ", xpred_abs)

    print("actual ratio: ", x_actual)
    print("actual abs: ", X[0])

    # ce = calculate_error(xprediction.data, xtargets)

    # print("error: ", ce)

    print("\n---- y prediciton ----")
    print("prediction: ", yprediction)
    print("prediction abs: ", ypred_abs)

    print("actual ratio: ", y_actual)
    print("actual abs: ", Y_test[i][0])

    # make classifier for rotation: 0, 90, 180, 270
    print("\n---- r prediciton ----")
    # print(rprediction.data)
    print("prediction: ", deg_classifer(rprediction.data.tolist()[0]))
    print(Y_test[i][1])

# # make classifier for rotation: 0, 90, 180, 270
# print("\n---- r prediciton ----")
# # print(rprediction.data)
# print("prediction: ", deg_classifer(rprediction.data.tolist()[0]))
# print(rtargets)