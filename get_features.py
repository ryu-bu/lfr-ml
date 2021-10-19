import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable


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

# lstm start
EMBEDDING_DIM = 1
HIDDEN_DIM = 5

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        # embeds = self.word_embeddings(sentence)
        # lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1)) # change this for output size
        tag_space = self.hidden2tag(sentence)
        # tag_scores = F.log_softmax(tag_space, dim=1) # TODO: change this for vector output
        return tag_space

model_x = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(device_to_ix), len(topological_list))
model_y = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(device_to_ix), len(topological_list))
loss_function = nn.MSELoss() #TODO: change this for linear regression, use mean square error etc...
optimizer_x = optim.SGD(model_x.parameters(), lr=0.1)
optimizer_y = optim.SGD(model_y.parameters(), lr=0.1)

# with torch.no_grad():
#     inputs = prepare_sequence(topological_list, device_to_ix)
#     tag_scores = model_x(inputs)
#     print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    model_x.zero_grad()
    device_in = prepare_sequence(topological_list, device_to_ix)
    xtargets = prepare_sequence(topological_list, xtag_to_ix)
    ytargets = prepare_sequence(topological_list, ytag_to_ix)

    device_in = np.array(device_in, dtype=np.float32)
    xtargets = np.array(xtargets, dtype=np.float32)
    ytargets = np.array(ytargets, dtype=np.float32)

    print(ytargets)

    inputs = Variable(torch.from_numpy(device_in))
    xlabels = Variable(torch.from_numpy(xtargets))
    ylabels = Variable(torch.from_numpy(ytargets))

    xtag_scores = model_x(inputs)
    ytag_scores = model_y(inputs)
    xloss = loss_function(xtag_scores, xlabels)
    yloss = loss_function(ytag_scores, ylabels)
    xloss.backward()
    yloss.backward()
    optimizer_x.step()
    optimizer_y.step()

    ## TODO: check deepdrawing paper to see normalization method

    # for sentence, tags in training_data:
    #     # Step 1. Remember that Pytorch accumulates gradients.
    #     # We need to clear them out before each instance
    #     model.zero_grad()

    #     # Step 2. Get our inputs ready for the network, that is, turn them into
    #     # Tensors of word indices.
    #     sentence_in = prepare_sequence(sentence, word_to_ix)
    #     targets = prepare_sequence(tags, tag_to_ix)

    #     # Step 3. Run our forward pass.
    #     tag_scores = model(sentence_in)

    #     # Step 4. Compute the loss, gradients, and update the parameters by
    #     #  calling optimizer.step()
    #     loss = loss_function(tag_scores, targets)
    #     loss.backward()
    #     optimizer.step()

with torch.no_grad():
    device_in = prepare_sequence(topological_list, device_to_ix)
    device_in = np.array(inputs, dtype=np.float32)
    inputs = Variable(torch.from_numpy(device_in))
    xtag_scores = model_x(inputs)
    ytag_scores = model_y(inputs)

    print(xtag_scores)
    print("correct: ", xlabels)
    print(ytag_scores) ## always off by 21.9%???
    print("correct: ", ylabels)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    # count = 0
    # for row in xtag_scores:
    #     max_valx = row.topk(1)[1]
    #     print("x: ", max_valx)
    #     max_valy = ytag_scores[count].topk(1)[1]
    #     print("y: ", max_valy)
        
    #     count += 1

print(topological_list)

pos = nx.spring_layout(G)
nx.draw_networkx(G, pos)
plt.savefig('networkx_graph.png')

f.close()