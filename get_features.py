import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
 
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

# add node (component) to the graph
for component in components:
    name = component['name']
    component_id = component['id']

    G.add_node(name)
    name_lookup[component_id] = name
    component_name_list.append(name)

# add connection to the graph
for connection in connections:
    source_name = name_lookup[connection['source']['component']]
    
    for sink in connection['sinks']:
        sink_name = name_lookup[sink['component']]

        G.add_edge(source_name, sink_name)


topological_list = list(nx.topological_sort(G))

print(topological_list)

pos = nx.spring_layout(G)
nx.draw_networkx(G, pos)
plt.savefig('networkx_graph.png')

f.close()