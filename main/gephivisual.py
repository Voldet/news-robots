# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 06:52:48 2018

@author: Administrator
"""

from gephistreamer import graph
from gephistreamer import streamer

# Create a Streamer
# adapt if needed : streamer.GephiWS(hostname="localhost", port=8080, workspace="workspace0")
# You can also use REST call with GephiREST (a little bit slower than Websocket)
GWS = streamer.GephiWS(hostname="localhost", port=8080, workspace="workspace1")
stream = streamer.Streamer(GWS)

# Create a node with a custom_property
node_a = graph.Node("A", custom_property=1)

# Create a node and then add the custom_property
node_b = graph.Node("B")
node_b.property['custom_property'] = 2

# Add the node to the stream
# you can also do it one by one or via a list
l = [node_a, node_b]
stream.add_node(*l)
node_c = graph.Node("C")
stream.add_node(node_a, node_b, node_c)

# Create edge
# You can also use the id of the node :  graph.Edge("A","B",custom_property="hello")
edge_ab = graph.Edge(node_a, node_b, custom_property="hello")
stream.add_edge(edge_ab)
