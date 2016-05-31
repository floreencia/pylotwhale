# -*- coding: utf-8 -*-
"""
network tools
Created on Fri May 27 22:12:01 2016


@author: florencia
"""

from __future__ import print_function
import networkx as nx
import pygraphviz as pgv
import matplotlib.pyplot as plt


#### graphviz
def nxGraph2pgv(G):
    return nx.to_agraph(G)    
    
def drawGraphviz(A, dotFile=None, figName=None):
    '''draws graphviz graph (A)'''
    if dotFile: A.write(dotFile)  # write previously positioned graph to PNG file
    A.layout(prog='dot')
    if figName: A.draw(figName)
    

def conceptualiseNodes(graphO, nodeLi=None):
    '''invisibilises nodes in nodeLi kkeping the edge label if any,
    useful for removing _ini and _end    
    WARNING: self linking nodes can leand to undesirable outcomes
    Parameters:
    ----------
        graphO : pygraphviz object (see nxGraph2pgv )
        nodeLi : list of nodes to remove    
    '''
    if nodeLi is None: nodeLi = []
    for nn in nodeLi:
        for edge in graphO.edges():
            if nn in edge:
                ed = graphO.get_edge(*edge)
                edLabel = ed.attr['label']
                graphO.delete_edge(ed)
                ## new node
                n1,n2 = ed
                if n1 == nn: 
                    #print('ini', nn)
                    newNode = "{}{}".format(nn,n2)
                    n1 = newNode
                if n2 == nn: 
                    #print('end', nn)
                    newNode = "{}{}".format(nn,n1)
                    n2 = newNode
                graphO.add_edge(n1,n2)
                ed = graphO.get_node(newNode)    
                ed.attr["style"] = 'invisible'
                newEd = graphO.get_edge(n1,n2)
                newEd.attr["label"] = edLabel
                
        graphO.delete_node(nn)
    return graphO
    
    
#### networkx    

def drawNetwCbar(G, pos, nodeAttr='callFreq', edgeAttr='cpd', 
                 nodeCmap=plt.cm.Reds, edgeCmap=plt.cm.Blues ):
    '''
    draw network with color map from node attr and for edge attr
    '''
    nodeW = [G.node[n][nodeAttr] for n in G.nodes()]
    ## nodes
    cnodes = nx.draw_networkx_nodes(G, pos=pos, node_color=nodeW, 
                                    cmap=nodeCmap, alpha=0.6)                                
    plt.colorbar(cnodes)
    nx.draw_networkx_labels(G, pos)
    ## edges
    edgeW = [e[2][edgeAttr] for e in G.edges(data=True)]
    cedges = nx.draw_networkx_edges(G, pos=pos, edge_color=edgeW, edge_cmap=edgeCmap)
    plt.colorbar(cedges)
    plt.axis('off')