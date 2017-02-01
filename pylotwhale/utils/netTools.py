# -*- coding: utf-8 -*-
"""
network tools
Created on Fri May 27 22:12:01 2016


@author: florencia
"""

from __future__ import print_function
import networkx as nx
#import pygraphviz as pgv
import numpy as np
import matplotlib.pyplot as plt

import pylotwhale.utils.dataTools as daT
import pylotwhale.utils.plotTools as pT


#### pygraphviz
def nxGraph2pgv(G):
    '''converts a nx network into a pygraphviz net'''
    return nx.to_agraph(G)    
    
def drawGraphviz(A, dotFile=None, figName=None):
    '''draws graphviz graph (A)'''
    if dotFile: A.write(dotFile)  # write previously positioned graph to PNG file
    A.layout(prog='dot')
    if figName: A.draw(figName)

def conceptualiseNodes(graphO, nodeLi=None):
    """invisibilises nodes in nodeLi keeping the edge label if any,
    useful for removing _ini and _end    
    WARNING: self linking nodes can lead to undesirable outcomes
    Parameters:
    ----------
        graphO : pygraphviz object (see nxGraph2pgv )
        nodeLi : list of nodes to remove    
    """
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
    
    
#### gatherer functions

def drawNetFrom2DimDict(twoDimDict, dot_file=None, fig_file=None,
                        edgeLabelDict=None, labelDecimals=1, 
                        rmEdge='default', rmNodes=None,
                        invisibleNodes='default'):
                            
    '''2dim dictionaty to graphviz network
    Parameters:                            
    -----------
    twoDimDict : two dim dictionary to define the network
    dot_file : graphviz netwotk generator
    fig_file : graph figure
    edgeLabel : dicitonary (~twoDimDict) with the edge labels
    labelDecimal : number of decimals to print in the edge label
    rmEdge : default removes the edge ('_end', '_ini')
    invisibleNodes : default invisibilises the nodes '_ini' and '_end'
    '''
    
    if invisibleNodes == 'default': invisibleNodes =['_ini', '_end'] 
    if rmEdge is 'default': rmEdge = '_end', '_ini'
    
    G = nx.DiGraph(twoDimDict)
    G.remove_edge(*rmEdge)
    try: G.remove_nodes_from(rmNodes)
    except TypeError: 'canot remove nodes'

    ## edge attribute --> cpd
    if edgeLabelDict:
        cpdw = ["{0:.{1}f}".format(edgeLabelDict[u][v], labelDecimals) for u, v in G.edges()]
        nx.set_edge_attributes(G, 'label', dict(zip(G.edges(), cpdw)))

    # node attribute
    #nw = [len(df[df['call']== c]) for c in G.nodes()]
    #nx.set_node_attributes(G, 'callFreq', dict(zip(G.nodes(), nw)))
     

    A = conceptualiseNodes( nx.to_agraph(G), invisibleNodes )
    drawGraphviz(A, dot_file, fig_file)    
    
### network properties  
    
def cfd2nxDiGraph(cfd, rmNodes='default'):
    if rmNodes == 'default':
        rmNodes = ['_ini', '_end']
    G0 = nx.DiGraph(cfd)
    for n in rmNodes:
        G0.remove_node(n)
    return G0
  
    
def pl_nx_propertie(G, nx_property, pltitle=None, oFig=None):
    
    deg_di = dict(nx_property(G).items())
    calls_by_deg = daT.returnSortingKeys(deg_di)
    
    labs = []
    h = np.zeros((1, len(calls_by_deg)))
    for i, c in enumerate(calls_by_deg):
        h[0,i] = deg_di[c]
        labs.append(c)

    labs = np.array(labs)
    fig, ax = pT.stackedBarPlot(h, labs)
    if pltitle: ax.set_title(pltitle)
    if oFig: fig.savefig(oFig, bbox_inches='tight')

    return( fig, ax)
    
def pl_degree_centrality(G, pltitle= 'degree centrality', oFig=None):
    return pl_nx_propertie(G, nx_property=nx.degree_centrality,
                            pltitle=pltitle, oFig=oFig)
    
def pl_betweenness_centrality(G, pltitle='betweenness centrality',
                              oFig=None):
    return pl_nx_propertie(G, nx_property=nx.betweenness_centrality,
                           pltitle=pltitle, oFig=oFig)  