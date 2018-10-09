# -*- coding: utf-8 -*-
"""
network tools
Created on Fri May 27 22:12:01 2016


@author: florencia
"""
from __future__ import print_function
from collections import defaultdict

import networkx as nx
#import pygraphviz as pgv
import numpy as np
import matplotlib.pyplot as plt

import pylotwhale.utils.dataTools as daT
import pylotwhale.utils.plotTools as pT


#### pygraphviz
def nxGraph2pgv(G):
    '''converts a nx network into a pygraphviz net'''
    return nx.nx_agraph.to_agraph(G) #nx.to_agraph(G)    
    
def drawGraphviz(A, dotFile=None, figName=None):
    '''draws graphviz graph (A)'''
    if dotFile: A.write(dotFile)  # write previously positioned graph to PNG file
    A.layout(prog='dot')
    if figName: A.draw(figName)

def conceptualiseNodes(graphO, nodeLi=None, attrLi=None):
    """invisibilises nodes in nodeLi keeping the edge label if any,
    useful for removing _ini and _end    
    WARNING: self linking nodes can lead to undesirable outcomes
    Parameters:
    ----------
        graphO : pygraphviz object (see nxGraph2pgv )
        nodeLi : list of nodes to remove    
    """
    if nodeLi is None: nodeLi = []
    for nn in nodeLi: # remove node nn
        for edge in graphO.edges():
            if nn in edge:
                u, v = edge
                edge_attrs = dict(graphO.get_edge(u,v).attr)#graphO.get_edge_data(u,v)
                ed = graphO.get_edge( u, v)
                #edLabel = ed.attr['label']
                graphO.delete_edge(ed) ## delet edge
                ## create dummies nodes and edges
                n1, n2 = u, v
                if n1 == nn: # starting node of the edge
                    #print('ini', nn)
                    newNode = "{}{}".format(nn, n2)
                    n1 = newNode
                if n2 == nn:  # ending node of the edge
                    #print('end', nn)
                    newNode = "{}{}".format(nn, n1)
                    n2 = newNode
                graphO.add_edge(n1,n2)
                #ed = graphO.get_node(newNode)    
                graphO.get_node(newNode).attr["style"] = 'invisible'
                #graphO.get_edge(n1,n2).attr = edge_attrs
                #newEdge = graphO.get_edge(n1,n2)
                newEd = graphO.get_edge(n1,n2)#.attr = edge_attrs
                for atts, vals in edge_attrs.items():
                    newEd.attr[atts]=vals
                
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

class dict2network():
    """network object
        G: networkx graph"""
    def __init__(self, twoDimDict, rmEdge='default', rmNodes=None):
        if rmEdge is 'default': rmEdge = '_end', '_ini'

        self.rmEdge = rmEdge
        #self.rmNodes = rmNodes
        self.net_dict = twoDimDict
        self.G = dict2nxGraph(self.net_dict, rmEdge=self.rmEdge, rmNodes=rmNodes)

    def add_edge_attr(self, attr_name, edge_dict):
        """adds attrs to network, eg. penwidth"""
        return add_edge_attr(self.G, attr_name, edge_dict)
        
    def remove_edges(self, ebunch):
        """removes edges from ebunch (a list of tuples)"""
        self.G.remove_edges_from(ebunch)
        
    def remove_nodes(self, nodeLi):
        self.G.remove_nodes_from(nodeLi)

    def drawGraphviz(self, dot_file, fig_file, invisibleNodes='default'):
        if invisibleNodes == 'default': invisibleNodes = ['_ini', '_end']
        self.A = conceptualiseNodes( nx.nx_agraph.to_agraph(self.G), #nx.to_agraph(self.G), 
                                    invisibleNodes )
        drawGraphviz(self.A, dot_file, fig_file)


def dict2nxGraph(twoDimDict, rmEdge='default', rmNodes=None):
    """2dim dictionary to graphviz network
    Parameters
    ----------
    twoDimDict : two dim dictionary to define the network
    rmEdge : default removes the edge ('_end', '_ini')
    """
    #if invisibleNodes == 'default': invisibleNodes = ['_ini', '_end']
    if rmEdge is 'default': rmEdge = '_end', '_ini'
    
    G = nx.DiGraph(twoDimDict)
    G.remove_edge(*rmEdge)
    try: G.remove_nodes_from(rmNodes)
    except TypeError: 'cannot remove nodes'
    return G


def cfd2nxDiGraph(cfd, rmNodes='default'):
    if rmNodes == 'default':
        rmNodes = ['_ini', '_end']
    G0 = nx.DiGraph(cfd)
    for n in rmNodes:
        G0.remove_node(n)
    return G0


def add_edge_attr(G, attr_name, edge_dict):
    """
    adds edge attribute to networkx graph
    Parameters
    ----------
    G: nx.DiGraph
    attr_name
    edge_dict: 2-dim dict
        edge_dict[n1][n2] = <edge_value>
    """
    cpdw=[]
    for u, v in G.edges():
        try:
            ed = "{}".format(edge_dict[u][v])
        except:
            ed = ""
        cpdw.append(ed)
    nx.set_edge_attributes(G, name=attr_name, values=dict(zip(G.edges(), cpdw)))
    return G


def drawNetFrom2DimDict(twoDimDict, dot_file=None, fig_file=None,
                        edgeAttrs=None,
                        #edgeLabelDict=None, labelDecimals=1, 
                        rmEdge='default', rmNodes=None,
                        invisibleNodes='default'):
                            
    '''2dim dictionary to graphviz network
    Parameters
    ----------
    twoDimDict : two dim dictionary to define the network
    dot_file : graphviz netwotk generator
    fig_file : graph figure
    edgeAttrs: list of attributes
        [('<label>', <edgeDict>)]
        'label': graphviz attr, eg: 'label', 'penwidth'
        edgeDict: edge dictionary, eg. conditional_probabilities (cpdw)
        see http://www.graphviz.org/doc/info/attrs.html
    edgeLabel : dictionary (~twoDimDict) with the edge labels
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
    try:
        for attr, edge_dict in edgeAttrs:
            print(attr)
            #nx.set_edge_attributes(G, attr_name, dict(zip(G.edges(), edge_dict)))

            
            nx.set_edge_attributes(G, attr, dict(zip(G.edges(), edge_dict)))
    except TypeError: 'No edge attrs'

    # node attribute
    #nw = [len(df[df['call']== c]) for c in G.nodes()]
    #nx.set_node_attributes(G, 'callFreq', dict(zip(G.nodes(), nw)))
     

    A = conceptualiseNodes( nx.nx_agraph.to_agraph(G), invisibleNodes )#nx.to_agraph(G), invisibleNodes )
    drawGraphviz(A, dot_file, fig_file)
    return A
    



    
### network properties  
    
  
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
                    
                    
### dictionary formating for graph drawing
                           
def format_cpd(edgeLabelDict, labelDecimals=2, treshold=0.1):
    """takes a nested dict (cpd) and reformats its values"""
    cpf_f = defaultdict(dict)
    for x in edgeLabelDict.iterkeys():
        for y in edgeLabelDict[x]:
            if edgeLabelDict[x][y] < treshold:
                continue               
            
            cpf_f[x][y] = "{0:.{1}f}".format(edgeLabelDict[x][y], labelDecimals)
    return cpf_f


def format_cpd_width(edgeLabelDict, m=2, b=0):
    """takes a nested dict (cpd) and reformats its values"""
    cpf_f = defaultdict(dict)
    for x in edgeLabelDict.iterkeys():
        for y in edgeLabelDict[x]:
            cpf_f[x][y] = m*edgeLabelDict[x][y] + b
    return cpf_f


def format_cfd_width(edgeLabelDict, m=1, b=0):
    """takes a nested dict (cpd) and reformats its values
    m and b are parameters for tunning the widht of the arrows"""
    N = edgeLabelDict.N()
    cpf_f = defaultdict(dict)
    for x in edgeLabelDict.iterkeys():
        for y in edgeLabelDict[x]:
            cpf_f[x][y] = m*edgeLabelDict[x][y]/N + b
    return cpf_f
    