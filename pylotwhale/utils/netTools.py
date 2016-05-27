# -*- coding: utf-8 -*-
"""
network tools
Created on Fri May 27 22:12:01 2016


@author: florencia
"""

from __future__ import print_function
import networkx as nx

import pygraphviz as pgv



def nxGraph2pgv(G):
    return nx.to_agraph(G)
    
    
def drawGraphviz(A, dotFile=None, figName=None):
    '''draws graphviz graph (A)'''
    if dotFile: A.write(dotFile)  # write previously positioned graph to PNG file
    A.layout(prog='dot')
    if figName: A.draw(figName)
    

def conceptualiseNodes(graphO, nodeLi=None):
    '''invisubilises nodes in nodeLi, useful for removing _ini and _end
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
        graphO.delete_node(nn)
    return graphO

