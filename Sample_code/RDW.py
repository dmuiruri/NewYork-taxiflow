# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:42:13 2019

@author: litong
This code is used to obtain the embeddings of regions
"""

import csv
import numpy as np
import networkx as nx
import random
from gensim.models import Word2Vec


def deepwalk_walk(Graph, walk_length, start_node):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(Graph.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk


def simulate_walks(Graph, nodes, num_walks, walk_length):
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for v in nodes:
            walks.append(deepwalk_walk(Graph, walk_length, v))
    return walks


if __name__ == '__main__':

    C = 1
    time_slot = 4
    num_walks = 1000
    walk_length = 100
    embedding_size = 20
#
#    ###############build spatial graph########################
    Distance = np.load('data/drive_dist.npy', allow_pickle=True)
    num_zone = len(Distance)
    num_zone
    Spatial_graph = np.zeros(np.shape(Distance))
    for index_1 in range(num_zone):
        for index_2 in range(num_zone):
            Spatial_graph[index_1][index_2] = float(
                np.exp(-C*Distance[index_1][index_2]))

    SpatialGraph = np.zeros((int(24/time_slot)*num_zone,
                             int(24/time_slot)*num_zone))
    for index_1 in range(int(24/time_slot)):
        for index_2 in range(int(24/time_slot)):
            SpatialGraph[index_1*num_zone:(index_1+1)*num_zone,
                         index_2*num_zone:(index_2+1)*num_zone] = Spatial_graph

    SG = nx.DiGraph(SpatialGraph)
    #############################################################
#
#
    ##############build flow graph############################
    data = []
    with open('../data/taxi_flow_2016-01.csv') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader:
            data.append(row)

    clean_data = []
    for line in data:
        clean_data.append([line[0], line[1], line[-2], line[-1]])

    np.save('../data/OD_1601.npy', clean_data)

    data = np.load('../data/OD_1601.npy', allow_pickle=True)
    FlowGraph = np.zeros((int(24/time_slot)*num_zone, int(24/time_slot)*num_zone))
    for item in data:
        Stime = int(item[0].split(' ')[1].split(':')[0])
        Dtime = int(item[1].split(' ')[1].split(':')[0])
        Snode = int(item[2])
        Dnode = int(item[3])
        FlowGraph[Snode+int(Stime/time_slot)*num_zone, Dnode+int(Dtime/time_slot)*num_zone] \
            = FlowGraph[Snode+int(Stime/time_slot)*num_zone, Dnode+int(Dtime/time_slot)*num_zone] + 1
    FG = nx.DiGraph(FlowGraph)
    #############################################################
##
    print('Finished graph building!')
####
##
##
##
    nodes = []
    for i in range(num_zone*int(24/time_slot)):
        nodes.append(i)

    ###############Random walk############################

    Walks_SG = simulate_walks(SG, nodes, num_walks, walk_length)
    Walks_FG = simulate_walks(FG, nodes, num_walks, walk_length)
    Walks = Walks_SG+Walks_FG
    Walks = np.array(Walks)
#
#    np.save('../data/Walks.npy',Walks)
#    #######################################################
#
#
#
#
    ################Word to vector embedding###############
#    Walks = np.load('../data/Walks.npy')
    Str_Walks = []
    Walks_shape = np.shape(Walks)
    for line in Walks:
        Walk = []
        for item in line:
            Walk.append(str(item))
        Str_Walks.append(Walk)

    w2v_model = Word2Vec(Str_Walks, size=embedding_size)
#    w2v_model.save('../data/model')
#
    model = w2v_model
#    model = Word2Vec.load('../data/model')
    ZoneEmbed = []
    for N in nodes:
        ZoneEmbed.append(model[str(N)])
    ZoneEmbed = np.array(ZoneEmbed)
#    np.save('../data/ZoneEmbed.npy', ZoneEmbed)
#
#
#
#
#
