import numpy as np
import pandas as pd
import random
import networkx as graph
import matplotlib.pyplot as plt

links = np.loadtxt('F:\IIITD\Semester_2\Information-Retrieval\A5\CA-GrQc.txt', delimiter='\t', dtype=int)


def create_adjacancy_matrix(link):
    num, index = np.unique(link[:, 0], return_index=True)
    size = len(num)
    print(size)

    num_list = []

    network = np.zeros((size, size))

    for row in link:
        if row[0] in num_list:
            i = num_list.index(row[0])
        else:
            i = len(num_list)
            num_list.append(row[0])

        if row[1] in num_list:
            j = num_list.index(row[1])
        else:
            j = len(num_list)
            num_list.append(row[1])

        network[i][j] = 1

    network = sub_sampling(network)
    g = graph.from_numpy_matrix(network)

    return network, g


def sub_sampling(network, r=False):
    if r:
        network = pd.DataFrame(network)
        index = random.sample(range(len(network)), 1000)

        for i in range(network.shape[0]):
            print(i)
            if i not in index:
                network = network.drop(i, axis=0)
                network = network.drop(i, axis=1)
        network = np.array(network)
    else:
        network = network[:1000, :1000]
    return network


def degree_net(network):
    degrees = []
    for i in range(network.shape[0]):
        row = network[i, :]
        c = np.count_nonzero(row)
        degrees.append(c)

    d, c = np.unique(degrees, return_counts=True)
    return d, c


def cluster_coefficient(network):
    cluster_coeff = []
    for i in range(network.shape[0]):
        row = network[i, :]
        c = np.count_nonzero(row)
        print("c : ", c)
        row = np.nonzero(row)[0]
        # print(row)
        edges = 0
        for j in range(len(row)):
            for k in range(j + 1, len(row), 1):
                edges += network[row[j]][row[k]]
        print("edges : ", edges)
        if c == 0 or c == 1:
            cf = 0
        else:
            cf = 2 * float(edges / (c * (c - 1)))
        cluster_coeff.append(cf)

    cluster_coeff = np.array(cluster_coeff)
    return cluster_coeff


def bfs(network, start):
    queue = []
    visited = [0] * len(network)
    # visited = []
    distance = []
    row = np.nonzero(network[start, :])[0]
    # visited.append((start))
    visited[start] = 1
    distance.append(0)
    queue.extend(row)
    parent_distance = [0] * len(row)

    for i in queue:
        # if i not in visited:
        if visited[i] == 0:
            # print("i : ", i)
            i_distance = parent_distance[queue.index(i)] + 1
            distance.append(i_distance)
            # print("here")
            i_row = np.nonzero(network[i, :])[0]
            # print("i_row : ", i_row)
            queue.extend(i_row)
            parent_distance.extend([i_distance] * len(i_row))
            visited[i] = 1
            # visited.append(i)

    return visited, distance


def all_shortest_path(network, r=False):
    tr = [1]
    if r:
        queue = []
        # visited = [0] * len(network)
        distance = []
        # row = np.nonzero(network[0, :])[0]

        distance.append(0)
        queue.extend(0)
        parent_distance = [0] * len(0)

        for i in queue:
            if i not in visited:
            # if visited[i] == 0:
                # print("i : ", i)
                i_distance = parent_distance[queue.index(i)] + 1
                distance.append(i_distance)
                # print("here")
                i_row = np.nonzero(network[i, :])[0]
                # print("i_row : ", i_row)
                queue.extend(i_row)
                parent_distance.extend([i_distance] * len(i_row))
                bfs(i)
                # visited[i] = 1
                # visited.append(i)
    else:
        return tr


def closeness_centrality(network):
    cc_all = []

    for i in range(len(network)):
        print("i : ", i)
        _, dis = bfs(network, i)
        # print(dis)
        c_val = np.sum(dis) + (len(network) - (len(dis) + 1)) * len(network)
        c_val = float(c_val / len(network))
        cc_all.append(c_val)

    return cc_all


def betweenness_centrality(network):
    bc_all = []

    for i in range(len(network)):
        _, dis = bfs(network, i)
        dis = all_shortest_path(network)
        # print(dis)
        c_val = np.sum(dis) + (len(network) - (len(dis) + 1)) * len(network)
        c_val = float(c_val / len(network))
        bc_all.append(c_val)


net, grap = create_adjacancy_matrix(links)

# degree distribution
degree, count = degree_net(net)

# plot degree distribution
plt.scatter(degree, count)
plt.title("degree distribution")
plt.show()

# cluster coefficients
cf = cluster_coefficient(net)
print("average cluster coefficient : ", np.mean(cf))

# closeness
cl = closeness_centrality(net)
print(cl)
print("mean closeness centrality : ", np.mean(cl))

# betweenness
graph_betweenness = graph.betweenness_centrality(grap)
print("betweenness centrality : ", graph_betweenness)
print(np.mean(degree))