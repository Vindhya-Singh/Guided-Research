import numpy as np
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import networkx as nx
import pandas as pd

from collections import Counter
from preprocess import *
from sub_find import *
from constants import *
from true_subs import *

def get_common_ingredients(recipe_list, n=20):
    """Returns the top n common ingredients occuring in recipe_list.

    Parameters
    ----------
    recipe_list : List
        List of objects of type Recipe or ReducedRecipe.
    n : int
        Specifies the number of top ingredients to return.

    Returns
    -------
    Tuple
        Returns both the most common n ingedients as well as the counts of all
        ingredients, to be used for the graph.

    """
    all_ings = []
    for r in recipe_list:
        all_ings.extend(r.proccessed_ing_list)
    counter = Counter(all_ings)
    size_dict = dict(counter)
    return counter.most_common(n), size_dict

def get_nodes(recipe_list, ing_sim_mat, id_2_ing, ing_2_id, common, neighbors, custom):
    """Returns the list of main ingredients and their substitution candidates.

    Parameters
    ----------
    recipe_list : list
        List of objects of type Recipe or ReducedRecipe.
    ing_sim_mat : matrix
        Matrix containing the ingredient similarity computed via PMI.
    id_2_ing : dict
        Mapping from ingredient id to ingredient text.
    ing_2_id : dict
        Mapping from ingredient text to ingredient id.
    common : int
        Number of top common ingredient to include in the graph and to calculate
        the substitutions for.
    neighbors : int
        Number of top substitution candiates to include in the graph.
    custom : list
        List of strings of ingredients to include in the graph. If given, then
        the number in "common" is ignored and this list is used instead.

    Returns
    -------
    Tuple
        result: list of ingredients to include in the graph.
        neighbors_dict: dict mapping between a main node and its substitutes
        size_dict: counts of all ingredients, to be used for the graph.

    """
    result = set()
    neighbors_dict = {}
    counter, size_dict = get_common_ingredients(recipe_list, common)
    if len(custom) > 0:
        common_ings = custom
    else:
        common_ings = [x[0] for x in counter]
    for i in common_ings:
        result.add(i)
        n = get_top_ranked(i, neighbors, ing_sim_mat, id_2_ing, ing_2_id)
        result.update(n)
        neighbors_dict[i] = n
    return list(result), neighbors_dict, size_dict

def construct_adjacency_matrix(recipe_list, ing_sim_mat, id_2_ing, ing_2_id,
                               common=10, neighbors=20, method="sim",
                               connect_all=False, custom=[]):
    """Constructs the adjacency matrix of the graph.

    Parameters
    ----------
    recipe_list : list
        List of objects of type Recipe or ReducedRecipe.
    ing_sim_mat : matrix
        Matrix containing the ingredient similarity computed via PMI.
    id_2_ing : dict
        Mapping from ingredient id to ingredient text.
    ing_2_id : dict
        Mapping from ingredient text to ingredient id.
    common : int
        Number of top common ingredient to include in the graph and to calculate
        the substitutions for.
    neighbors : int
        Number of top substitution candiates to include in the graph.
    method : string
        Defines the way of how to calculate the edges between the ingredients. If
        "sim", then the PMI similarity value is the weight of the edge, else then the
        weight is the 1/r where r is the rank of the substitute in the candidate list
    connect_all : boolean
        decides whether to connect all ingredients with their respective similarity scores
    custom : list
        List of strings of ingredients to include in the graph. If given, then
        the number in "common" is ignored and this list is used instead.

    Returns
    -------
    Tuple
        adj_mat: Adjacency matrix of the graph
        nodes: list of ingredients to include in the graph.
        neighbors_dict: dict mapping between a main node and its substitutes
        size_dict: counts of all ingredients, to be used for the graph.

    """
    nodes, neighbors_dict, size_dict = get_nodes(recipe_list, ing_sim_mat, id_2_ing, ing_2_id, common, neighbors, custom)
    print(nodes)
    nodes2id = {i:e for e,i in enumerate(nodes)}
    print(nodes2id)
    dim = len(nodes)
    adj_mat = np.zeros((dim, dim))
    if not connect_all:
        for i in neighbors_dict:
            from_id = nodes2id[i]
            for e,n in enumerate(neighbors_dict[i]):
                to_id = nodes2id[n]
                if method == "sim":
                    adj_mat[from_id][to_id] = ing_sim_mat[ing_2_id[i]][ing_2_id[n]]
                else:
                    adj_mat[from_id][to_id] = 1/e
    else:
        for i in nodes:
            from_id = nodes2id[i]
            for n in nodes:
                to_id = nodes2id[n]
                if method == "sim":
                    adj_mat[from_id][to_id] = ing_sim_mat[ing_2_id[i]][ing_2_id[n]]
                else:
                    adj_mat[from_id][to_id] = get_rank(i, n, ing_sim_mat, ing_2_id)

    return adj_mat, nodes, size_dict, neighbors_dict

def plot_documents_graph(dist_matrix):
    """Constructs the networkx graph instance to be plotted.

    Parameters
    ----------
    dist_matrix : matirx
        Adjacency matrix of the graph.

    Returns
    -------
    Tuple
        G1: c.
        pos: positions of the nodes in the graph.

    """
    G1=nx.from_numpy_matrix(dist_matrix*100000)
    pos=nx.spring_layout(G1)
    return G1, pos

def plotly_graph(G, pos, common, neighbors, size_dict, neighbors_dict, prefix, labels=[], custom=[]):
    """Plots the networkx graph using plotly.

    Parameters
    ----------
    G : networkx graph
        Graph containing the info about nodes and edges.
    pos : list
        List of x and y positions of the nodes.
    common : int
        Number of top common ingredient to include in the graph and to calculate
        the substitutions for.
    neighbors : int
        Number of top substitution candiates to include in the graph.
    size_dict : dict
        counts of all ingredients, to be used for the graph.
    neighbors_dict : dict
        maps each of the main nodes to their top sustitutes in an ordered list
    prefix : string
        Prefix of the filename to write to.
    labels : list
        List of strings to appear on the nodes in the graph.
    custom : list
        List of strings of ingredients to include in the graph. If given, then
        the number in "common" is ignored and this list is used instead.

    Returns
    -------
    None
        Writes the plotly graph to a HTML file.

    """
    weights = [G[e[0]][e[1]]['weight'] for e in G.edges]
    a = np.array(weights)
    weights = np.interp(a, (a.min(), a.max()), (1, 4))
    edge_trace = [go.Scatter(
    x=tuple([pos[e[0]][0], pos[e[1]][0], None]),
    y=tuple([pos[e[0]][1], pos[e[1]][1], None]),
    line=dict(width=weights[k],color='#000'),
    hoverinfo='none',
    mode='lines') for k,e in enumerate(G.edges)]

    a = np.array([size_dict[x] for x in labels])
    sizes = np.interp(a, (a.min(), a.max()), (10, 100))
    hover = []
    for l in labels:
        if l in neighbors_dict:
            hover.append(str({l:neighbors_dict[l]}))
        else:
            hover.append(l)
    node_trace = go.Scatter(
        x=[],
        y=[],
        text = labels,
        mode='markers+text',
        hoverinfo="text",
        marker = dict(
            color = 'green',
            size = sizes
        ),
            line=dict(width=2))

    for n,node in enumerate(G.nodes()):
        x, y = pos[node][0], pos[node][1]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    if len(custom)>0:
        t = f"Top {neighbors} substitutes for the {str(custom)}"
    else:
        t = f"Top {neighbors} substitutes for the {common} most common ingredients"

    fig = go.Figure(data=edge_trace +[node_trace],
             layout=go.Layout(
                title=t,
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    if len(custom)>0:
        f = f"{prefix}_custom_top_{neighbors}.html"
    else:
        f = f"{prefix}_common_{common}_top_{neighbors}.html"

    print(f"Writing graph to {f}")
    plotly.offline.plot(fig, filename=f, auto_open=False)

def graph_call(recipe_list, ing_sim_mat, id_2_ing, ing_2_id, common, neighbors, prefix, custom=[], connect_all=False):
    """A wrapper to call all graph evaluation methods

    Parameters
    ----------
    recipe_list : list
        List of objects of type Recipe or ReducedRecipe.
    ing_sim_mat : matrix
        Matrix containing the ingredient similarity computed via PMI.
    id_2_ing : dict
        Mapping from ingredient id to ingredient text.
    ing_2_id : dict
        Mapping from ingredient text to ingredient id.
    common : int
        Number of top common ingredient to include in the graph and to calculate
        the substitutions for.
    neighbors : int
        Number of top substitution candiates to include in the graph.
    prefix : string
        Prefix of the filename to write to.
    custom : list
        List of strings of ingredients to include in the graph. If given, then
        the number in "common" is ignored and this list is used instead.
    connect_all : boolean
        decides whether to connect all ingredients with their respective similarity scores

    Returns
    -------
    None

    """
    adj, nodes, size_dict, neighbors_dict = construct_adjacency_matrix(recipe_list, ing_sim_mat, id_2_ing, ing_2_id, common, neighbors, connect_all=connect_all, custom=custom)
    G, pos = plot_documents_graph(adj)
    plotly_graph(G, pos, common, neighbors, size_dict, neighbors_dict, prefix, labels=nodes, custom=custom)
