# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:33:03 2019

@author: rp13102
"""

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from dijsktra_algorithm import Graph, dijsktra_toall, dijsktra_tosome


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_edges(kernel):
    edges = []

    n_samples = kernel.shape[0]
    for i in range(n_samples):
        for j in range(n_samples):
            if kernel[i, j] != 0 :
                edges.append([i, j, kernel[i, j]])
    return edges

def plot_decision_boundary(X, y, func, method):
    h = 0.1
    xmin, ymin = np.min(X, axis=0)
    xmax, ymax = np.max(X, axis=0)

    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, h),
        np.arange(ymin, ymax, h)
        )

    cm = plt.cm.RdBu
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
    
    newx = np.c_[xx.ravel(), yy.ravel()]
        
    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Z = clf.predict_proba(newx)[:, 1]
    Z = func(newx)
    Z = Z.reshape(xx.shape)

    v=contour_plot = ax.contourf(
        xx, yy,
        Z, 
        levels=20,
        cmap=cm, 
        alpha=.8)
    
    ax.scatter(X[:, 0], X[:, 1], 
                c=y, 
                cmap=cm_bright,
                edgecolors='k',
                zorder=1)

    ax.grid(color='k', 
            linestyle='-', 
            linewidth=0.50, 
            alpha=0.75)

    plt.colorbar(v, ax=ax)
    return ax

def plot_path(X, path, ax, color='lightgreen', extra_point=None):
    if X.shape[1] != 2:
        return 0

    n_nodes = len(path)
    if isinstance(extra_point, np.ndarray):
        ax.plot([X[-1, 0], extra_point[0]],
                [X[-1, 1], extra_point[1]],
                'k', alpha=0.50)
        ax.scatter(extra_point[0], extra_point[1],
                    color='k',
                    marker='o',
                    facecolors='lightyellow',
                    edgecolors='lightyellow',
                    alpha = 0.80,
                    zorder=1,
                    s=250)
        
    args = {'color': 'lightgreen',
            'marker': 'x',
            's': 100}
    
    for idx in range(n_nodes-1):
        i = int(path[idx])
        j = int(path[idx + 1])
        ax.plot(X[[i, j], 0], X[[i, j], 1], 'k', alpha=0.50)
    
    ax.scatter(X[path[-1], 0], X[path[-1], 1],
                color='k',
                marker='o',
                facecolors=color,
                edgecolors=color,
                alpha = 0.50,
                zorder=2,
                s=150)

def plot_paths(X, method, howmanypaths, ax, all_paths):
    counter = 0
    colors=cm.Greens(np.linspace(0,1,howmanypaths))

    for idx, item in enumerate(all_paths):
        if counter > howmanypaths - 1:
            break
        path = item[-1]
        if method in ['kde']:
            plot_path(X, path, ax, colors[counter])
        else:
            plot_path(X, path, ax, colors[counter])
        counter += 1   
         
def plot_density(self, ax):
    newx = self.prepare_grid()
    if (self.n_features == 2):
        Z = np.exp(self.density_estimator.score_samples(newx))
        self.plot_density_scores(Z, ax)
         

class CFGenerator(object):
    def __init__(
        self,
        predictor,
        method=None,
        weight_function=None,
        prediction_threshold=None,
        density_threshold=None,
        K=None,
        radius_limit=None,
        n_neighbours=None,
        epsilon=None,
        distance_threshold=None,
        edge_conditions=None,
        howmanypaths=None,
        undirected=False
        ):
        
        self.edge_conditions = edge_conditions
        self.undirected = undirected
        
        if method in ['knn', 'kde', 'egraph']:
            self.method = method
        else:
            raise ValueError('Unknown method')
          
        if howmanypaths is None:
            self.howmanypaths = 5
        else:
            self.howmanypaths = howmanypaths
            
        if weight_function is None:
            self.weight_function = lambda x: -np.log(x)
        else:
            self.weight_function = weight_function

        if not hasattr(predictor, 'predict_proba'):
            raise ValueError('Predictor needs to have attribute: \'predict proba\'')
        else:
            self.predictor = predictor
    
        if prediction_threshold is None:
            self.prediction_threshold = 0.60    
        else:
            self.prediction_threshold = prediction_threshold
         
        if density_threshold is None:
            self.density_threshold = 1e-5
        else:
            self.density_threshold = density_threshold
            
        if K is None:
            self.K = 10
        else:
            self.K = K
        
        if epsilon is None:
            self.epsilon = 0.75
        else:
            self.epsilon = epsilon
            self.distance_threshold = distance_threshold
            
        if radius_limit is None:
            self.radius_limit = 1.10
        else:
            self.radius_limit = radius_limit
            
        if n_neighbours is None:
            self.n_neighbours = 20
        else:
            self.n_neighbours = n_neighbours
        
        if distance_threshold is None:
            self.distance_threshold = 'a'
        else:
            self.distance_threshold = distance_threshold
       
    def get_weights_e(self):
        k = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(i):
                v1 = self.X[j, :].reshape(-1, 1)
                if not self.check_conditions(v0, v1):
                    continue
                dist = np.linalg.norm(v0 - v1)
                if dist <= self.epsilon:
                    k[i, j] = self.weight_function(dist)
                    k[j, i] = k[i, j]
        return k

    def get_weights_kNN(self):       
        k = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            counter = 0
            for j in range(self.n_samples):
                v1 = self.X[j, :].reshape(-1, 1)
                dist = np.linalg.norm(v0 - v1)
                if (self.check_conditions(v0, v1) 
                    and dist <= self.distance_threshold):
                    k[i, j] = dist  #self.distance_threshold
                else:
                    counter += 1
            t = np.argsort(k[i, :])[(1+counter+self.n_neighbours):]
            mask = np.ix_(t)
            k[i, mask] = 0
            
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(self.n_samples):
                v1 = self.X[j, :].reshape(-1, 1)
                if k[i, j] != 0:
                    current_value = k[i, j]
                    k[i, j] = self.weight_function(current_value)
        return k

    def get_weights_kde(self, 
                    density_scorer,
                    mode):
        k = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(self.n_samples):#range(i):
                v1 = self.X[j, :].reshape(-1, 1)
                if not self.check_conditions(v0, v1):
                    continue
                dist = np.linalg.norm(v0 - v1)
                if dist <= self.distance_threshold:
                    midpoint = (v0 + v1)/2
                    density = density_scorer(midpoint.reshape(1, -1))
                    if mode == 1:
                        k[i, j] = self.weight_function(np.exp(density)) * dist
                    else:
                        k[i, j] = self.weight_function(sigmoid(density)) * dist
                else:
                    k[i, j] = 0
                #k[j, i] = k[i, j]
        return k
            
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        if self.n_samples != self.y.shape[0]:
            raise ValueError('Inconsistent dimensions')
        self.predictions = self.predictor.predict_proba(X)
        self.kernel = self.get_kernel()
        self.fit_graph()
 
    def get_kernel(self):
        if self.method == 'kde':
            self.get_kde()
            density_scorer = self.density_estimator.score_samples
            kernel = self.get_weights_kde(density_scorer, 1)

        elif self.method == 'knn':
            kernel = self.get_weights_kNN()
            
        elif self.method == 'egraph':
            kernel = self.get_weights_e()
         
        self.kernel = kernel
        return kernel
    
    def fit_graph(self):        
        self.graph = Graph(undirected=self.undirected)
        edges = get_edges(self.kernel)
        for edge in edges:
            self.graph.add_edge(*edge) 

    def condition(self, item):        
        pred = self.predictions[item, self.y[item]]
        if (self.y[item] == self.target_class
                and pred >= self.prediction_threshold):
            if self.method == 'kde':
                kde = np.exp(self.density_estimator.score_samples(self.X[int(item), :].reshape(1, -1))) 
                if kde >= self.density_threshold:
                    return (pred, kde), True
            elif self.method in ['knn', 'egraph']:
                return (pred), True
        return 0, False

    def check_conditions(self, v0, v1):
        if self.edge_conditions is None:
            return True
        else:
            return self.edge_conditions(v0, v1)
        
    def check_individual_conditions(self, v0, v1):
        return self.individual_edge_conditions(v0, v1)
    
    def modify_kernel(self):
        personal_kernel = self.kernel.copy()
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(i):
                v1 = self.X[j, :].reshape(-1, 1)

                if not self.check_individual_conditions(v0, v1):                   
                    personal_kernel[i, j] = 0
                    personal_kernel[j, i] = 0
        return personal_kernel
                    
    def compute_path(
        self, 
        starting_point,
        target_class,
        plot = True,
        individual_edge_conditions=None
        ):
        self.individual_edge_conditions = individual_edge_conditions
        if self.n_features != 2:
            plot = False
        self.target_class = target_class

        starting_point_index = np.where((self.X == starting_point).all(axis=1))[0][0]
        t0 = np.where(self.predictions >= self.prediction_threshold)[0]
        t1 = np.where(self.y == self.target_class)[0]
        if self.method == 'kde':
            kde = np.exp(self.density_estimator.score_samples(deepcopy(self.X))) 
            t2 = np.where(kde >= self.density_threshold)[0]
            self.candidate_targets = list(set(t0).intersection(set(t1)).intersection(set(t2)))
        else:
            self.candidate_targets = list(set(t0).intersection(set(t1)))
            
        if self.individual_edge_conditions is None:
            dist, paths = dijsktra_toall(
                                deepcopy(self.graph), 
                                starting_point_index
                                )
        else:
            self.personal_kernel = self.modify_kernel(self.kernel)
            self.personal_graph = Graph()
            edges = get_edges(self.personal_kernel)
            for edge in edges:
                self.personal_graph.add_edge(*edge) 
            dist, paths = dijsktra_tosome(self.personal_graph, 
                                          starting_point_index, 
                                          self.candidate_targets)
        
        if plot:
            def pred_func(x):
                return self.predictor.predict_proba(x)[:, 1]

            ax = plot_decision_boundary(self.X, self.y, pred_func, self.method)

        all_paths = []
        for item, path in paths.items():
            value, satisfied = self.condition(item)

            if satisfied:
                all_paths.append((item, self.X[item, :], dist[item], value, path))
        all_paths = sorted(all_paths, key=lambda x: x[2])
        
        if plot:
            plot_paths(self.X, self.method, self.howmanypaths, ax, all_paths)             
            
        return all_paths 

    def get_kde(self):
        bandwidths = np.logspace(-2, 0, 20)  
        #bandwidths = [0.65]
        grid = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            {'bandwidth': bandwidths},
            # cv=4,
        )    
        grid.fit(deepcopy(self.X))
        self.density_estimator = grid.best_estimator_
