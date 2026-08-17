import numpy as np
import random

class AntColonyCVRP:
    def __init__(self,n_ants=10, n_iterations=50, alpha=1, beta=2, evaporation=0.5, Q=100):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        
        self.global_best_distance = float("inf")
        self.global_best_path = None

    