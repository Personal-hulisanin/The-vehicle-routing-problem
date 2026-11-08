import re
import math
import time
import numpy as np
from pathlib import Path
import pandas as pd
import utils


class AntColonyCVRP:
    def __init__(self, n_ants=10, n_iterations=50, alpha=1, beta=2, evaporation=0.5, Q=100, seed=None):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        self.rng = np.random.default_rng(seed)

        self.global_best_distance = float("inf")
        self.global_best_routes = None
        self.discarded = 0

    def construct_solution(self, D, demand, capacity, pheromone, K):
        n = len(D)
        unvisited = np.ones(n, dtype=bool)
        unvisited[0] = False
        routes = [[0]]
        load = 0
        current = 0

        while unvisited.any():
            fits = unvisited & (demand <= capacity - load)
            candidates = np.flatnonzero(fits)

            if candidates.size == 0:
                if len(routes) >= K or current == 0:
                    return None, None
                routes[-1].append(0)
                routes.append([0])
                current = 0
                load = 0
                continue

            tau = pheromone[current, candidates] ** self.alpha
            eta = (1.0 / np.maximum(D[current, candidates], 1e-10)) ** self.beta
            weights = tau * eta
            probabilities = weights / weights.sum()

            next_node = int(self.rng.choice(candidates, p=probabilities))

            routes[-1].append(next_node)
            unvisited[next_node] = False
            load += demand[next_node]
            current = next_node

        routes[-1].append(0)

        total_distance = sum(
            D[r[i], r[i + 1]] for r in routes for i in range(len(r) - 1)
        )

        return routes, float(total_distance)

    def solve(self, distances, demand, capacity, K):
        D = np.asarray(distances, dtype=float)
        demand = np.asarray(demand, dtype=float)
        n = len(D)
        pheromone = np.ones((n, n))

        for iteration in range(self.n_iterations):
            valid_routes = []
            valid_distances = []

            for ant in range(self.n_ants):
                routes, distance = self.construct_solution(D, demand, capacity, pheromone, K)

                if routes is None:
                    self.discarded += 1
                    continue

                valid_routes.append(routes)
                valid_distances.append(distance)

                if distance < self.global_best_distance:
                    self.global_best_distance = distance
                    self.global_best_routes = routes

            pheromone *= (1 - self.evaporation)

            for routes, distance in zip(valid_routes, valid_distances):
                deposit = self.Q / distance
                for r in routes:
                    for i in range(len(r) - 1):
                        a, b = r[i], r[i + 1]
                        pheromone[a, b] += deposit
                        pheromone[b, a] += deposit

        return self.global_best_distance, self.global_best_routes, self.discarded


def instance_size(item):
    name, inst = item
    match = re.search(r"-n(\d+)", name)
    if match:
        return int(match.group(1))
    return len(inst["demands"])


if __name__ == "__main__":

    N_RUNS = 10
    BASE_SEED = 2026

    results = []

    instances = utils.load_all_instances("data/christofides-et-al-1979-set-m")
    for name, inst in sorted(instances.items(), key=instance_size):
        D, demand = utils.data_inputs_to_cvrp(inst["demands"], inst["dist"])
        capacity = inst["capacity"]

        print(f"\n=== {name} ===")

        if len(demand) != len(D):
            print("Customer demand and Distance matrix length not matching")
            continue

        K = math.ceil(sum(demand) / capacity)

        run_costs = []
        run_times = []
        best_cost = float("inf")
        best_routes = None
        discarded = 0

        for run in range(N_RUNS):
            aco = AntColonyCVRP(n_ants=10, n_iterations=50, seed=BASE_SEED + run)

            start_time = time.time()
            cost, routes, run_discarded = aco.solve(D, demand, capacity, K)
            run_times.append(time.time() - start_time)
            discarded += run_discarded

            if routes is not None:
                run_costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_routes = routes

        print("K:", K)
        print("Best Total Cost:", best_cost if best_routes else None)

        if run_costs:
            print(f"Mean cost over {N_RUNS} runs: {np.mean(run_costs):.2f} (std {np.std(run_costs):.2f})")
        print(f"Mean ACO runtime: {np.mean(run_times):.4f}")
        print("Discarded ants:", discarded)

        if best_routes:
            print("Routes:")
            for r in best_routes:
                print(r)
        else:
            print("There is no feasible solution")

        results.append({
            "instance": name,
            "n_nodes": len(D),
            "capacity": capacity,
            "num_vehicles": K,
            "best_cost": best_cost if best_routes else None,
            "mean_cost": float(np.mean(run_costs)) if run_costs else None,
            "std_cost": float(np.std(run_costs)) if run_costs else None,
            "feasible_runs": len(run_costs),
            "mean_runtime_sec": float(np.mean(run_times)),
            "discarded_ants": discarded,
            "feasible": best_routes is not None,
        })

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "aco_results.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")