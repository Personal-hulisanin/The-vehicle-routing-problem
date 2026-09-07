import numpy as np
from pathlib import Path
import time
import pandas as pd
import utils

class AntColonyCVRP:
    def __init__(self, n_ants=10, n_iterations=50, alpha=1, beta=2, evaporation=0.5, Q=100):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q

        self.global_best_distance = float("inf")
        self.global_best_routes = None

    def construct_solution(self, distances, demand, capacity, pheromone):
        n = len(distances)
        visited = set()
        routes = [[0]]
        load = 0
        current = 0

        while len(visited) < n - 1:
            candidates = [c for c in range(1, n)
                          if c not in visited and load + demand[c] <= capacity]

            if not candidates:
                # no feasible customer fits discard and start a new one route
                routes[-1].append(0)
                routes.append([0])
                current = 0
                load = 0
                continue

            probabilities = []
            for node in candidates:
                tau = pheromone[current][node] ** self.alpha
                try:
                    eta = (1 / distances[current][node]) ** self.beta
                except ZeroDivisionError:
                    eta = 1e10 ** self.beta
                probabilities.append(tau * eta)

            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()

            next_node = int(np.random.choice(candidates, p=probabilities))

            routes[-1].append(next_node)
            visited.add(next_node)
            load += demand[next_node]
            current = next_node

        # close the final route
        if routes[-1][-1] != 0:
            routes[-1].append(0)

        routes = [r for r in routes if len(r) > 2]  # drop empty [0,0] routes

        total_distance = sum(
            distances[r[i]][r[i + 1]]
            for r in routes for i in range(len(r) - 1)
        )

        return routes, total_distance

    def solve(self, distances, demand, capacity):
        n = len(distances)
        pheromone = np.ones((n, n))

        for iteration in range(self.n_iterations):
            all_routes = []
            all_distances = []

            for ant in range(self.n_ants):
                routes, distance = self.construct_solution(distances, demand, capacity, pheromone)
                all_routes.append(routes)
                all_distances.append(distance)

                if distance < self.global_best_distance:
                    self.global_best_distance = distance
                    self.global_best_routes = routes

            # evaporation
            pheromone *= (1 - self.evaporation)

            # deposit
            for routes, distance in zip(all_routes, all_distances):
                deposit = self.Q / distance
                for r in routes:
                    for i in range(len(r) - 1):
                        a, b = r[i], r[i + 1]
                        pheromone[a][b] += deposit
                        pheromone[b][a] += deposit

        return self.global_best_distance, self.global_best_routes


if __name__ == "__main__":

    results = []

    instances = utils.load_all_instances("data/christofides-et-al-1979-set-m")
    for name, inst in instances.items():
        D, demand = utils.data_inputs_to_cvrp(inst["demands"], inst["dist"])
        capacity = inst["capacity"]

        print(f"\n=== {name} ===")

        if len(demand) != len(D):
            print("Customer demand and Distance matrix length not matching")
            continue

        aco = AntColonyCVRP(n_ants=10, n_iterations=50)

        start_time = time.time()
        cost, routes = aco.solve(D, demand, capacity)
        end_time = time.time() - start_time

        print("Best Total Cost:", cost)
        print(f"ACO runtime: {end_time:.4f}")

        if routes:
            print("Routes:")
            for r in routes:
                print(r)
        else:
            print("There is no feasible solution")

        results.append({
            "instance": name,
            "n_nodes": len(D),
            "capacity": capacity,
            "best_cost": cost,
            "runtime_sec": end_time,
            "feasible": routes is not None,
        })

        # break

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "aco_results.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")