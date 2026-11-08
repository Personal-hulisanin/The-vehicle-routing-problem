import math
from pathlib import Path
import re
import sys
import time
import heapq
import pandas as pd
import utils
from tqdm import tqdm


inf = float('inf')


class Node:
    def __init__(self):
        self.cost = 0
        self.bound = 0
        self.current_node = 0
        self.visited = set()
        self.load = 0
        self.routes = [[0]]

    def __lt__(self, other):
        if self.bound != other.bound:
            return self.bound < other.bound
        return len(self.visited) > len(other.visited)


def compute_bound(node, cost_matrix, demand, capacity, num_vehicles):
    n = len(cost_matrix)
    bound = node.cost
    cur = node.current_node

    unvisited = [i for i in range(1, n) if i not in node.visited]

    if not unvisited:
        if cur != 0:
            bound += cost_matrix[cur][0]
        return bound

    remaining = sum(demand[i] for i in unvisited)
    spare = capacity - node.load
    extra_routes = max(0, -(-(remaining - spare) // capacity))
    if len(node.routes) + extra_routes > num_vehicles:
        return inf

    targets = unvisited if cur == 0 else unvisited + [0]
    bound += min(cost_matrix[cur][j] for j in targets)

    for i in unvisited:
        bound += min(cost_matrix[i][j] for j in unvisited + [0] if j != i)

    if extra_routes:
        bound += extra_routes * min(cost_matrix[0][j] for j in unvisited)

    return bound


def expand_node(node, cost_matrix, demand, capacity, num_vehicles):
    n = len(cost_matrix)
    children = []

    for j in range(n):

        if j == node.current_node:
            continue

        if j != 0 and j in node.visited:
            continue

        new_node = Node()
        new_node.cost = node.cost
        new_node.current_node = j
        new_node.visited = node.visited.copy()
        new_node.routes = [r[:] for r in node.routes]
        new_node.load = node.load

        if j == 0:

            if node.current_node == 0:
                continue

            new_node.routes[-1].append(0)

            if len(new_node.visited) < n - 1:
                if len(new_node.routes) >= num_vehicles:
                    continue
                new_node.routes.append([0])

            new_node.load = 0

        else:

            if node.load + demand[j] > capacity:
                continue

            if node.current_node == 0 and len(node.routes) > 1:
                if j < node.routes[-2][1]:
                    continue

            new_node.routes[-1].append(j)
            new_node.load += demand[j]
            new_node.visited.add(j)

        new_node.cost += cost_matrix[node.current_node][j]

        children.append(new_node)

    return children


def solve_vrp(cost_matrix, demand, capacity, num_vehicles, time_limit=1800):
    n = len(cost_matrix)

    root = Node()
    root.current_node = 0
    root.bound = compute_bound(root, cost_matrix, demand, capacity, num_vehicles)

    pq = []
    if root.bound < inf:
        heapq.heappush(pq, root)

    best_cost = inf
    best_routes = None
    optimal = True

    start_time = time.time()

    with tqdm(desc="B&B", unit=" nodes") as progress_bar:

        while pq:

            if time.time() - start_time >= time_limit:
                optimal = False
                print(f"\nTime limit reached ({time_limit}s)")
                break

            current = heapq.heappop(pq)

            if current.bound >= best_cost:
                continue

            progress_bar.update(1)

            if len(current.visited) == n - 1:

                final_cost = current.cost

                if current.current_node != 0:
                    final_cost += cost_matrix[current.current_node][0]

                if final_cost < best_cost:

                    best_cost = final_cost

                    final_routes = [r[:] for r in current.routes]

                    if final_routes[-1][-1] != 0:
                        final_routes[-1].append(0)

                    best_routes = [r for r in final_routes if len(r) > 2]

                    progress_bar.set_postfix(
                        best_cost=best_cost,
                        queue=len(pq)
                    )

                continue

            for child in expand_node(current, cost_matrix, demand,
                                     capacity, num_vehicles):

                child.bound = compute_bound(child, cost_matrix, demand,
                                            capacity, num_vehicles)

                if child.bound < best_cost:
                    heapq.heappush(pq, child)

            if progress_bar.n % 500 == 0:
                progress_bar.set_postfix(
                    best_cost=best_cost,
                    queue=len(pq)
                )

    if optimal:
        best_bound = best_cost
    else:
        open_bound = min((nd.bound for nd in pq), default=best_cost)
        best_bound = min(open_bound, best_cost)

    if best_cost == inf or best_cost == 0:
        gap = None
    else:
        gap = 100 * (best_cost - best_bound) / best_cost

    return best_cost, best_routes, optimal, best_bound, gap


def instance_size(item):
    name, inst = item
    match = re.search(r"-n(\d+)", name)
    if match:
        return int(match.group(1))
    return len(inst["demands"])


if __name__ == "__main__":

    results = []

    instances = utils.load_all_instances(
        "data/christofides-et-al-1979-set-m"
    )

    for name, inst in sorted(instances.items(), key=instance_size):

        D, demand = utils.data_inputs_to_cvrp(
            inst["demands"],
            inst["dist"]
        )

        capacity = inst["capacity"]
        
        K = math.ceil(sum(demand) / capacity)

        print(f"\n=== {name} (K = {K}) ===")

        if len(demand) != len(D):
            print("Customer demand and Distance matrix length not matching")
            sys.exit(0)

        start_time = time.time()

        cost, routes, optimal, best_bound, gap = solve_vrp(D, demand, capacity, K, time_limit=1800)

        end_time = time.time() - start_time

        print("Minimum Total Cost:", cost)
        print("Status:", "OPTIMAL" if optimal else "TIME_LIMIT")
        print("Best Bound:", best_bound)

        if gap is not None:
            print(f"Gap (%): {gap:.2f}")

        print(f"Branch-and-Bound runtime: {end_time:.4f}")

        if routes:
            print("Routes:")
            for r in routes:
                print(r)
        else:
            print("There is no feasible solution")

        results.append({
            "instance": name,
            "n_nodes": len(D),
            "num_vehicles": K,
            "capacity": capacity,
            "best_cost": cost,
            "best_bound": best_bound,
            "gap_percent": gap,
            "runtime_sec": end_time,
            "status": "OPTIMAL" if optimal else "TIME_LIMIT",
            "feasible": routes is not None,
        })

    project_root = Path(__file__).resolve().parent.parent

    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "bnb_results.csv"

    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"Saved results to {output_path}")