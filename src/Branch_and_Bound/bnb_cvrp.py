import argparse
import sys
import time
import heapq
import pandas as pd


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
        return self.bound < other.bound


def compute_bound(node, cost_matrix, demand, capacity):
    n = len(cost_matrix)
    bound = node.cost

    unvisited = [i for i in range(1, n) if i not in node.visited]

    if not unvisited:
        return bound
    
    bound += min(cost_matrix[node.current_node][j] for j in unvisited + [0])

    for i in unvisited:
        min_edge = min(cost_matrix[i][j] for j in range(n) if j != i)
        bound += min_edge

    # lower bound
    demand_remaining = sum(demand[i] for i in unvisited)
    min_vehicles = (demand_remaining + capacity - 1) // capacity

    min_depot_edge = min(cost_matrix[0][j] for j in range(1, n))
    bound += min_vehicles * 2 * min_depot_edge

    return bound


def expand_node(node, cost_matrix, demand, capacity):
    n = len(cost_matrix)
    children = []

    for j in range(n):
        if j == node.current_node:
            continue
        if j != 0 and j in node.visited:
            continue
        if len(node.visited) == 0 and j != 1:
            continue

        new_node = Node()
        new_node.cost = node.cost
        new_node.current_node = j
        new_node.visited = node.visited.copy()
        new_node.routes = [r[:] for r in node.routes]
        new_node.load = node.load

        # Depot
        if j == 0:
            if node.current_node == 0:
                continue

            # Return if there is no feasible customer nodes remaining
            feasible = False
            for k in range(1, n):
                if k not in node.visited and node.load + demand[k] <= capacity:
                    feasible = True
                    break

            if feasible:
                continue

            new_node.routes[-1].append(0)
            new_node.routes.append([0])
            new_node.load = 0

        else:
            if node.load + demand[j] > capacity:
                continue

            new_node.routes[-1].append(j)
            new_node.load += demand[j]
            new_node.visited.add(j)

        new_node.cost += cost_matrix[node.current_node][j]

        children.append(new_node)

    return children


def solve_vrp(cost_matrix, demand, capacity):
    n = len(cost_matrix)

    root = Node()
    root.current_node = 0
    root.bound = compute_bound(root, cost_matrix, demand, capacity)

    pq = []
    heapq.heappush(pq, root)

    best_cost = inf
    best_routes = None

    while pq:
        current = heapq.heappop(pq)

        if current.bound >= best_cost:
            continue

        # All customers nodes has been visited
        if len(current.visited) == n - 1:

            final_cost = current.cost
            if current.current_node != 0:
                final_cost += cost_matrix[current.current_node][0]

            if final_cost < best_cost:
                best_cost = final_cost

                final_routes = [r[:] for r in current.routes]

                if final_routes[-1][-1] != 0:
                    final_routes[-1].append(0)

                final_routes = [r for r in final_routes if len(r) > 1]

                best_routes = final_routes

            continue

        # Expanding child nodes
        children = expand_node(current, cost_matrix, demand, capacity)

        for child in children:
            child.bound = compute_bound(child, cost_matrix, demand, capacity)

            if child.bound < best_cost:
                heapq.heappush(pq, child)

    return best_cost, best_routes


def load_data(distances_path, demands_path):
    cost_matrix = pd.read_csv(distances_path, header=None).values.tolist()
    demand = pd.read_csv(demands_path, header=None).values.tolist()[0]
    return cost_matrix, demand


def parse_args():
    parser = argparse.ArgumentParser(description="Branch and Bound for CVRP")
    parser.add_argument(
        "-d", "--distances",
        type=str,
        required=True,
        help="Path to the distance.csv file"
    )
    parser.add_argument(
        "-D", "--demands",
        type=str,
        required=True,
        help="Path to the demands.csv file"
    )
    parser.add_argument(
        "-c", "--capacity",
        type=int,
        default=None,
        help="Vehicle capacity (if omitted, you'll be prompted)"
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    D, demand = load_data(args.distances, args.demands)

    capacity = args.capacity
    if args.capacity is None:
        capacity = int(input(f"Enter vehicles capacity >= {max(demand)}: "))

    if len(demand) != len(D):
        print("Customer demand and Distance matrix length not matching")
        sys.exit(0)

    start_time = time.time()
    cost, routes = solve_vrp(D, demand, capacity)
    end_time = time.time() - start_time

    print("Minimum Total Cost:", cost)
    print("Branch and Bound runtime: {:.4f}".format(end_time))

    if routes:
        print("Routes:")
        for r in routes:
            print(r)
    else:
        print("There is no feasible solution")