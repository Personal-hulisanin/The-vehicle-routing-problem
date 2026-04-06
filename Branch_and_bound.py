import time
import heapq
import sys

inf = float('inf')


class Node:
    def __init__(self):
        self.cost = 0              # current cost
        self.bound = 0             # lower bound on total cost
        self.current_node = 0
        self.visited = set()
        self.load = 0              # current load
        self.routes = [[0]]        # list of routes

    def __lt__(self, other):
        return self.bound < other.bound


def compute_bound(node, cost_matrix):
    n = len(cost_matrix)

    bound = node.cost

    # for each unvisited customer node, add the cheapest outgoing edge
    for i in range(1, n):
        if i not in node.visited:
            min_edge = min(cost_matrix[i][j] for j in range(n) if j != i)
            bound += min_edge

    return bound


def expand_node(node, cost_matrix, demand, capacity):
    """
    ---------- Branching ----------
    """ 
    n = len(cost_matrix)
    children = []

    for j in range(n):
        
        if j == node.current_node:
            continue
        if j == 0 and node.current_node == 0:
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
            # return to depot
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
    """
    ---------- CVRP Branch-and-Bound Solver ----------
    """
    n = len(cost_matrix)

    root = Node()
    root.current_node = 0
    root.bound = compute_bound(root, cost_matrix=cost_matrix)

    pq = []
    heapq.heappush(pq, root)

    best_cost = inf
    best_routes = None

    while pq:
        current = heapq.heappop(pq)

        if current.bound >= best_cost:
            continue

        if len(current.visited) == n - 1:

            final_cost = current.cost
            if current.current_node != 0:
                final_cost += cost_matrix[current.current_node][0]

            if final_cost < best_cost:
                best_cost = final_cost

                final_routes = [r[:] for r in current.routes]
                if final_routes[-1][-1] != 0:
                    final_routes[-1].append(0)

                best_routes = final_routes

            continue

        # expand node
        children = expand_node(current, cost_matrix=cost_matrix, demand=demand, capacity=capacity)

        for child in children:
            child.bound = compute_bound(child, cost_matrix=cost_matrix)

            if child.bound < best_cost:
                heapq.heappush(pq, child)

    return best_cost, best_routes


if __name__ == "__main__":
    cost_matrix = [
        [0, 72, 184, 136, 115],
        [72, 0, 143, 180, 58],
        [184, 143, 0, 23, 182],
        [136, 180, 23, 0, 156],
        [115, 58, 182, 156, 0]
    ]

    demand = [0, 6, 6, 6, 6]
    capacity = 15

    if len(demand) != len(cost_matrix):
        print(" \nError: demand and cost matrix size mismatch")
        sys.exit(0)
    
    start_time = time.time()
    cost, routes = solve_vrp(cost_matrix, demand, capacity)
    bb_time = time.time() - start_time

    print("Minimum cost:", cost)
    print("Branch-and-Bound time: {:.4f} seconds".format(bb_time))

    if routes:
        print("Routes:")
        for r in routes:
            print(r)
    else:
        print("No feasible solution found")