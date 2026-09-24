from pathlib import Path
import sys
import time
import math
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import utils


def build_and_solve(D, demand, capacity, time_limit=None, verbose=True):
    n = len(D) 
    customers = list(range(1, n))
    total_demand = sum(demand)
    K = math.ceil(total_demand / capacity)

    model = gp.Model("CVRP")
    model.Params.LazyConstraints = 1
    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit:
        model.Params.TimeLimit = time_limit
    
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    model.setObjective(gp.quicksum(D[i][j] * x[i, j] for i, j in x), GRB.MINIMIZE)

    for j in customers:
        model.addConstr(gp.quicksum(x[i, j] for i in range(n) if i != j) == 1, name=f"in_{j}")
    for i in customers:
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1, name=f"out_{i}")

    model.addConstr(gp.quicksum(x[0, j] for j in customers) == K, name="depot_out")
    model.addConstr(gp.quicksum(x[i, 0] for i in customers) == K, name="depot_in")

    def find_components(edges):
        adj = {c: set() for c in customers}
        for (i, j) in edges:
            if i != 0 and j != 0:
                adj[i].add(j)
                adj[j].add(i)

        visited = set()
        components = []
        for start in customers:
            if start in visited:
                continue
            comp = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in comp:
                    continue
                comp.add(node)
                stack.extend(adj[node] - comp)
            visited |= comp
            components.append(comp)
        return components

    def depot_touches(edges, comp):
        return any((i == 0 and j in comp) or (j == 0 and i in comp) for (i, j) in edges)

    def subtourelim(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        xvals = model.cbGetSolution(x)
        edges = [(i, j) for (i, j) in x if xvals[i, j] > 0.5]

        for comp in find_components(edges):
            comp_demand = sum(demand[k] for k in comp)
            min_vehicles = math.ceil(comp_demand / capacity)

            if not depot_touches(edges, comp) or min_vehicles > 1:
                model.cbLazy(
                    gp.quicksum(x[i, j] for i in comp for j in comp if i != j)
                    <= len(comp) - min_vehicles
                )

    model.optimize(subtourelim)

    if model.SolCount == 0:
        return None, None, model.Status

    xvals = model.getAttr("x", x)
    edges = [(i, j) for (i, j) in x if xvals[i, j] > 0.5]
    routes = []
    used = set()
    for (i, j) in edges:
        if i == 0 and (i, j) not in used:
            route = [0, j]
            used.add((i, j))
            current = j
            while current != 0:
                for (a, b) in edges:
                    if a == current and (a, b) not in used:
                        route.append(b)
                        used.add((a, b))
                        current = b
                        break
            routes.append(route)

    return model.ObjVal, routes, model.Status


if __name__ == "__main__":

    results = []

    instances = utils.load_all_instances("data/christofides-et-al-1979-set-m")
    for name, inst in instances.items():
        D, demand = utils.data_inputs_to_cvrp(inst["demands"], inst["dist"])
        capacity = inst["capacity"]

        print(f"\n=== {name} ===")

        if len(demand) != len(D):
            print("Customer demand and Distance matrix length not matching")
            sys.exit(0)

        start_time = time.time()
        cost, routes, status = build_and_solve(D, demand, capacity, time_limit=600)
        end_time = time.time() - start_time

        print("Minimum Total Cost:", cost)
        print(f"Gurobi runtime: {end_time:.4f}")

        if routes:
            print("Routes:")
            for r in routes:
                print(r)
        else:
            print("There is no feasible solution found")

        results.append({
            "instance": name,
            "n_nodes": len(D),
            "capacity": capacity,
            "best_cost": cost,
            "runtime_sec": end_time,
            "status": status,
            "feasible": routes is not None,
        })

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "gurobi_results.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")