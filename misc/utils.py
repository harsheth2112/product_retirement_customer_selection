import pandas as pd
import numpy as np
import copy
import json
from pathlib import Path


def parse_json(file):
    with open(file, "r") as read_file:
        data = json.load(read_file)
    return data


def get_paths(json_file):
    paths_dict = parse_json(json_file)
    return {key: Path(paths_dict[key]) for key in paths_dict.keys()}


def inputs_from_file(file_loc, customer_class):
    inputs = pd.read_csv(file_loc, header=None)
    n = inputs.shape[1] - 1
    m = inputs.shape[0] - 2
    assert inputs.iloc[0, n] == n
    r = inputs.iloc[0, :n].astype(float).values
    c = inputs.iloc[1, :n].astype(int).values
    customers = [customer_class(inputs.iloc[j + 2, :n].astype(float).values,
                                int(inputs.iloc[j + 2, n])) for j in range(m)]
    return r, c, customers


def read_scenarios(products_file):
    products_dat = pd.read_csv(products_file, header=None)
    n = products_dat.shape[1]-1
    assert products_dat.shape[0] % 2 == 0
    num_scenarios = products_dat.shape[0]//2
    scenarios = []
    for i in range(num_scenarios):
        assert products_dat.iloc[2 * i, n] == n
        r = products_dat.iloc[2 * i, :n].astype(float).values
        c = products_dat.iloc[2 * i + 1, :n].astype(int).values
        time_horizon = int(products_dat.iloc[2 * i + 1, n])
        scenarios.append({'revenues': r, 'inventories': c, 'horizon': time_horizon})
    return scenarios


def generate_instances(c, customers, params: dict):
    total_customers = np.sum([customer.counts for customer in customers])
    supply = np.sum(c)

    load_factor = params.get("load_factor", total_customers / supply)  # demand / supply
    num = params.get("instance_count", 1)
    supply_spread = params.get("supply_spread", 0)
    demand_spread = params.get("demand_spread", 0)
    scale = params.get("scale", 1)
    selection_rate = params.get("selection_rate", 1)  # demand / total customers

    time_horizon = int(np.round(total_customers * selection_rate * scale, 0))
    n = c.size
    scenarios = []

    for i in range(num):
        new_customers = copy.deepcopy(customers)
        for j in range(len(new_customers)):
            epsilon_b = demand_spread * (2 * np.random.beta(1.5, 1.5) - 1)
            new_customers[j].counts = np.array(np.round(customers[j].counts * scale * (1 + epsilon_b)), dtype=int)
        epsilon_c = supply_spread * (2 * np.random.beta(1.5, 1.5, n) - 1)
        new_c = np.array(np.round((c / supply) * (time_horizon / load_factor) * (1 + epsilon_c), 0), dtype=int)
        scenarios.append((new_c, new_customers, time_horizon))
    return scenarios
