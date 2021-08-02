import pandas as pd
import numpy as np
import copy
import sys


import main
from customer import MNLCustomer
import algorithms.homogeneous as hom
import algorithms.heterogeneous as het
from lp import mnl_sales_lp_retirement
from misc.utils import inputs_from_file, generate_instances, get_paths


def base_testing(paths):
    final_df_list = []
    for sel in [0.25, 0.5, 0.75, 1.0]:
        for lf in [0.2, 0.4, 0.6, 0.8, 1.0]:
            parameters = {
                "input_file": paths["input_directory"] / "input_mnl_het_T1000_100.csv",
                "instance_count": 10,
                "iteration_count": 10,
                "load_factor": lf,
                "selection_rate": sel,
                "demand_spread": 0.8,
                "supply_spread": 0.8,
                "scale": 0.5
            }
            final_df_list.append(main.run_scenarios(parameters, MNLCustomer))
    pd.concat(final_df_list, ignore_index=True).to_csv(
        paths["results_directory"] / "het_mnl_results_test_xyz.csv")


def base_testing_remote(i1, i2, paths):
    sels = [0.25, 0.5, 0.75, 1.0]
    lfs = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    sel = sels[i1]
    lf = lfs[i2]
    parameters = {
        "input_file": paths["input_directory"] / "input_mnl_het_T1000_100.csv",
        "instance_count": 100,
        "iteration_count": 50,
        "load_factor": lf,
        "selection_rate": sel,
        "demand_spread": 0.8,
        "supply_spread": 0.8,
        "scale": 1
    }
    df = main.run_scenarios(parameters, MNLCustomer)
    df.to_csv("output/het_mnl_results_rmp_2021_2_%.2f_%.2f.csv" % (sel, lf), index=False)


def downscale_testing(paths):
    parameters = {
        "input_file": paths["input_directory"] / "input_mnl_het_T1000_100.csv",
        "instance_count": 20,
        "iteration_count": 40,
        "load_factor": None,
        "selection_rate": 0.75,
        "demand_spread": 0.8,
        "supply_spread": 0.8,
        "scale": 0.2
    }
    results_df = pd.DataFrame(columns=['load_factor', 'downscale', 'instance', 'ratio_to_optimal'])
    for lf in np.linspace(0.2, 1.2, 6):
        parameters["load_factor"] = lf
        r, c, customers = inputs_from_file(parameters["input_file"], MNLCustomer)
        instances = generate_instances(c, customers, parameters)
        for k in range(len(instances)):
            c_k, customers_k, time_horizon_k = instances[k]
            sales, opt = mnl_sales_lp_retirement(r, c_k, copy.deepcopy(customers_k), time_horizon_k)
            for ds in np.linspace(0, 1, 11):
                print("lf:{} ds:{} k: {}".format(lf, ds, k))
                algorithm = het.OurParametrizedAlgorithm(r, c_k, customers_k, time_horizon_k, param=ds)
                results = algorithm.run(parameters["iteration_count"])

                results_df = results_df.append({'load_factor': lf, 'downscale': ds, 'instance': k,
                                                'ratio_to_optimal': results["revenue"] / opt}, ignore_index=True)
    results_df = results_df.groupby(['load_factor', 'downscale']).agg({'ratio_to_optimal': 'mean'})
    results_df = results_df.pivot_table('ratio_to_optimal', 'downscale', 'load_factor')
    results_df.to_csv(paths["results_directory"] / "downscale_testing.csv")


def convergence_testing(paths):
    time_horizon = 1000
    revenues = np.array([1])
    attraction = 25 / time_horizon
    customers = [MNLCustomer([attraction], time_horizon)]
    num_runs = 1000
    # base = 2
    # result = []

    result_df = pd.DataFrame({
        'run_num': np.arange(num_runs)
    })

    for i in range(40, 51):
        print(i)
        c = i
        inventories = np.array([c])
        algorithm = hom.SingleMNLAlgorithm(revenues, inventories, customers, time_horizon)
        result_df['inventory_{}'.format(c)] = algorithm.run(num_runs)["revenue_iterations"]
    result_df.to_csv(paths["results_directory"] / 'hom_convergence_6.csv', index=False)


remote = False
if __name__ == '__main__':
    path_names = get_paths("paths.json")
    if not remote:
        np.random.seed(1405)
        base_testing(path_names)
    else:
        a = int(sys.argv[1]) - 1
        b = int(sys.argv[2]) - 1
        np.random.seed(1405 + a * 100 + b * 1000)
        base_testing_remote(a, b, path_names)
