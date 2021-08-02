import pandas as pd
import numpy as np
import copy

from misc.utils import generate_instances, inputs_from_file
import algorithms.heterogeneous as het
import algorithms.nested_policies as nested
from customer import MNLCustomer, IndCustomer
from lp import mnl_sales_lp_retirement, ind_sales_lp_selection


algorithm_list = {
    MNLCustomer: {
        "algorithms": [het.OurMNLAlgorithm, het.RandomNaiveMNLAlgorithm, het.SmartNaiveMNLAlgorithm,
                       het.MaxEarlyRetirementAlgorithm, nested.HetMNLNestedPolicy, het.OurParametrizedAlgorithm],
        "lp": mnl_sales_lp_retirement
    },
    IndCustomer: {
        "algorithms": [het.OurIndAlgorithm, het.LPIndAlgorithm],
        "lp": ind_sales_lp_selection
    }
}


def run_scenarios(params, customer_class):
    r, c, customers = inputs_from_file(params["input_file"], customer_class)
    instances = generate_instances(c, customers, params)
    iteration_count = params.get("iteration_count", 1)
    total_result_df = pd.DataFrame(
        columns=["selection_rate", "load_factor", "optimal"] + ['{}_{}'.format(alg.__name__, x) for alg in
                                                                algorithm_list[customer_class]["algorithms"] for x in
                                                                ['min', 'rev', 'max', 'std']])
    failed_instances = 0
    for k in range(len(instances)):
        print("Instance: {}".format(k + 1))
        c_k, customers_k, time_horizon_k = instances[k]
        print("Demand: {}".format(np.sum([customer.counts for customer in customers_k])), end=" ")
        print("Supply: {}".format(np.sum(c_k)), end=" ")
        print("T: {}".format(time_horizon_k), end="\n")
        sales, opt = algorithm_list[customer_class]["lp"](r, c_k, copy.deepcopy(customers_k), time_horizon_k)
        print("SLP value:", opt)
        result_df = pd.DataFrame(columns=total_result_df.columns)
        result_df = result_df.append({'selection_rate': params['selection_rate'],
                                      'load_factor': params['load_factor'],
                                      'optimal': opt}, ignore_index=True)
        for algorithm_class in algorithm_list[customer_class]["algorithms"]:
            algorithm = algorithm_class(r, c_k, customers_k, time_horizon_k)
            results = algorithm.run(iteration_count)
            result_df.loc[0, '{}_rev'.format(algorithm_class.__name__)] = results["revenue"]/opt
            result_df.loc[0, '{}_std'.format(algorithm_class.__name__)] = (results["revenue_std"]/opt)**2
            if algorithm_class.__name__ == "HetMNLNestedPolicy":
                if not results["success"]:
                    failed_instances += 1
        total_result_df = total_result_df.append(result_df, ignore_index=True)
    row_result_df = total_result_df.mean(skipna=False)
    row_result_df["failed_instances"] = failed_instances
    row_result_df.loc[['HetMNLNestedPolicy_rev', 'HetMNLNestedPolicy_std']] = total_result_df[
        ['HetMNLNestedPolicy_rev', 'HetMNLNestedPolicy_std']].mean()
    for alg in algorithm_list[customer_class]["algorithms"]:
        name = alg.__name__
        mean_variation = 0
        if len(instances) > 1:
            mean_variation = total_result_df[name + '_rev'].var()
        row_result_df[name + '_min'] = total_result_df[name + '_rev'].min(skipna=True)
        row_result_df[name + '_max'] = total_result_df[name + '_rev'].max(skipna=True)
        row_result_df[name + '_std'] = np.sqrt(
            (row_result_df[name + '_std'] + mean_variation) / (iteration_count * len(instances)))
    return pd.DataFrame(row_result_df).transpose()
