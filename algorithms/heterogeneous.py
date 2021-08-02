import numpy as np


from algorithms.algorithm import Algorithm
import algorithms.sequences as seq


class StaticAlgorithm(Algorithm):
    def __init__(self, revenues, initial_inventory, customers, time_horizon, sequence_function=seq.random_sequence):
        super(StaticAlgorithm, self).__init__(revenues, initial_inventory, customers, time_horizon, sequence_function,
                                              is_seq_static=True)

    def calculate_assortment(self, customer, t):
        block = self.customer_sequence.get_current_customer()
        return np.array(self.revenues >= block["key"][0]) * self.remaining_products


class NaiveGreedyAlgorithm(Algorithm):
    def __init__(self, revenues, initial_inventory, customers, time_horizon):
        super(NaiveGreedyAlgorithm, self).__init__(revenues, initial_inventory, customers, time_horizon,
                                                   seq.naive_greedy_sequence, is_seq_static=False)


class OurMNLAlgorithm(StaticAlgorithm):
    def __init__(self, revenue, initial_inventory, customers, time_horizon):
        super(OurMNLAlgorithm, self).__init__(revenue, initial_inventory, customers, time_horizon,
                                              sequence_function=seq.our_sequence)


class HeuristicLPMNLAlgorithm(StaticAlgorithm):
    def __init__(self, revenue, initial_inventory, customers, time_horizon):
        super(HeuristicLPMNLAlgorithm, self).__init__(revenue, initial_inventory, customers, time_horizon,
                                                      sequence_function=seq.mnl_lp_sequence)


class LPIndAlgorithm(StaticAlgorithm):
    def __init__(self, revenue, initial_inventory, customers, time_horizon):
        super(LPIndAlgorithm, self).__init__(revenue, initial_inventory, customers, time_horizon,
                                             sequence_function=seq.ind_lp_sequence)


class OurDynamicMNLAlgorithm(OurMNLAlgorithm):
    def __init__(self, revenue, initial_inventory, customers, time_horizon):
        super(OurDynamicMNLAlgorithm, self).__init__(revenue, initial_inventory, customers, time_horizon)

    def calculate_assortment(self, customer, t):
        available = self.previous_assortment * self.remaining_products
        assortment = super(OurDynamicMNLAlgorithm, self).calculate_assortment(customer, t)
        revenue = customer.expected_revenue(assortment, self.revenues)
        to_be_retired = available - assortment
        for i in range(self.n):
            if to_be_retired[i] == 1 and self.revenues[i] >= revenue:
                assortment[i] = 1
            else:
                break
        return assortment


class OurParametrizedAlgorithm(StaticAlgorithm):
    def __init__(self, revenue, initial_inventory, customers, time_horizon, param=0.75):
        super(OurParametrizedAlgorithm, self).__init__(revenue, initial_inventory, customers, time_horizon,
                                                       sequence_function=seq.near_opt_sequence)
        self.adjustment_param = param


class OurIndAlgorithm(NaiveGreedyAlgorithm):
    pass


class RandomNaiveMNLAlgorithm(Algorithm):
    pass


class SmartNaiveMNLAlgorithm(NaiveGreedyAlgorithm):
    pass


class EarlyRetirementAlgorithm(NaiveGreedyAlgorithm):
    def __init__(self, revenue, initial_inventory, customers, time_horizon):
        super(EarlyRetirementAlgorithm, self).__init__(revenue, initial_inventory, customers, time_horizon)
        self.selected = None

    def calculate_assortment(self, customer, t):
        return self.selected * self.remaining_products

    def reset(self):
        super(EarlyRetirementAlgorithm, self).reset()


class MaxEarlyRetirementAlgorithm(EarlyRetirementAlgorithm):
    def one_time(self):
        available = np.array(self.initial_inventory > 0, dtype=int)
        best_revenue = 0
        for j in range(self.m):
            if self.customers[j].counts > 0:
                customer = self.customers[j]
                assortment, revenue = customer.optimal_assortment(available, self.revenues, self.order,
                                                                  self.reverse_order)
                if revenue > best_revenue:
                    best_revenue = revenue
        self.selected = np.array(self.revenues >= best_revenue, dtype=int)


class SmartEarlyRetirementAlgorithm(EarlyRetirementAlgorithm):
    def one_time(self):
        available = np.array(self.initial_inventory > 0, dtype=int)
        best_size = 0
        best_assortment = np.zeros(self.n)
        for j in range(self.m):
            if self.customers[j].counts > 0:
                customer = self.customers[j]
                assortment, revenue = customer.optimal_assortment(available, self.revenues, self.order,
                                                                  self.reverse_order)
                if np.sum(assortment) > best_size:
                    best_size = np.sum(assortment)
                    best_assortment = np.copy(assortment)
        self.selected = best_assortment


# class MNLAlgorithm(Algorithm):
#     def calculate_assortment(self, *args):
#         block = self.customer_sequence.get_current_customer()
#         min_revenue = block["key"]
#         return np.array(np.logical_and(self.inventory > 0, self.revenues >= min_revenue), dtype=int)
#
#
# class MNLFixedAlgorithm(MNLAlgorithm):
#     def calculate_assortment(self, customer, t):
#         assortment = super().calculate_assortment(self)
#         u_assort = 1 + np.sum(assortment * customer.attraction)
#         r_assort = customer.expected_revenue(assortment, self.revenues)
#         r_ord = self.revenues[self.order]
#         to_be_retired = (self.available_products - assortment)[self.order]
#         u = customer.attraction[self.order]
#         for i in range(self.n):
#             if to_be_retired[i] == 1:
#                 if r_ord[i] >= r_assort:
#                     assortment[self.reverse_order[i]] = 1
#                     r_assort = (u_assort * r_assort + u[i] * r_ord[i])
#                     u_assort += u[i]
#                     r_assort /= u_assort
#         return assortment
