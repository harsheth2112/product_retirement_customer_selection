import numpy as np


from algorithms.algorithm import Algorithm
from lp import mnl_sales_lp_retirement


class DeadlineAlgorithm(Algorithm):
    def __init__(self, revenue, initial_inventory, customers, time_horizon):
        super(DeadlineAlgorithm, self).__init__(revenue, initial_inventory, customers, time_horizon)
        self.deadlines = None

    def calculate_assortment(self, customer, t):
        return self.remaining_products * (t + 1 <= self.deadlines)


class TargetAlgorithms(Algorithm):
    def __init__(self, revenue, initial_inventory, customers, time_horizon):
        super(TargetAlgorithms, self).__init__(revenue, initial_inventory, customers, time_horizon)
        self.targets = None

    def calculate_assortment(self, customer, t):
        return self.remaining_products * (self.sales < self.targets)


class SingleMNLAlgorithm(DeadlineAlgorithm):
    def one_time(self):
        super(SingleMNLAlgorithm, self).one_time()
        threshold = 1e-10
        customer = self.customers[0]

        args = (self.revenues, self.initial_inventory, self.customers, self.T)
        solution, opt = mnl_sales_lp_retirement(*args)

        products_left = np.ones(self.n)

        simulated_sales_left = solution[0, :self.n]
        products_left[np.where(simulated_sales_left <= threshold)] = 0

        self.deadlines = np.ones(self.n) * self.T
        self.deadlines[np.where(simulated_sales_left <= threshold)] = 0
        t = 0

        while np.any(products_left):
            rates = customer.purchase_probabilities(products_left)
            max_sales = np.array(
                [simulated_sales_left[i] / rates[i] if products_left[i] else np.inf for i in range(self.n)])
            delta = np.min(max_sales)
            retiring = np.argmin(max_sales)
            simulated_sales_left = simulated_sales_left - delta * rates
            t = t + delta
            products_left[retiring] = 0
            self.deadlines[retiring] = np.ceil(t)
