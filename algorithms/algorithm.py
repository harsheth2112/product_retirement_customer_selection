import copy
import numpy as np

from algorithms.sequences import random_sequence


class Instance:
    """
    A class to allow the problem instance to move through the pipeline easily

    Attributes
    __________
    revenues : numpy.array(float)
        revenues of products
    initial_inventory : numpy.array(float)
        initial inventories of products
    customers : list(Customer)
        list of different customer types that can be selected from
    T : int
        time horizon
    n : int
        total number of products
    m : int
        total number of customer types
    revenue_order : numpy.array(int)
        order of products from highest to lowest revenue
    reverse_revenue_order : numpy.array(int)
        inverse function for revenue_order used to retrieve original revenue_order
    """
    def __init__(self, revenues, initial_inventory, customers, time_horizon):
        self.revenues = np.array(revenues)
        self._revenue_order = np.argsort(self.revenues)[::-1]
        self.reverse_revenue_order = np.argsort(self._revenue_order)
        self.initial_inventory = np.array(initial_inventory)
        self.customers = customers
        self.T = time_horizon
        self.n = self.revenues.size
        self.m = len(self.customers)

    @property
    def preference_weights(self):
        if type(self.customers).__name__ == "ndarray":
            return np.vstack(self.customers)
        else:
            return np.vstack([self.customers[j].attraction for j in range(self.m)])


class Algorithm:
    """
    A class providing the general framework to define product retirement and customer selection algorithms

    Attributes
    __________
    revenues : numpy.array(float)
        revenues of products
    initial_inventory : numpy.array(float)
        initial inventories of products
    customers : list(Customer)
        list of different customer types that can be selected from
    T : int
        time horizon
    n : int
        total number of products
    m : int
        total number of customer types
    sequence_function : fun
        function according to which customers are selected
    is_seq_static : bool
        is False is customer sequence is dynamic, is True if customer sequence is static
    customer_sequence : CustomerSequence
        current state of static customer sequence
    full_sequence : CustomerSequence
        copy of entire static customer sequence
    sales : numpy.array(int)
        current sales of products
    available_customers : list(Customer)
        copy of customers with current counts
    inventory : numpy.array(int)
        current inventory of products
    previous_assortment : numpy.array(int)
        assortment offered in previous time period
    times_offered : numpy.array(int)
        number of times products have been offered
    total_revenue : int
        total revenue currently earned
    order : numpy.array(int)
        order of products
    reverse_order : numpy.array(int)
        inverse function for order used to retrieve original order
    remaining_products : numpy.array(int)
        products with inventory remaining
    visited_customer_counts : list(int)
        number of times customers of each type have been visited
    """

    def __init__(self, revenues, initial_inventory, customers, time_horizon,
                 sequence_function=random_sequence, is_seq_static=False):
        self.revenues = np.array(revenues)
        self._order = np.argsort(self.revenues)[::-1]
        self.reverse_order = np.argsort(self._order)
        self.initial_inventory = np.array(initial_inventory)
        self.customers = customers
        self.T = time_horizon
        self.n = self.revenues.size
        self.m = len(self.customers)
        self.sequence_function = sequence_function
        self.is_static = is_seq_static

        self.full_sequence = None
        self.customer_sequence = None
        self.sales = None
        self.available_customers = None
        self.inventory = None
        self.previous_assortment = None
        self.times_offered = None

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = np.copy(order)
        self.reverse_order = np.argsort(order)

    @property
    def total_revenue(self):
        return np.sum(self.revenues * self.sales)

    @property
    def remaining_products(self):
        return np.array(self.inventory > 0, dtype=int)

    @property
    def visited_customer_counts(self):
        return np.array([self.customers[j].counts - self.available_customers[j].counts for j in range(self.m)])

    def reset(self):
        self.inventory = np.copy(self.initial_inventory)
        self.available_customers = copy.deepcopy(self.customers)
        self.sales = np.zeros(self.n)
        self.previous_assortment = self.remaining_products
        self.times_offered = np.zeros(self.n)
        self.customer_sequence = copy.deepcopy(self.full_sequence)

    def update_inventories(self, choice):
        if choice != self.n:
            self.inventory[choice] -= 1
            self.sales[choice] += 1

    def populate_sequence(self):
        self.full_sequence = self.sequence_function(self)

    def one_time(self):
        if self.is_static:
            self.populate_sequence()

    def preliminaries(self):
        self.reset()

    def update_customer_counts(self, customer_index):
        try:
            assert self.available_customers[customer_index].counts > 0
        except AssertionError:
            print(customer_index)
            print(self.visited_customer_counts)
            print([customer.counts for customer in self.customers])
            raise AssertionError
        self.available_customers[customer_index].counts -= 1

    def get_customer(self):
        if self.is_static:
            self.customer_sequence.update_customer()
            customer = self.customer_sequence.get_current_customer()
            if customer is not None:
                self.update_customer_counts(customer["index"])
                return customer["customer"]
            else:
                return None
        else:
            index = self.sequence_function(self)
            if index == -1:
                return None
            else:
                self.update_customer_counts(index)
                return self.available_customers[index]

    def calculate_assortment(self, customer, t):
        return self.remaining_products

    def get_assortment(self, customer, t):
        assortment = self.calculate_assortment(customer, t)
        self.times_offered += assortment
        self.previous_assortment = np.copy(assortment)
        return assortment

    def resolve(self):
        pass

    def minimum_revenue(self, assortment):
        rev_offered = self.revenues * assortment
        return np.min(rev_offered[np.nonzero(rev_offered)])

    def run(self, iters=1):
        res_revenue = np.zeros(iters)
        res_inv = np.zeros((iters, self.n))
        ret_times = np.zeros((iters, self.n))
        customer_selections = np.zeros((iters, self.m))
        self.one_time()
        for j in range(iters):
            print(".", end="")
            # print("Algorithm: {} Simulation Count: {}".format(self.__class__.__name__, j + 1), end="\r")
            self.preliminaries()
            for t in range(self.T):
                customer = self.get_customer()
                if customer is None:
                    break
                assortment = self.get_assortment(customer, t)
                choice = customer.purchase(assortment)
                self.update_inventories(choice)
                self.resolve()
            res_revenue[j] = self.total_revenue
            res_inv[j, :] = self.inventory
            ret_times[j, :] = self.times_offered
            customer_selections[j, :] = self.visited_customer_counts
        # print("Done", end='\r')
        print("\n")
        return {"revenue": np.mean(res_revenue), "revenue_std": np.std(res_revenue), "revenue_iterations": res_revenue,
                "inventory": np.mean(res_inv, axis=0), "retirement_times": np.mean(ret_times, axis=0),
                "customers": np.mean(customer_selections, axis=0), "success": True}
