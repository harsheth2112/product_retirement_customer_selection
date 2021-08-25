import heapq as hq
import numpy as np
import copy


from algorithms.algorithm import Algorithm
from customer import CustomerSequence
from lp import mnl_sales_lp_retirement, nested_assortments


threshold = 1e-10


class NestedPolicy(Algorithm):
    def __init__(self, revenue, initial_inventory, customers, time_horizon, policy_constructor):
        new_initial_inventory = np.r_[initial_inventory, time_horizon]
        new_revenue = np.r_[revenue, 0.0001]
        new_customers = copy.deepcopy(customers)
        for customer in new_customers:
            customer.attraction = np.r_[customer.attraction, 100]
        super(NestedPolicy, self).__init__(new_revenue, new_initial_inventory, new_customers, time_horizon, is_seq_static=True)
        self.policy_constructor = policy_constructor
        self.probability_vectors = None

    def one_time(self):
        partial_sequence, probability_vectors = self.policy_constructor(self)
        self.full_sequence = partial_sequence
        self.probability_vectors = probability_vectors

    def preliminaries(self):
        super(NestedPolicy, self).preliminaries()
        for vector in self.probability_vectors:
            probabilities, assortments, customer_index = vector
            assortment = assortments[np.random.choice(len(probabilities), p=probabilities)]
            block = {"count": 1,
                     "customer": self.customers[customer_index],
                     "index": customer_index,
                     "assortment": np.copy(assortment),
                     "key": (-np.sum(assortment),)}
            self.customer_sequence.add_customer_block(block)

    def calculate_assortment(self, customer, t):
        available = self.previous_assortment * self.remaining_products
        target_assortment = self.customer_sequence.get_current_customer()["assortment"]
        return available * target_assortment


class HetMNLNestedPolicy(NestedPolicy):
    def __init__(self, revenue, initial_inventory, customers, time_horizon):
        super(HetMNLNestedPolicy, self).__init__(revenue, initial_inventory, customers, time_horizon,
                                                 policy_constructor=_mnl_nested_policy_sequence_constructor)

    def run(self, iters=1):
        args = (self.revenues, self.initial_inventory, self.customers, self.T)
        solution, opt = mnl_sales_lp_retirement(*args)
        if _mnl_is_nested(self, solution):
            return super(HetMNLNestedPolicy, self).run(iters)
        else:
            print("Assortments not consistent")
            return {"revenue": np.NaN, "revenue_std": np.NaN, "success": False}


def _mnl_is_nested(algorithm, solution):
    m = algorithm.m
    assortment_dict = {}
    for j in range(m):
        offer_times, assortments = nested_assortments(solution[j, :], algorithm.customers[j], output_assortment=True)
        for assortment in assortments:
            assortment_size = np.sum(assortment)
            if assortment_size in assortment_dict:
                if np.any(assortment_dict[assortment_size] != assortment):
                    print(j)
                    print(offer_times)
                    print(assortment_dict)
                    print(assortment)
                    return False
            else:
                assortment_dict[assortment_size] = np.copy(assortment)
    return True


def _mnl_nested_policy_sequence_constructor(algorithm):
    m = algorithm.m
    n = algorithm.n
    args = (algorithm.revenues, algorithm.initial_inventory, algorithm.customers, algorithm.T)
    solution, opt = mnl_sales_lp_retirement(*args)
    revenue_ordered_heap = []
    total_customers = 0
    heap_index = 0
    for j in range(m):
        customer = algorithm.customers[j]
        offer_times = nested_assortments(solution[j, :], customer)
        if not float(offer_times[n]).is_integer():
            offer_times[n] = np.ceil(offer_times[n] - threshold)
        total_customers += offer_times[n]
        available = np.array(offer_times > threshold, dtype=int)
        while np.sum(offer_times) > threshold:
            max_block_size = np.min(offer_times[np.where(available == 1)])
            block_size = int(np.floor(max_block_size + threshold))
            frac = max_block_size - block_size
            if block_size > 0:
                block = {"count": block_size,
                         "customer": customer,
                         "index": j,
                         "assortment": available[:n],
                         "key": (-np.sum(available),)}
                hq.heappush(revenue_ordered_heap,
                            (customer.expected_revenue(available, algorithm.revenues), heap_index, "block", block))
                heap_index += 1
            offer_times = offer_times - block_size * available
            if frac >= threshold:
                probabilities = [frac]
                assortments = [available[:n]]
                offer_times = offer_times - frac * available
                space = 1 - frac
                revenue = customer.expected_revenue(available, algorithm.revenues) * frac
                while space > threshold:
                    available = np.array(offer_times > threshold, dtype=int)
                    max_capacity = np.min(offer_times[np.where(available == 1)])
                    capacity = min(space, max_capacity)
                    probabilities.append(capacity)
                    assortments.append(available[:n])
                    offer_times = offer_times - capacity * available
                    revenue += customer.expected_revenue(available, algorithm.revenues) * capacity
                    space = space - capacity
                hq.heappush(revenue_ordered_heap, (revenue, heap_index, "p_vector", (probabilities, assortments, j)))
                heap_index += 1
            available = np.array(offer_times > threshold, dtype=int)
    while total_customers > algorithm.T:
        delta = total_customers - algorithm.T
        if revenue_ordered_heap[0][2] == "block":
            block_size = revenue_ordered_heap[0][3]["count"]
            remove_count = min(delta, block_size)
            total_customers = total_customers - remove_count
            if remove_count == block_size:
                hq.heappop(revenue_ordered_heap)
        else:
            hq.heappop(revenue_ordered_heap)
            total_customers = total_customers - 1
    customer_sequence = CustomerSequence(m, num_keys=1)
    probability_vector_list = []
    for elem in revenue_ordered_heap:
        if elem[2] == "block":
            customer_sequence.add_customer_block(elem[3])
        else:
            probability_vector_list.append(elem[3])
    return customer_sequence, probability_vector_list
