import numpy as np
from misc.object_heap import ObjectHeap


class Customer:
    def __init__(self, counts):
        self.counts = counts
        self._optimal_assortment = None
        self._revenues = None
        self._products = None

    def optimal_assortment(self, products, revenues, order, reverse_order, largest=True):
        if self._optimal_assortment is None or np.any(self._optimal_assortment * products != self._optimal_assortment) \
                or np.any(revenues != self._revenues) or np.any(products - self._products > 0):
            self._optimal_assortment = self._calculate_optimal_assortment(products, revenues, order,
                                                                          reverse_order, largest)
            self._revenues = np.copy(revenues)
            self._products = np.copy(products)
        return self._optimal_assortment, self.expected_revenue(self._optimal_assortment, revenues)

    def _calculate_optimal_assortment(self, products, revenues, order, reverse_order, largest=True):
        return products

    def purchase_probabilities(self, assort):
        return assort

    def purchase(self, assort):
        prob = self.purchase_probabilities(assort)
        product = np.random.choice(prob.size, p=prob)
        return product

    def expected_revenue(self, assort, revenues):
        prob = self.purchase_probabilities(assort)
        prob = prob[:prob.size - 1]
        return np.sum(prob * revenues)


class MNLCustomer(Customer):
    def __init__(self, attraction, counts):
        self.attraction = np.array(attraction)
        super().__init__(counts)

    def purchase_probabilities(self, assort):
        utility_zero = np.r_[self.attraction, 1]
        n = self.attraction.size
        assort_zero = assort if assort.size == n+1 else np.r_[assort, 1]
        weights = utility_zero * assort_zero
        prob = weights / np.sum(weights)

        return prob

    def _calculate_optimal_assortment(self, products, revenues, order, reverse_order, largest=True):
        revenues_ordered = np.copy(revenues[order])
        products_ordered = np.copy(products[order])
        utilities = np.copy(self.attraction[order])
        assort = np.zeros(revenues.size)
        revenue = 0
        denominator = 1
        for i in range(revenues.size):
            if products_ordered[i] == 1:
                if revenues_ordered[i] >= revenue if largest else revenues_ordered[i] > revenue:
                    assort[i] = 1
                    revenue = (denominator * revenue + utilities[i] * revenues_ordered[i])
                    denominator += utilities[i]
                    revenue /= denominator
                else:
                    break
        return assort[reverse_order]


class IndCustomer(Customer):
    def __init__(self, demand_rates, counts):
        self.demand_rates = np.array(demand_rates)
        if np.sum(self.demand_rates) > 1:
            self.demand_rates = self.demand_rates/(1+np.sum(self.demand_rates))
        super().__init__(counts)

    def purchase_probabilities(self, assort):
        prob = assort * self.demand_rates
        prob = np.r_[prob, 1 - np.sum(prob)]
        return prob


class CustomerSequence(ObjectHeap):
    def __init__(self, size, num_keys):
        super().__init__(initial=None, keys=tuple([lambda x: x["key"][i] for i in range(num_keys)]))
        self._active = None
        self.size = size
        self.counts = np.zeros(self.size)

    def _push(self, customer_block):
        super().push(customer_block)
        self.counts[customer_block["index"]] += customer_block["count"]

    def add_customer_block(self, customer_block):
        self._push(customer_block)

    def update_customer(self):
        if self._active is None or self._active["count"] == 0:
            self._active = self.pop() if not self.is_empty else None
            if self._active is None:
                return
        self._active["count"] -= 1
        self.counts[self._active["index"]] -= 1

    def get_current_customer(self):
        return self._active

    @property
    def is_empty(self):
        if len(self._data) == 0:
            return True
        else:
            return False
