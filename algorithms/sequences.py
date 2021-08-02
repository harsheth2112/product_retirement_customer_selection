import numpy as np


from lp import mnl_sales_lp_retirement, ind_sales_lp_selection
from customer import CustomerSequence

threshold = 1e-10


def random_sequence(algorithm):
    available_counts = np.array([customer.counts for customer in algorithm.available_customers], dtype=float)
    if np.sum(available_counts) > 0:
        proportions = available_counts / np.sum(available_counts)
        return np.random.choice(algorithm.m, p=proportions)
    else:
        return -1


def greedy_select(algorithm, sub_opt_param=1):
    n = algorithm.n
    m = algorithm.m

    customer_sequence = CustomerSequence(m, num_keys=1)
    pseudo_sales = np.zeros(n)
    customer_selection_count = np.zeros(m)
    available_products = np.array(algorithm.initial_inventory - pseudo_sales >= 1, dtype=int)
    previous_assortment = []
    # Initialization of best assortments
    for j in range(m):
        assortment, revenue = algorithm.customers[j].optimal_assortment(available_products, algorithm.revenues,
                                                                        algorithm.order, algorithm.reverse_order)
        previous_assortment.append((assortment, revenue))

    s = 0
    while s < algorithm.T:
        best_index = -1
        best_revenue = 0
        best_assortment = np.zeros(algorithm.n)
        # Step 1: Update and compare optimal assortments
        for j in range(m):
            if customer_selection_count[j] < algorithm.customers[j].counts:
                # Check if selected assortment from previous iteration can still be used
                if np.all(available_products * previous_assortment[j][0] == previous_assortment[j][0]):
                    assortment, revenue = previous_assortment[j]
                else:
                    customer = algorithm.customers[j]
                    assortment, revenue = customer.optimal_assortment(available_products, algorithm.revenues,
                                                                      algorithm.order, algorithm.reverse_order)
                    previous_assortment[j] = (assortment, revenue)
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_index = j
                    best_assortment = np.copy(assortment)
        # Step 2: Find max customers that can be accommodated for best assortment
        if best_revenue > 0:
            # Customers left
            best_customer = algorithm.customers[best_index]
            best_customers_available = best_customer.counts - customer_selection_count[best_index]
            # Max time periods each product in assortment can be sold for
            purchase_probabilities = best_customer.purchase_probabilities(best_assortment)[:algorithm.n]
            max_purchase = np.floor(
                threshold + np.divide(algorithm.initial_inventory - pseudo_sales - 1, purchase_probabilities,
                                      out=np.inf * np.ones(algorithm.n), where=purchase_probabilities > 0)) + 1
            # Add customer block to heap
            max_customers = np.min(np.r_[max_purchase, best_customers_available])
            if max_customers == 0:
                import pdb
                pdb.set_trace()
            block = {"count": max_customers,
                     "customer": best_customer,
                     "index": best_index,
                     "key": (sub_opt_param*best_revenue,)}
            customer_sequence.add_customer_block(block)
            customer_selection_count[best_index] += max_customers
            pseudo_sales += max_customers * purchase_probabilities
            available_products = np.array(algorithm.initial_inventory - pseudo_sales >= 1, dtype=int)
            s += max_customers
        else:
            break
    return customer_sequence


def our_sequence(algorithm):
    assert algorithm.is_static
    return greedy_select(algorithm, sub_opt_param=1)


def near_opt_sequence(algorithm):
    assert algorithm.is_static
    return greedy_select(algorithm, sub_opt_param=algorithm.adjustment_param)


def mnl_lp_sequence(algorithm):
    assert algorithm.is_static
    m = algorithm.m
    n = algorithm.n
    customer_sequence = CustomerSequence(m, num_keys=2)
    args = (algorithm.revenues, algorithm.initial_inventory, algorithm.customers, algorithm.T)
    solution, opt = mnl_sales_lp_retirement(*args)
    sales = solution[:, :n]
    no_purchase = solution[:, n:n + 1]
    u = np.r_[[customer.attraction for customer in algorithm.customers]]
    offered_products = np.array(sales >= threshold, dtype=int)
    always_offered_products = np.array(np.abs(u * no_purchase - sales) <= threshold, dtype=int)
    counts = np.sum(solution, axis=1)
    for j in range(m):
        if int(round(counts[j])) > 0:
            block = {"count": int(round(counts[j])),
                     "customer": algorithm.customers[j],
                     "index": j,
                     "key": (algorithm.minimum_revenue(offered_products[j, :]),
                             algorithm.minimum_revenue(always_offered_products))}
            customer_sequence.add_customer_block(block)
    # print(algorithm.customer_sequence._data)
    return customer_sequence


def ind_lp_sequence(algorithm):
    assert algorithm.is_static
    m = algorithm.m
    n = algorithm.n
    customer_sequence = CustomerSequence(m, num_keys=2)
    args = (algorithm.revenues, algorithm.initial_inventory, algorithm.customers, algorithm.T)
    customer_counts, opt = ind_sales_lp_selection(*args)
    customer_counts = np.ceil(customer_counts - threshold)
    for j in range(m):
        if customer_counts[j] > 0:
            block = {
                "count": customer_counts[j],
                "customer": algorithm.customers[j],
                "index": j,
                "key": (0, -algorithm.customers[j].expected_revenue(np.ones(n), algorithm.revenues))
            }
            customer_sequence.add_customer_block(block)
    return customer_sequence


def naive_greedy_sequence(algorithm):
    assert not algorithm.is_static
    m = algorithm.m
    available = algorithm.previous_assortment * algorithm.remaining_products
    best_index = -1
    best_revenue = 0
    for j in range(m):
        if algorithm.available_customers[j].counts > 0:
            revenue = algorithm.available_customers[j].expected_revenue(available, algorithm.revenues)
            if revenue > best_revenue:
                best_index = j
                best_revenue = revenue
    return best_index


def optimal_greedy_sequence(algorithm):
    assert not algorithm.is_static
    m = algorithm.m
    available = algorithm.previous_assortment * algorithm.remaining_products
    best_index = -1
    best_revenue = 0
    for j in range(m):
        if algorithm.available_customers[j].counts > 0:
            customer = algorithm.available_customers[j]
            assortment, revenue = customer.optimal_assortment(available, algorithm.revenues, algorithm.order,
                                                              algorithm.reverse_order)
            if revenue > best_revenue:
                best_index = j
                best_revenue = revenue
    return best_index

# def get_better_assortment(customer, available, compulsory, algorithm):
#     u = np.copy(customer.attraction)
#     u_0 = 1 + np.sum(compulsory * u)
#     customer.attraction[compulsory == 1] = 0
#     customer.attraction /= u_0
#     assort = customer.calculate_optimal_assortment(np.bitwise_or(available, compulsory), algorithm.revenues,
#                                                    algorithm.order, algorithm.reverse_order)
#     customer.attraction = u
#     return assort
#
#
# def our_better_sequence(algorithm: Algorithm):
#     n = algorithm.n
#     fake_inventory = np.array(algorithm.initial_inventory, dtype=float)
#     available = np.array(fake_inventory >= 1, dtype=int)
#     compulsory = np.zeros(n, dtype=int)
#     m = algorithm.m
#     customer_selection_count = np.zeros(m)
#     s = 0
#     least_revenue = np.inf
#     while s < algorithm.T:
#         best_index = -1
#         best_revenue = 0
#         best_assortment = np.zeros(n)
#         for j in range(m):
#             if customer_selection_count[j] < algorithm.customers[j].counts:
#                 customer = algorithm.customers[j]
#                 assortment = get_better_assortment(customer, available, compulsory, algorithm)
#                 revenue = customer.expected_revenue(assortment, algorithm.revenues * available)
#                 if revenue > best_revenue:
#                     best_revenue = revenue
#                     best_index = j
#                     best_assortment = np.copy(assortment)
#
#         if best_index != -1:
#             min_revenue = algorithm.minimum_revenue(best_assortment)
#             if min_revenue < least_revenue:
#                 least_revenue = min_revenue
#             push = np.array(np.bitwise_and(algorithm.revenues >= least_revenue, 1 - available), dtype=bool)
#             compulsory[push] = 1
#
#             print(compulsory, available, best_assortment, best_revenue, min_revenue)
#             best_customer = algorithm.customers[best_index]
#             best_customers_left = best_customer.counts - customer_selection_count[best_index]
#             inventory_left = fake_inventory - 1
#             purchase_probabilities = best_customer.purchase_probabilities(best_assortment)[:n] * available
#             max_purchase = np.ceil(np.divide(inventory_left, purchase_probabilities,
#                                              out=np.inf * np.ones(n), where=purchase_probabilities > 0))
#             # Add customer to heap
#             max_customers = np.min(np.r_[max_purchase, best_customers_left])
#             if max_customers == 0:
#                 import pdb
#                 pdb.set_trace()
#             block = {"count": max_customers,
#                      "customer": best_customer,
#                      "index": best_index,
#                      "key": min_revenue,
#                      "assortment": None}
#             algorithm.customer_sequence.add_customer_block(block)
#             customer_selection_count[best_index] += max_customers
#             fake_inventory -= max_customers * purchase_probabilities
#             available = np.array(fake_inventory >= 1, dtype=int)
#             s += max_customers
#         else:
#             break
