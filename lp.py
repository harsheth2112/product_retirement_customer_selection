import gurobipy as gp
import numpy as np


def total_product_sales(sales):
    """
    Input Parameters:
    sales: m x (n+1) matrix with sales for each customer type
    Output: (n+1) length column for total sales of each product
    """
    return np.sum(sales, axis=0)


def mnl_sales_lp(revenues, initial_inventory, customers, time_horizon, proportions=None):
    n = revenues.size
    if type(customers).__name__ == "ndarray":
        attraction = np.vstack(customers)
        m = int(attraction.size / n)
    else:
        m = len(customers)
        attraction = np.vstack([customers[j].attraction for j in range(m)])
    if m == 1:
        proportions = np.ones(1)
    model = gp.Model("mnl_lp")
    model.setParam('OutputFlag', False)
    var_sales = model.addVars(m, n + 1, ub=time_horizon, name='sales')
    var_sales_by_product = model.addVars(n, name='sales_by_product')
    model.addConstrs((var_sales_by_product[i] == gp.quicksum([var_sales[j, i] for j in range(m)]) for i in range(n)))
    model.addConstrs((var_sales_by_product[i] <= initial_inventory[i] for i in range(n)), name="supply")
    model.addConstrs((var_sales[j, i] <= attraction[j, i] * var_sales[j, n] for i in range(n) for j in range(m)),
                     name="mnl")
    model.addConstrs(
        (gp.quicksum([var_sales[j, i] for i in range(n + 1)]) == proportions[j] * time_horizon for j in range(m)),
        name="demand")

    revenue = gp.LinExpr(list(revenues), var_sales_by_product.values())
    model.setObjective(revenue, gp.GRB.MAXIMIZE)
    model.optimize()
    # sol = model.getAttr('x', variables)
    # print(np.array(sol))
    sales = np.array([model.getVarsNyName('sales[{},{}]'.format(j, i)).x for i in range(n+1) for j in range(m)])
    no_purchase = sales[:, -1:]
    sales = sales[:, n]
    value = model.getObjective().getValue()
    # duals = np.array([v.getAttr('Pi') for v in model.getConstrs()])
    # thetas = duals[n:2 * n]
    # betas = duals[2 * n:(2 + m) * n].reshape((n, m)).T
    # lambda_alt = duals[-m:]
    return sales, no_purchase, value


def mnl_sales_lp_retirement(revenues, inventories, customers, time_horizon):
    n = revenues.size
    m = len(customers)
    model = gp.Model("het_mnl_lp")
    model.setParam('OutputFlag', False)
    var_sales = model.addVars(m, n + 1, obj=list(np.r_[-revenues, 0]) * m)
    model.addConstrs(
        (var_sales[j, i] <= customers[j].attraction[i] * var_sales[j, n] for i in range(n) for j in range(m)),
        name="utility")
    model.addConstrs((gp.quicksum([var_sales[j, i] for j in range(m)]) <= inventories[i] for i in range(n)),
                     name="capacity")
    model.addConstrs((gp.quicksum([var_sales[j, i] for i in range(n + 1)]) <= customers[j].counts for j in range(m)),
                     name="customer")
    model.addConstr(gp.quicksum(var_sales) <= time_horizon, name="horizon")
    model.optimize()
    sales = np.array([v.x for v in model.getVars()]).reshape((m, n + 1))
    relevant_customers = np.sum(sales[:, :n], axis=1) > 0
    sales[:, -1] = relevant_customers*sales[:, -1]
    value = -1 * model.getObjective().getValue()
    return sales, value


def ind_sales_lp_selection(revenues, inventories, customers, time_horizon):
    m = len(customers)
    n = revenues.size
    # expected_revenues = np.array([customer.expected_revenue(np.ones(n), revenues) for customer in customers])
    model = gp.Model("het_ind_lp")
    model.setParam('OutputFlag', False)
    var_customer_counts = model.addVars(m, name="customers", ub=[customer.counts for customer in customers])
    var_sales = model.addVars(n, name="sales", ub=inventories)
    model.addConstrs(
        (var_sales[i] <= gp.quicksum([var_customer_counts[j] * customers[j].demand_rates[i] for j in range(m)]) for i
         in
         range(n)), name="idm")
    model.addConstr(gp.quicksum(var_customer_counts) <= time_horizon, name="horizon")
    revenue = gp.LinExpr(list(revenues), var_sales.values())
    model.setObjective(revenue, gp.GRB.MAXIMIZE)
    model.optimize()
    customer_counts = np.array([v.x for v in model.getVars()])
    value = model.getObjective().getValue()
    return customer_counts, value


def nested_assortments(solution, customer, output_assortment=False):
    threshold = 1e-10
    n = solution.size
    times = np.zeros(n)
    available = np.ones(n)
    assortments = []  # global initializer
    if output_assortment:
        assortments = [available[:n - 1]]
    while np.sum(solution) > threshold:
        purchase_probabilities = customer.purchase_probabilities(available)
        delta = np.min(
            np.divide(solution, purchase_probabilities, out=np.inf * np.ones(n), where=purchase_probabilities > 0))
        solution = solution - delta * purchase_probabilities
        times[np.where(available == 1)] += delta
        available = np.array(solution > threshold, dtype=int)
        if output_assortment:
            assortments.append(available[:n - 1])
    if output_assortment:
        return times, sorted(assortments, key=lambda x: np.sum(x), reverse=True)
    else:
        return times
