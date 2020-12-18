import numpy as np
import statsmodels.api as sm

def bestfit(x, y):
    x_const = sm.add_constant(x)
    mod = sm.OLS(y, x_const)
    res = mod.fit()
    bestfit = (res.params[0], res.params[1])
    bestfit_line = (np.arange(-100, 100), res.params[0] + res.params[1] * np.arange(-100, 100))
    return bestfit, bestfit_line, res

def bootstrap(vals, num_iters, siglevel):
    bs_values = []
    for i in range(num_iters):
        bs_values.append(np.mean(np.random.choice(vals, len(vals), replace=True)))
    bs_values = np.sort(bs_values)
    lb = bs_values[int(num_iters * siglevel) - 1]
    ub = bs_values[num_iters - int(num_iters * siglevel) - 1]
    estimate = np.median(bs_values)
    std = np.std(bs_values)
    return estimate, lb, ub, std

def get_axis_range(type):
    if type == 'fr':
        return range(-50, 50)
    else:
        return range(-30, 30)


