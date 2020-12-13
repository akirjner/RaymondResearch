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

def get_axis_range(s, gain, direction, type, diff):
    if diff:
        if type == 'fr':
            # if gain == 'x2':
            #     if direction == 'ipsi':
            #         if s == 0:
            #             return range(-40, 41)
            #         elif s == 1:
            #             return range(-60, 41)
            #         elif s == 2:
            #             return range(-50, 41)
            #         else:
            #             return range(-40, 41)
            #     else:
            #         if s == 0:
            #             return range(-40, 41)
            #         elif s == 1:
            #             return range(-40, 41)
            #         elif s == 2:
            #             return range(-40, 41)
            #         else:
            #             return range(-10, 41)
            return range(-50, 50)
        else:
            if gain == 'x2':
                if direction == 'ipsi':
                    if s == 0:
                        return range(-5, 1)
                    elif s == 1:
                        return range(-20, -9)
                    elif s == 2:
                        return range(-30, -14)
                    elif s == 3:
                        return range(-30, -14)
                else:
                    if s == 0:
                        return range(0, 5)
                    elif s == 1:
                        return range(10, 21)
                    elif s == 2:
                        return range(15, 31)
                    elif s == 3:
                        return range(15, 31)
            else:
                if direction == 'ipsi':
                    if s == 0:
                        return range(-1, 2)
                    elif s == 1:
                        return range(-15, -4)
                    elif s == 2:
                        return range(-21, -12)
                    elif s == 3:
                        return range(-18, -6)
                else:
                    if s == 0:
                        return range(-2, 2)
                    elif s == 1:
                        return range(8, 15)
                    elif s == 2:
                        return range(12, 24)
                    elif s == 3:
                        return range(9, 15)
    else:
        if type == 'fr':
            if gain == 'x2':
                if direction == 'ipsi':
                    if s == 0:
                        return range(-40, 41)
                    elif s == 1:
                        return range(-50, 11)
                    elif s == 2:
                        return range(-50, 51)
                    else:
                        return range(-50, 21)
                else:
                    if s == 0:
                        return range(-30, 31)
                    elif s == 1:
                        return range(-20, 51)
                    elif s == 2:
                        return range(-30, 41)
                    else:
                        return range(-10, 51)
            else:
                if direction == 'ipsi':
                    if s == 0:
                        return range(-40, 31)
                    elif s == 1:
                        return range(-50, 31)
                    elif s == 2:
                        return range(-40, 41)
                    else:
                        return range(-40, 41 )
                else:
                    if s == 0:
                        return range(-30, 31)
                    elif s == 1:
                        return range(-30, 41)
                    elif s == 2:
                        return range(-30, 41)
                    else:
                        return range(-40, 31)
        else:
            if gain == 'x2':
                if direction == 'ipsi':
                    if s == 0:
                        return range(-5, 1)
                    elif s == 1:
                        return range(-20, -10)
                    elif s == 2:
                        return range(-29, -16)
                    elif s == 3:
                        return range(-28, -15)
                else:
                    if s == 0:
                        return range(0, 5)
                    elif s == 1:
                        return range(10, 21)
                    elif s == 2:
                        return range(15, 31)
                    elif s == 3:
                        return range(15, 31)
            else:
                if direction == 'ipsi':
                    if s == 0:
                        return range(-2, 2)
                    elif s == 1:
                        return range(-10, -5)
                    elif s == 2:
                        return range(-21 , -12)
                    elif s == 3:
                        return range(-18, -6)
                else:
                    if s == 0:
                        return range(-2, 2)
                    elif s == 1:
                        return range(8, 15)
                    elif s == 2:
                        return range(12, 24)
                    elif s == 3:
                        return range(9, 15)

