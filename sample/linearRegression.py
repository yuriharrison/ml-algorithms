"""Linear Regression Algorithm"""
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


def create_data(range, variance, step=2, correlation=False):
    """Generate a random data
    
    -- Arguments
    range: int, required
        range of the data

    variance: float, required
        range of variance of the random data

    step: int, optional
        value to generate correlational data

    correlation: 'pos', 'neg' or False, optional, default False
        Determines if the data will have 
        a positive or negative correlation or None

    -- Returns
        The generated data as a float64 numpy array
    """
    val = 1
    ys = []
    for _ in range(range):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    """Calculate the slope and the Y intercept of the best fit line

    -- Returns
        m: slope
        b: Y intercept
    """
    m = ( (mean(xs) * mean(ys) - (mean(ys*xs)))
        / (mean(xs)*mean(xs) - mean(xs*xs)) )
    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)


def coefficient_of_determination(ys, regression_line):
    y_mean_line = [mean(ys) for y in ys]
    squared_error_regr = squared_error(ys, regression_line)
    squared_error_y_mean = squared_error(ys, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


if __name__ == '__main__':
    #Data
    xs, ys = create_data(40, 20, 2, correlation='pos')

    #Calculation
    m, b = best_fit_slope_and_intercept(xs, ys)
    regression_line = [(m*x) + b for x in xs]

    #Prediction
    predict_x = 41
    predict_y = (m*predict_x) + b

    #Accuracy
    r_squared = coefficient_of_determination(ys, regression_line)
    print('Acuracy:', r_squared)

    plt.scatter(xs,ys)
    plt.scatter(predict_x, predict_y, s=100, color='g')
    plt.plot(xs, regression_line)
    plt.show()
