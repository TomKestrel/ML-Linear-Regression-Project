import matplotlib.pyplot as plt
from methods import *

"""
-----------------------------------------------------------------------------------------------------------------------
What is this about?

This is a self project to practice a machine learning concept into an application regarding linear regression with
one parameter (x[i]) value. The data is grabbed from Zillow which shows home sales in Point Loma in San Diego. 
Column 1 is the square footage value.   Will be denoted as "x_data" (will serve as the one parameter, x[i])
Column 2 is the asking sale price.      Will be denoted as "y_data"

Note: f(w,b) is called the Cost Function
The goal is to receive the smallest f(w,b) output through the application of gradient descent.
(This means have the machine optimize the w and b coefficient values that will minimize the f(w,b) output).

Self project goal:
- Plot a chart of the data for visualization purpose
- Create a method that will calculate y_hat prediction value. 
- Create a method that will apply the appropriate Gradient Descent algorithm
-----------------------------------------------------------------------------------------------------------------------
"""
#       [Charting the Data]
#----------------------------------------------------------------------------------------------------------------------
# loading the dataset
x_data, y_data = load_data()

# Scatter parameters
plt.scatter(x_data, y_data, marker='x', c='r')

# Title of chart
plt.title("Square Feet vs. Price of Home in Point Loma")
# y-axis label
plt.ylabel('Price of Home in thousands')
# x-axis label
plt.xlabel('Total Square Feet')
plt.show()
#----------------------------------------------------------------------------------------------------------------------

#   [Method to Calculate Cost Function]
def calculate_cost_function(x, y, w, b):
    """
    -------------------------------------------------------------------------------------------------------------------
    :param x: square footage
    :param y: price
    :param w: linear regression coefficient
    :param b: linear regression coefficient
    :return: f(w,b) which is the Cost Function value (which will then be utilized by a gradient descent algorithm)
    -------------------------------------------------------------------------------------------------------------------
    
    The Cost Function formula for linear regression application is called the "Square Error Cost Function" which is
    
    f(w,b) = (Cost Function value) 
    f(w,b) = (1/2m)E((y_hat)-(actual data))^2
                        |
                        |
                        v
    the prediction (y_hat) formula model for linear regression is:
    
    y_hat = m*x + b
    y_hat = m*x[i] + b      # add [i] (indexes) to be able to iterate through all the values in column 1 in the data
    y_hat = m*x[i] + b      # note: y_hat IS the prediction value whereas just y is the actual supplied data
    -------------------------------------------------------------------------------------------------------------------
    """

    # [Initializing Variables]
    m = x.shape[0]                  # m = number of training examples
    total_cost_function_value = 0   # f(w,b) will be denoted as total_cost_function_value

    for i in range(m):
        y_hat = m*x[i]+b
        difference_squared_valued = (y_hat - y[i])**2
        total_cost_function_value += difference_squared_valued
    return total_cost_function_value
#----------------------------------------------------------------------------------------------------------------------



