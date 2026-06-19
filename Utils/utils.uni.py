import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from matplotlib.color import LinearSegmentedColormap
from ipywidgets import interact
from utils.utils_common import compute_cost, compute_gradient_matrix


###################################
#   Plotting routines
###################################

def plt_house(X, y, f_wb=None, ax= None):
    """
    Plots the house prices and the linear regression line.

    Args:
        X (ndarray): Input features, shape (m, n).
        y (ndarray): Target values, shape (m,).
        f_wb (ndarray): Predicted values from the linear regression, shape (m,).
        ax (matplotlib.axes.Axes): The axes to plot on. If None, a new figure and axes will be created.
    Returns:
        None
    """
    if not ax:
        fig, ax = plt.subplots(1,1)
        ax.scatter(X, y, marker='x', c='red', label='Actual Prices') # Scatter plot of actual prices

        ax.set_title('House Prices')
        ax.set_xlabel('Size (sq ft)')
        ax.set_ylabel('Price (in 1000d f dollars)')
        if f_wb is not None:
            ax.plot(X, f_wb, c='blue', label='Predicted Prices') # Line plot of predicted prices
        ax.legend()

def mk_cost_lines(X, y, w, b, ax):
    """ Plots the cost function lines for linear regression.
    Args:
        X (ndarray): Input features, shape (m, n).
        y (ndarray): Target values, shape (m,).
        w (ndarray): Weights, shape (n,).
        b (float): Bias term.
        ax (matplotlib.axes.Axes): The axes to plot on.
    Returns:
        None
    """

    cstr = "cost = (1 / m) *("
    ctot = 0   
    label = 'cost for point'
    addedbreak = False
    for p in zip(X,y): # Loop through each point (x, y) in the dataset
        f_wb_p = w*p[0] + b # Predicted value for the point
        c_p = ((f_wb_p - p[1])**2) / 2 # Cost for the point
        c_p_txt = c_p # Cost value for the point
        ax.vlines(p[0], 0, p[1], f_wb_p, lw=3, colors='purple', ls='dotted',label=label) # Vertical line from x-axis to the point
        label =''
        cxy = (p[0], p[1] + (f_wb_p - p[1]) / 2) # Position for the cost annotation
        ax.annotate(f'{c_p_txt:0.0f}', xy=cxy, xycoords='data', color = "purple", xytext=(5,0), textcoords='offset points')# Annotate the cost value
        cstr += f"{c_p_txt:0.0f} + "
        if len(cstr) > 40 and addedbreak is False:
            cstr += "\n"
            addedbreak = True
        ctot += c_p
    ctot = ctot / len(X) # Average cost for all points
    cstr = cstr[:-1] + f") = {ctot:0.0f}" # Final cost function string
    ax.text(0.15,0.02, cstr, transform = ax.transAxes, color= "purple")


##########################
#  Cost lab
##########################

def plt_intuition(x_train, y_train):
    """
    Plots the intuition behind linear regression cost function.

    Args:
        x_train (ndarray): Input features, shape (m, n).
        y_train (ndarray): Target values, shape (m,).
    Returns:
        None
    """
    w_range = np.array([200-200, 200+200]) # Range of weights to explore
    temp_b = 100 # Fixed bias term

    w_array = np.arange(*w_range, 5 ) # Array of weights to explore
    cost =np.zeros_like(w_array) 
    for i in range(len(w_array)):
        tmp_w = w_array[i]
        cost[i] = compute_cost(x_train, y_train, tmp_w, temp_b)

    fig, ax = plt.subplots(1, 2,  constrained_layout=True, figsize=(8,4))
    @interact(w=(*w_range,10), continuous_update=False)
    def func(w=150):
        ax[0].cla()
        ax[1].cla()
        f_wb = np.dot(x_train, w) + temp_b

        fig.canvas.toolbar_position = 'button'

        mk_cost_lines(x_train, y_train, w, temp_b, ax[0])
        plt_house(x_train, y_train, f_wb=f_wb, ax=ax[0])

        ax[1].plot(w_array, cost)
        cur_cost = compute_cost(x_train, y_train, w, temp_b)
        ax[1].scatter(w, cur_cost, s =100, color="darkred", zorder=10, label=f"cost at w={w}")
        ax[1].hlines(cur_cost, ax[1].get_xlim()[0], w, lw=4, color="purple", ls='dotted')
        ax[1].vlines(w, ax[1].get_ylim, cur_cost, lw=4, color="purple", ls='dotted')
        ax[1].set_title("cost vs w, (b fixed at 100)")
        ax[1].set_ylabel('cost')
        ax[1].set_xlabel('w')
        ax[1].legend(loc='upper center')
        fig.suptitle(f"minimize cost: Current Cost = {cur_cost:0.0f}", fontsize=12)
        plt.show()


