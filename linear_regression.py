import numpy as np
import matplotlib.pyplot as plt

# x_train is the input variable or feature(s)
x_train = np.array([1.0, 2.0])
# y_train is the target or label
y_train = np.array([300.0 , 500.0])

print(f"x_tain  = {x_train}")
print(f"y_tain  = {y_train}")

# x = np.sum(x_train)
# y = np.sum(y_train)
# w = x/y
# print(w)

m = x_train.shape[0]
print("m : ", m)
print("m using len : ", len(x_train))
i = 0

x_i = x_train[i]
y_i = y_train[i]
print(f"The i-th values are : {x_i} , {y_i}")

w = 100
b = 100


#function to compute the model output and returns as a numpy array
def compute_model_output(x, w, b):
    m = len(x)
    f_wb = np.zeros(m)
    #f_wb = [0] * m

    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

#function to plot the training set. 
def plot_graph1():
    # Plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r')
    # Set the title
    # Set the y-axis label
    plt.ylabel('Price (in 1000s of dollars)')
    # Set the x-axis label
    plt.xlabel('Size (1000 sqft)')
    plt.show()

#plots the graph of the Training set and the model
def plot_graph(y):
    # Plot our model prediction
    plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

    # Plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
    # # Set the title
    plt.title("Housing Prices")
    # # Set the y-axis label
    plt.ylabel('Price (in 1000s of dollars)')
    # # Set the x-axis label
    plt.xlabel('Size (1000 sqft)')
    plt.legend()
    plt.show()


tmp_f_wb = compute_model_output(x_train, w, b)
plot_graph(tmp_f_wb)

tmp_f_wb = compute_model_output(x_train, w, b,)

#function to predict the cost of a house
def model_prediction(x_i):
    w = 200
    b = 100
    x_i = x_i/1000
    cost_1200sqft = w * x_i + b

    print(f"${cost_1200sqft:.0f} thousand dollars")

model_prediction(1.2)






