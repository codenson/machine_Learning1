# Housing Price Prediction Model

A simple linear regression model that predicts housing prices based on house size.

## Overview

This code implements a basic linear regression model with the form `f(x) = w*x + b` where:
- `x` is the house size in 1000 sqft
- `w` is the weight parameter
- `b` is the bias parameter
- Output is the predicted price in thousands of dollars

## Features

- Data visualization of training examples
- Model prediction functionality
- Plotting capabilities to compare predictions with actual values

## Functions

- `compute_model_output(x, w, b)`: Calculates model predictions for input values
- `plot_graph1()`: Visualizes the training data
- `plot_graph(y)`: Plots both training data and model predictions
- `model_prediction(x_i)`: Predicts house price for a given size input

## Usage Example

```python
# Predict price for a 1200 sqft house
model_prediction(1.2)  # Output: $340 thousand dollars
```

## Dependencies

- NumPy
- Matplotlib
