import numpy as np

def compute_linear_regression(X, y, alpha=0.01, max_iterations=1000):
    """main linear regression function with feature selection"""
    # adding bias column after feature selection
    X = np.c_[np.ones(X.shape[0]), X]
    
    #to initialize weights including bias term
    weights = np.zeros((X.shape[1], 1))
    
    # running gradient descent updates
    for i in range(max_iterations):
        grad = gradient(X, y, weights)
        weights -= alpha * grad
        
        # progress printing for result
        if i % 100 == 0:
            current_cost = compute_cost(X, y, weights)
            print(f'iteration {i}: cost {current_cost:.4f}')
    
    final_cost = compute_cost(X, y, weights)
    return weights.flatten(), final_cost

def compute_cost(X, y, weights):
    """calculate mean squared error"""
    h = X @ weights
    error = h - y.reshape(-1,1)
    return (error.T @ error).item() / (2 * len(y))



def gradient(X, y, weights):
    """calculate gradient of MSE cost function"""
    h = X @ weights
    error = h - y.reshape(-1,1)
    return (X.T @ error) / len(y)
