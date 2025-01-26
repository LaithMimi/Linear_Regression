import numpy as np

def compute_linear_regression(X, y, alpha=0.01, max_iterations=1000, max_features_to_discard=1):
    """main linear regression function with feature selection"""
    # handling feature selection before adding bias
    if max_features_to_discard > 0:
        keep_mask = discard_features(X, y, max_features_to_discard)
        X = X[:, keep_mask]
    
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



def discard_features(X, y, max_discard):
    """select features based on correlation with target"""
    #to calculate the absolute correlations for each feature
    corrs = [ abs(np.corrcoef(X[:,i], y.ravel())[0,1]) for i in range(X.shape[1]) ]
    
    #to determine how many features to keep
    n_features = X.shape[1]
    n_keep = max(1, n_features - max_discard)  # keep at least 1
    
    #to get indices of top correlated features
    ranked = np.argsort(corrs)[::-1][:n_keep]
    
    #to create boolean mask for kept features
    mask = np.zeros(n_features, dtype=bool)
    mask[ranked] = True
    return mask





if __name__ == "__main__":
    # test case - simple linear relationship
    np.random.seed(42)
    X_test = np.random.rand(100, 2)
    y_test = 3*X_test[:,0] + 2*X_test[:,1] + np.random.randn(100)*0.5
    
    # run with 1 feature discarded
    weights, cost = compute_linear_regression(X_test, y_test, 
                                            max_features_to_discard=1)
    print(f"\ntest weights: {weights}")
    print(f"final cost: {cost:.4f}")