import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_linear_regression(X, y, alpha=0.01, max_iterations=1000, 
                             max_features_to_discard=1, plot=False):
    """main linear regression function with feature selection and plotting"""
    # store original features for plotting
    original_X = X.copy()
    
    # feature selection logic
    if max_features_to_discard > 0:
        keep_mask = discard_features(X, y, max_features_to_discard)
        X = X[:, keep_mask]
    else:
        keep_mask = np.ones(X.shape[1], dtype=bool)

    # add bias column after selection
    X_with_bias = np.c_[np.ones(X.shape[0]), X]
    
    # initialize weights
    weights = np.zeros((X_with_bias.shape[1], 1))
    
    # gradient descent loop
    for i in range(max_iterations):
        grad = gradient(X_with_bias, y, weights)
        weights -= alpha * grad

    # create plot if requested
    fig = None
    if plot:
        fig = plot_regression(original_X[:, keep_mask], y, weights)
        
    return weights.flatten(), compute_cost(X_with_bias, y, weights), keep_mask, fig

def plot_regression(X, y, weights):
    """visualize regression fit for 1D or 2D features"""
    plt.figure(figsize=(12, 5))
    n_features = X.shape[1]
    
    # 1D plot
    if n_features == 1:
        plt.scatter(X[:, 0], y, label='data points')
        x_vals = np.linspace(X.min(), X.max(), 100)
        y_vals = weights[0] + weights[1] * x_vals
        plt.plot(x_vals, y_vals, color='red', label='regression line')
        plt.xlabel('Feature 1')
        plt.ylabel('Target')
    
    # 2D plot
    elif n_features == 2:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], y, label='data points')
        
        # create prediction surface
        x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
        x2_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
        x1_mesh, x2_mesh = np.meshgrid(x1_vals, x2_vals)
        y_vals = (weights[0] + 
                 weights[1] * x1_mesh + 
                 weights[2] * x2_mesh)
        
        ax.plot_surface(x1_mesh, x2_mesh, y_vals, alpha=0.4, cmap='viridis')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target')
    
    else:
        print(f"can't plot {n_features}D data")
        return None
    
    plt.title(f'linear regression fit (weights: {np.round(weights, 2)})')
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

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
 # create sample data
    X = np.random.rand(100, 2)
    y = 2.5*X[:,0] + 1.8*X[:,1] + np.random.normal(0, 0.2, 100)

    # run with visualization
    weights, cost, mask, fig = compute_linear_regression(
        X, y,
        alpha=0.05,
        max_iterations=500,
        plot=True
    )

    # save visualization
    fig.savefig('training_progress.png')

    print("Kept features:", np.where(mask)[0])  # Shows which feature was kept
    plt.show()