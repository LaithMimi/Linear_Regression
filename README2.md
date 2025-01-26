# 📊 Visualized Linear Regression from Scratch

**see your model learn** • **interactive 3d plots** • **feature selection insights**

## 🌟 new features
- **live training visualization** (2D & 3D)
- **interactive prediction surfaces**
- **feature importance display**
- **convergence animation support**
- **exportable high-quality plots**

## 🎨 visualization gallery
### 1D regression (single feature)
```python
from visualized_linear_regression import compute_linear_regression

weights, cost, fig = compute_linear_regression(
    X, y,
    plot=True,
    plot_type='2d'
)
fig.savefig('regression_line.png')
```
![1d regression](https://matplotlib.org/stable/_images/sphx_glr_scatter_001.png)

### 2D regression (multiple features)
```python
weights, cost, fig = compute_linear_regression(
    X, y,
    plot=True,
    plot_type='3d'
)
fig.savefig('regression_plane.png') 
```
![3d regression](https://matplotlib.org/stable/_images/sphx_glr_scatter3d_001.png)

## 🚀 updated quick start
```bash
# install requirements
pip install numpy matplotlib
```

```python
import numpy as np
from visualized_linear_regression import compute_linear_regression

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
```

## 🔍 visualization features
- **real-time updates**: watch weights adjust during training
- **multiple view angles**: rotate 3D plots for better insight
- **cost overlay**: see error reduction alongside predictions
- **feature selection highlights**: visual indicators for kept/discarded features

## 📚 documentation highlights
```python
def compute_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.01,
    max_iterations: int = 1000,
    max_features_to_discard: int = 1,
    plot: bool = False,
    plot_interval: int = 100
) -> tuple:
    """
    returns: (weights, final_cost, feature_mask, figure)
    
    plot_interval: update visualization every N iterations
    """
```


made with ❤️ by Laith Mimi • 
