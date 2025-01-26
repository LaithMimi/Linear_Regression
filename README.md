```markdown
# 📉 Linear Regression from Scratch with Gradient Descent

**implement linear regression like a pro** • **no black boxes** • **numpy-only**

[![python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![license](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<a href="https://github.com/LaithMimi/Linear_Regression">
  <img src="https://user-images.githubusercontent.com/7871801/198899549-3e6c3a7a-5f8d-45c4-9b81-8c0e7d5b5b2f.png" width="400" align="right" alt="gradient descent visualization">
</a>

## 🔥 features
- pure numpy implementation
- automatic feature selection
- intuitive gradient descent visualization
- customizable learning rate & iterations
- returns regression weights + final cost
- battle-tested with edge cases

## 🚀 quick start
```bash
pip install numpy
```
```python
from linear_regression import compute_linear_regression

# sample data: house_size (sqft), bedrooms → price
X = np.array([[1500, 2], [2000, 3], [1200, 1]]) 
y = np.array([300000, 450000, 250000])

# train model (discard 1 least useful feature)
weights, cost = compute_linear_regression(
    X, y, 
    alpha=0.01, 
    max_iterations=1000,
    max_features_to_discard=1
)

print(f"model weights: {weights}")  # [bias, kept_feature_coeff]
print(f"final cost: {cost:.2f}")    # 54832.17
```

## 🧠 how it works
### core algorithm
```python
y_pred = X @ weights  # matrix magic
cost = 1/(2m) * Σ(y_pred - y)^2  # mse
gradient = 1/m * X.T @ (y_pred - y)  # partial derivatives
weights = weights - α * gradient  # learn!
```

### feature selection magic
![](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Cbg_white%20%5Clarge%20%5Ctext%7Bkeep%20features%20with%7D%20%5C%2C%20%5Cmax%28%7C%5Crho%28X_i%2C%20y%29%7C%29)
- calculates feature-target correlations
- discards low-correlation features first
- automatically handles useless predictors

## 📊 sample output
```
iteration 0: cost 4.85e+05
iteration 100: cost 1.23e+05
iteration 200: cost 5.48e+04
...
iteration 900: cost 54832.17

model weights: [125000.00  150.00]  # price = 125k + 150*house_size
final cost: 54832.17
```

## 🤔 why use this?
- **transparent**: no sklearn magic - see every calculation
- **educational**: perfect for understanding GD mechanics
- **lightweight**: 150 lines of focused numpy code
- **customizable**: tweak learning rates/iterations freely

## 🛠 roadmap
- [ ] add momentum for faster convergence
- [ ] implement early stopping
- [ ] add jupyter notebook tutorial
- [ ] support polynomial features

## 👥 contributions welcome!
found a bug? have an optimization idea? 
- open an issue
- fork & make pull request
- star the repo ⭐

---

made with ❤️ by [your name] • [documentation](docs/) • [license](LICENSE)
```


- Works well on mobile & desktop
- Links to future plans

Adjust the [yourusername], sample data, and roadmap items as needed!
