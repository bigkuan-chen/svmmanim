import numpy as np
from sklearn.svm import SVC

def train_svm(X, y, kernel='rbf', C=10.0, gamma=1.0, degree=3):
    """
    Trains an SVM classifier on the provided dataset.
    
    Parameters:
        X (np.ndarray): Training data features of shape (n_samples, 2).
        y (np.ndarray): Training data labels of shape (n_samples,).
        kernel (str): Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid').
        C (float): Regularization parameter.
        gamma (float or str): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        degree (int): Degree of the polynomial kernel function ('poly').
        
    Returns:
        model (SVC): The trained scikit-learn SVC model.
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=42)
    model.fit(X, y)
    return model

def make_decision_grid(x_range=(-3.0, 3.0), y_range=(-3.0, 3.0), resolution=80):
    """
    Creates a coordinate grid for evaluating the decision function.
    
    Parameters:
        x_range (tuple): (x_min, x_max) coordinate range.
        y_range (tuple): (y_min, y_max) coordinate range.
        resolution (int): Number of grid points along each axis.
        
    Returns:
        xx (np.ndarray): Meshgrid X coordinates of shape (resolution, resolution).
        yy (np.ndarray): Meshgrid Y coordinates of shape (resolution, resolution).
        grid_points (np.ndarray): Reshaped coordinate pairs of shape (resolution * resolution, 2).
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid_points

def compute_decision_surface(model, grid_points, grid_shape=None):
    """
    Evaluates the decision function of the model on grid points and reshapes it.
    
    Parameters:
        model (SVC): The trained scikit-learn SVC model.
        grid_points (np.ndarray): Grid coordinate pairs of shape (N, 2).
        grid_shape (tuple): Optional target shape to reshape the decision values to.
        
    Returns:
        Z (np.ndarray): Decision function scores reshaped to match the grid.
    """
    Z = model.decision_function(grid_points)
    if grid_shape is not None:
        return Z.reshape(grid_shape)
    resolution = int(np.round(np.sqrt(len(grid_points))))
    return Z.reshape(resolution, resolution)
