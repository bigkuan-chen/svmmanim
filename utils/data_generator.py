import numpy as np

def generate_ring_dataset(
    n_inner=35,
    n_outer=45,
    inner_radius_range=(0.0, 1.0),
    outer_radius_range=(1.6, 2.5),
    noise=0.0,
    random_seed=7
):
    """
    Generates a 2D circular dataset consisting of an inner cluster (class 0)
    and an outer ring (class 1).
    
    Parameters:
        n_inner (int): Number of points in the inner class.
        n_outer (int): Number of points in the outer class.
        inner_radius_range (tuple): (min_radius, max_radius) for the inner class.
        outer_radius_range (tuple): (min_radius, max_radius) for the outer class.
        noise (float): Standard deviation of Gaussian noise added to the coordinates.
        random_seed (int): Seed for the random number generator to ensure reproducibility.
        
    Returns:
        X (np.ndarray): Coordinate array of shape (n_inner + n_outer, 2).
        y (np.ndarray): Label array of shape (n_inner + n_outer,) containing 0 for inner and 1 for outer.
    """
    rng = np.random.RandomState(random_seed)
    
    # Generate inner class (class 0)
    r_inner = rng.uniform(inner_radius_range[0], inner_radius_range[1], n_inner)
    theta_inner = rng.uniform(0, 2 * np.pi, n_inner)
    X_inner = np.stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)], axis=1)
    
    # Generate outer class (class 1)
    r_outer = rng.uniform(outer_radius_range[0], outer_radius_range[1], n_outer)
    theta_outer = rng.uniform(0, 2 * np.pi, n_outer)
    X_outer = np.stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)], axis=1)
    
    # Combine classes
    X = np.vstack([X_inner, X_outer])
    y = np.concatenate([np.zeros(n_inner, dtype=int), np.ones(n_outer, dtype=int)])
    
    # Add noise if specified
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
        
    return X, y
