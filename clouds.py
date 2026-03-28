import numpy as np

def make_cube(h=1.0):
    coords = np.array([-h, 0, h])
    x, y, z = np.meshgrid(coords, coords, coords)
    # Combinamos y reestructuramos a una lista de puntos (27, 3)
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)    
    return pts

def make_sphere(n=200):
    golden = (1+np.sqrt(5))/2
    i = np.arange(n)
    theta = 2*np.pi*i/golden
    phi = np.arccos(1-2*(i+0.5)/n)
    return np.column_stack([np.sin(phi)*np.cos(theta),
                             np.sin(phi)*np.sin(theta),
                             np.cos(phi)])

def make_sklearn(n=300):
    from sklearn.datasets import make_swiss_roll
    X, _ = make_swiss_roll(n, noise=0.1, random_state=0)
    return ((X - X.min(0)) / (X.max(0) - X.min(0))) * 2 - 1