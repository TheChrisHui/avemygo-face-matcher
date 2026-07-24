import numpy as np

def normalize_landmarks(points):
    """
    Takes a 28x2 array of facial landmarks and normalizes them 
    for translation (position) and scale (size).
    """
    # Convert to float arrays so our division doesn't get rounded to zero
    points = np.array(points, dtype=np.float32)
    
    # 1. Translation: Find the center and teleport the face to (0,0)
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # 2. Scaling: Calculate the overall "size" of the shape and divide by it
    scale = np.linalg.norm(centered_points)
    
    if scale > 0:
        normalized_points = centered_points / scale
    else:
        normalized_points = centered_points
        
    # 3. Flatten the matrix from 28x2 down to a single 1D vector of 56 numbers.
    # This makes calculating the difference between faces ridiculously fast.
    return normalized_points.flatten()
