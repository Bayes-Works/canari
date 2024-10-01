import numpy as np

def block_diag(*arrays: np.ndarray) -> np.ndarray:
    """
    Create a block diagonal matrix from the provided arrays.

    Parameters:
        *arrays (np.ndarray): Variable number of 2D arrays to be placed along the diagonal.

    Returns:
        np.ndarray: A block diagonal matrix.
    """
    if not arrays:
        return np.array([[]])
    
    # Calculate the total size of the new matrix
    total_rows = sum(a.shape[0] for a in arrays)
    total_cols = sum(a.shape[1] for a in arrays)
    
    # Initialize the block diagonal matrix with zeros
    block_matrix = np.zeros((total_rows, total_cols), dtype=arrays[0].dtype)
    
    current_row = 0
    current_col = 0
    
    for a in arrays:
        rows, cols = a.shape
        block_matrix[current_row:current_row+rows, current_col:current_col+cols] = a
        current_row += rows
        current_col += cols
    
    return block_matrix