import cv2
import numpy as np
import torch
from math import hypot  # for Euclidean distance
import multiprocessing as mp
from tqdm import tqdm

def process_rows(start: int, end: int, binary: np.ndarray) -> np.ndarray:
    """
    DEPRECATED: Processes a subset of rows in a binary image to find the nearest non-zero pixel for each zero pixel.

    Args:
        start (int): The starting row index (inclusive).
        end (int): The ending row index (exclusive).
        binary (np.ndarray): A 2D numpy array representing the binary image.

    Returns:
        np.ndarray: A 3D numpy array of shape (end - start, width, 2) where each element contains the coordinates 
                    of the nearest non-zero pixel for each zero pixel in the specified row range. If the pixel is 
                    non-zero, it retains its own coordinates.
    """
    h, w = binary.shape
    result = np.zeros((end - start, w, 2), dtype=int)
    pid = mp.current_process().pid
    print(f"Process {pid} is processing rows {start} to {end}")
    for i in tqdm(range(start, end)):
        for j in range(w):
            if binary[i, j] == 0:
                min_dist = float('inf')
                nearest_coord = (i, j)
                for x in range(h):
                    for y in range(w):
                        if binary[x, y] != 0:
                            d = hypot(i - x, j - y)
                            if d < min_dist:
                                min_dist = d
                                nearest_coord = (x, y)
                result[i - start, j] = nearest_coord
            else:
                result[i - start, j] = (i, j)
    return result

def process_rows2(start: int, end: int, binary: np.ndarray, print_progress: bool = False) -> np.ndarray:
    """
    Processes rows of a binary matrix to find the nearest non-zero element for each zero element.
    Args:
        start (int): The starting row index (inclusive) for processing.
        end (int): The ending row index (exclusive) for processing.
        binary (np.ndarray): A 2D numpy array representing the binary matrix.
        print_progress (bool, optional): If True, prints the progress of the processing. Defaults to False.
    Returns:
        np.ndarray: A 3D numpy array where each element at position (i, j) contains the coordinates of the nearest 
                    non-zero element in the binary matrix for the corresponding zero element, or the coordinates of 
                    the element itself if it is non-zero.
    """
    occupied = []
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            if binary[i, j] != 0:
                occupied.append((i, j))
                
    r = tqdm(range(start, end)) if print_progress else range(start, end)

    h, w = binary.shape
    result = np.zeros((end - start, w, 2), dtype=int)
    pid = mp.current_process().pid
    if(print_progress): print(f"Process {pid} is processing rows {start} to {end}")
    for i in r:
        for j in range(binary.shape[1]):
            if binary[i, j] == 0:
                min_dist = float('inf')
                nearest_coord = (i, j)
                for x, y in occupied:
                    d = hypot(i - x, j - y)
                    if d < min_dist:
                        min_dist = d
                        nearest_coord = (x, y)
                result[i - start, j] = nearest_coord
            else:
                result[i - start, j] = (i, j)
    return result


def get_nearest_coords(image_path: str, fun = process_rows2) -> np.ndarray:
    """
    Processes an image to find the nearest coordinates using parallel processing.
    Args:
        image_path (str): The file path to the image to be processed.
        fun (callable, optional): The function to process rows of the image. Defaults to process_rows2.
    Returns:
        np.ndarray: An array of the nearest coordinates.
    Raises:
        ValueError: If the provided image path is not valid.
    """
    if not image_path:
        raise ValueError("Please provide a valid image path.")
    
    print(f"Processing {image_path}")
    
    path = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(path, 1, 255, cv2.THRESH_BINARY)
    
    h, w = binary.shape
    num_processes = mp.cpu_count()
    chunk_size = h // num_processes
    
    tasks = []
    for p in range(num_processes):
        start = p * chunk_size
        # Make sure the last chunk includes any remaining rows
        end = h if p == num_processes - 1 else (p + 1) * chunk_size
        tasks.append((start, end, binary))

    # Use a Pool to process rows in parallel
    with mp.Pool(num_processes) as pool:
        results = pool.starmap(fun, tasks)

    # Concatenate the results from each process
    nearest_coords = np.concatenate(results, axis=0)
    
    return nearest_coords



def coords_to_coldmap(coords, threshold: float = 20, exponent: float = 1.25, normalize: int = 1):
    """
    Transforms a coordinate array into a coldmap based on Euclidean distances and a threshold.
    Args:
        coords (numpy.ndarray): A 2D array of coordinates.
        threshold (float, optional): The distance threshold above which the transformation is applied. Default is 10.
        exponent (float, optional): The exponent used in the transformation for distances above the threshold. Default is 0.75.
        normalize (int, optional): The normalization factor for the transformed values. Default is 1.
    Returns:
        numpy.ndarray: The transformed and normalized coldmap as a 2D array.
    """
    rows, cols = coords.shape[0], coords.shape[1]
    
    # Compute the Euclidean distances.
    distances = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distances[i, j] = hypot(i - coords[i, j][0], j - coords[i, j][1])
    
    # Initialize the transformed array with the original distances.
    transformed = distances.copy()
    
    # Create a mask for pixels with distance > threshold.
    mask = distances > threshold
    # Only update those pixels: apply the transformation for distances above the threshold.
    transformed[mask] = threshold + (distances[mask] - threshold) ** exponent
    
    # Normalize the transformed values to the 0–normalize range.
    transformed_normalized = normalize * (transformed - transformed.min()) / (transformed.max() - transformed.min())
    
    return_type = np.uint8 if normalize == 255 else np.float32
    
    print("Coldmap generated.")
    
    return transformed_normalized.astype(return_type)


def save_coldmap_png(coldmap: np.ndarray, output_path: str):
    cv2.imwrite(output_path, coldmap)
    print(f"Coldmap saved to {output_path}")
    
def save_coldmap_torch(coldmap: np.ndarray, output_path: str):
    torch.save(coldmap, output_path)
    print(f"Coldmap saved to {output_path}")
    
def save_coldmap_npy(coldmap: np.ndarray, output_path: str):
    np.save(f"{output_path}.npy", coldmap)
    print(f"Coldmap saved to {output_path}.npy")
    
    
    
    
    
    
def main():
    input_path = "dataset/intersection_001/paths/path_3/path_line.png"
    output_path = "dataset/intersection_001/paths/path_3/cold_map.png"
    coords = get_nearest_coords(input_path)
    coldmap = coords_to_coldmap(coords, threshold=20, exponent=0.5)
    save_coldmap_png(coldmap, output_path)
    
if __name__ == "__main__":
    import time
    start = time.time()
    main()
    end = time.time()
    minutes, seconds = divmod(end - start, 60)
    print(f"Execution time: {minutes:.0f} minutes {seconds:.2f} seconds")