from typing import List, Tuple

import numpy as np


def center_of_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Get the center of each object.
    :param mask: np.ndarray
        A 3D NumPy array of shape (n, h, w) where each mask[i] is a binary
        mask of shape (h, w).
    :return:
        center of each object(mask)
    """
    centers = []
    num_tasks = mask.shape[0]

    for i in range(num_tasks):
        center_mask = mask[i, :, :]
        y_indices, x_indices = np.where(center_mask == 1)
        if len(y_indices) > 0:
            center_y = int(np.round(np.mean(y_indices)))
            center_x = int(np.round(np.mean(x_indices)))
            centers.append((center_y, center_x))
        else:
            centers.append((-1, -1))

    return centers


def remove_outliers_iqr(data: List) -> float:
    """
    Remove abnormal data from the sequence and get the average of the rest.
    :param data:
        data sequence (sampled from the specific pixel from the depth image)
    :return:
    """
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Calculate the bounds for normal data
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify normal data
    normal_data = data[(data >= lower_bound) & (data <= upper_bound)]

    # Return the average of the normal data
    return float(np.mean(normal_data))

