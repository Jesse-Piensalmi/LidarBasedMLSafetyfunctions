import pickle
import numpy as np

def pickle_to_numpy(input_pickle_file, output_numpy_dir):
    """
    Convert a pickled dictionary to numpy arrays and save each array as a .npy file.
    Args:
        input_pickle_file (str): Path to the input pickle file.
        output_numpy_dir (str): Directory where the numpy arrays will be saved.
    Returns:
        None
    """
    import os
    os.makedirs(output_numpy_dir, exist_ok=True)

    with open(input_pickle_file, 'rb') as file:
        data_dict = pickle.load(file)

    if not isinstance(data_dict, dict):
        raise ValueError("The pickled file does not contain a dictionary.")

    # Convert each value in the dictionary to a numpy array and save
    for key, value in data_dict.items():
        try:
            # Convert the value to a numpy array
            numpy_array = np.array(value)

            # Create a filename for the numpy array
            output_file = os.path.join(output_numpy_dir, f"{key}.npy")

            # Save the numpy array to a .npy file
            np.save(output_file, numpy_array)

            print(f"Saved {key} to {output_file}")
        except Exception as e:
            print(f"Could not convert or save {key}: {e}")

if __name__ == "__main__":
    input_pickle_file = "sim__data.pkl"  # Replace with pickle file name
    output_numpy_dir = "numpy_arrays"    # Replace with your output folder

    pickle_to_numpy(input_pickle_file, output_numpy_dir)