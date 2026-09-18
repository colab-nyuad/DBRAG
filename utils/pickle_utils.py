import pickle

def load_pickle_file(file_path):
    """
    Load a file from disk.
    
    Args:
        file_path (str): The path to the file.
    
    Returns:
        object: The loaded file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    

# function to save pickle file
def save_as_pickle(data, file_path):
    """
    Save a file to disk.
    
    Args:
        data (object): The data to save.
        file_path (str): The path to save the file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


