import pyterrier as pt
import os

def initialize_pyterrier():
    """Initializes PyTerrier."""
    if not pt.started():
        pt.init()

def load_index(index_path):
    """
    Loads an existing index if it exists at the specified path.
    
    Parameters:
    - index_path: str, the path where the index is stored.
    
    Returns:
    - index: PyTerrier Index object
    """
    # Initialize PyTerrier
    initialize_pyterrier()

    # Set the path to the index properties
    index_properties_path = os.path.join(index_path, "data.properties")
    
    # Check if index exists
    if os.path.exists(index_properties_path):
        # Load the existing index
        indexref = pt.IndexRef.of(index_properties_path)
        print("Index already exists. Loading existing index...")
        # Load and return the index
        index = pt.IndexFactory.of(indexref)
        return index
    else:
        raise FileNotFoundError(f"No index found at {index_properties_path}. Please index the dataset first.")

def get_collection_statistics(index):
    """Prints and returns collection statistics."""
    stats = index.getCollectionStatistics()
    print(stats.toString())
    return stats


