import yaml

def read_yaml_config(path: str) -> dict:
    """
    Reads a YAML config file and returns its content as a dictionary.
    
    Args:
        path (str): Path to the YAML config file.
        
    Returns:
        dict: Parsed YAML config.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        yaml.YAMLError: If there's an error parsing the YAML.
    """
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Config file not found at: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"❌ Failed to parse YAML config: {e}")
