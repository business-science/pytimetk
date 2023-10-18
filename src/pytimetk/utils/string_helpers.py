import re

def parse_freq_str(freq_str):
    match = re.match(r'(\d+)?([A-Z]+|min)', freq_str)
    if not match:
        raise ValueError(f"Invalid frequency string: {freq_str}")
    
    quantity, unit = match.groups()
    
    if quantity is None:
        quantity = 1
    
    return quantity, unit
