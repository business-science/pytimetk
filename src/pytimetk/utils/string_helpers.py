import re

def parse_freq_str(freq_str):
    match = re.match(r'(\d+)?([A-Z]+|min)', freq_str)
    if not match:
        raise ValueError(f"Invalid frequency string: {freq_str}")
    
    quantity, unit = match.groups()
    
    if quantity is None:
        quantity = 1
    
    return quantity, unit


def parse_freq_str2(freq_str):
    # Regular expression to match patterns like '2H', '30T', etc.
    matches = re.findall(r'(\d+)?([A-Z]+|min)', freq_str)
    if not matches:
        raise ValueError(f"Invalid frequency string: {freq_str}")
    
    # Process each match to extract quantity and unit
    parsed = []
    for match in matches:
        quantity, unit = match
        if quantity == '':
            quantity = 1
        else:
            quantity = int(quantity)
        parsed.append((quantity, unit))
    
    return parsed
