import re
import matplotlib.colors as mcolors

def name_to_hex(color_name):
    
    try:
        hex_color = mcolors.to_hex(color_name)
    except:
        hex_color = None
    
    return hex_color

def parse_rgba(s):
    # Find all numbers (including floating points)
    numbers = re.findall(r'[\d.]+', s)

    # Convert the strings to appropriate number types (int or float)
    parsed_numbers = [int(num) if '.' not in num else float(num) for num in numbers]

    return parsed_numbers

def hex_to_rgba(hex_color, alpha=1):
    """
    Convert hex to rgba.

    :param hex_color: str, hex color string (e.g. '#RRGGBB' or '#RRGGBBAA')
    :param alpha: float, alpha value ranging from 0 (transparent) to 1 (opaque)
    :return: str, a string representation of rgba color 'rgba(R, G, B, A)'
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 8:  # If alpha is provided in hex
        r, g, b, a = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), int(hex_color[6:8], 16)/255.0
    elif len(hex_color) == 6:  # If alpha is not provided in hex
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        a = alpha  # Use provided alpha
    else:
        raise ValueError('Input hex color is not in #RRGGBB or #RRGGBBAA format')
    return f'rgba({r}, {g}, {b}, {a})'

def rgba_to_hex(r, g, b, a):
    """
    Convert RGBA values to their 8-character hexadecimal representation.

    Parameters:
    - r, g, b: Red, Green, and Blue components (integers from 0 to 255).
    - a: Alpha/opacity component (float from 0.0 to 1.0).

    Returns:
    - The 8-character hex representation of the RGBA values (string).
    """
    hex_alpha = int(a * 255)
    extended_hex_color = "#{:02x}{:02x}{:02x}{:02x}".format(r, g, b, hex_alpha)
    return extended_hex_color

