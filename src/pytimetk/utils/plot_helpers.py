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