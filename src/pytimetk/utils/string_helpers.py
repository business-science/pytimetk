import re


_MODERN_TO_LEGACY_FREQ_UNITS = {
    "H": "H",
    "MIN": "min",
    "T": "T",
    "S": "S",
    "L": "L",
    "MS": "MS",
    "U": "U",
    "US": "U",
    "N": "N",
    "NS": "N",
    "D": "D",
    "W": "W",
    "M": "M",
    "ME": "M",
    "Q": "Q",
    "QE": "Q",
    "QS": "QS",
    "Y": "Y",
    "YE": "Y",
    "YS": "YS",
    "A": "A",
    "AS": "AS",
    "BM": "BM",
    "BME": "BM",
    "BMS": "BMS",
    "BQ": "BQ",
    "BQE": "BQ",
    "BQS": "BQS",
    "BY": "BY",
    "BYE": "BY",
    "BYS": "BYS",
    "BA": "BA",
    "BAS": "BAS",
    "CBM": "CBM",
    "CBME": "CBM",
    "SM": "SM",
    "SME": "SM",
}


def canonicalize_frequency_unit(unit: str) -> str:
    """Map modern pandas frequency spellings back to pytimetk's canonical units."""

    if not isinstance(unit, str):
        raise TypeError("`unit` must be a string.")

    token = unit.strip()
    if not token:
        raise ValueError("`unit` cannot be empty.")

    base, _, _suffix = token.partition("-")
    return _MODERN_TO_LEGACY_FREQ_UNITS.get(base.upper(), base)


def parse_freq_str(freq_str):
    match = re.match(r"(\d+)?([A-Za-z]+(?:-[A-Za-z]+)?)", freq_str)
    if not match:
        raise ValueError(f"Invalid frequency string: {freq_str}")

    quantity, unit = match.groups()

    if quantity is None:
        quantity = 1

    unit = canonicalize_frequency_unit(unit)

    return quantity, unit


def parse_freq_str2(freq_str):
    # Regular expression to match patterns like '2H', '30T', etc.
    matches = re.findall(r"(\d+)?([A-Za-z]+(?:-[A-Za-z]+)?)", freq_str)
    if not matches:
        raise ValueError(f"Invalid frequency string: {freq_str}")

    # Process each match to extract quantity and unit
    parsed = []
    for match in matches:
        quantity, unit = match
        if quantity == "":
            quantity = 1
        else:
            quantity = int(quantity)
        unit = canonicalize_frequency_unit(unit)
        parsed.append((quantity, unit))

    return parsed
