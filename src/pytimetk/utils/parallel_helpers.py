
from tqdm import tqdm

def conditional_tqdm(iterable, display=True, **kwargs):
    if display:
        return tqdm(iterable, **kwargs)
    else:
        return iterable