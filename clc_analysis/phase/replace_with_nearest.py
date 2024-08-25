import numpy as np


def replace_with_nearest(lst1, lst2):
    """
    replace lst1 elements with their nearest elements in lst2

    Parameters
    ----------
    lst1 : list or array
    lst2 : list or array

    Returns
    -------
    list or array of lst1 elements with their nearest counterparts in lst2
    """
    return lst2[[np.abs(lst2 - e).argmin() for e in lst1]]