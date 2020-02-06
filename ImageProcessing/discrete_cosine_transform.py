import numpy as np
import matplotlib.pyplot as plt
import math
def dct_dict(n_atoms, size):
    """
    Create a dictionary using the Discrete Cosine Transform (DCT) basis. If n_atoms is
    not a perfect square, the returned dictionary will have ceil(sqrt(n_atoms))**2 atoms
    :param n_atoms:
        Number of atoms in dict
    :param size:
        Size of first patch dim
    :return:
        DCT dictionary, shape (size*size, ceil(sqrt(n_atoms))**2)
    """
    # todo flip arguments to match random_dictionary
    p = int(math.ceil(math.sqrt(n_atoms)))
    dct = np.zeros((size, p))

    for k in range(p):
        basis = np.cos(np.arange(size) * k * math.pi / p)
        if k > 0:
            basis = basis - np.mean(basis)

        dct[:, k] = basis

    kron = np.kron(dct, dct)

    for col in range(kron.shape[1]):
        norm = np.linalg.norm(kron[:, col]) or 1
        kron[:, col] /= norm

    return kron

# print(dct_dict(50,64))
dct = dct_dict(64,128)
np.save('./dct.npy',dct)
