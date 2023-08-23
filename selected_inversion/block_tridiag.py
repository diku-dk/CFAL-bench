"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: https://doi.org/10.1016/j.jcp.2009.03.035
@reference: Numerical Recipes in C (2nd Ed.): The Art of Scientific Computing (10.5555/148286)

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
License: BSD 3-clause (https://opensource.org/licenses/BSD-3-Clause)
"""

import numpy as np


def generateRandomNumpyMat(matrice_size: int, 
                           is_complex: bool = False,
                           seed: int = None) -> np.ndarray:
    """ Generate a dense matrix of shape: (matrice_size x matrice_size) filled 
    with random numbers. The matrice may be complex or real valued.

    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
    is_complex : bool, optional
        Whether the matrice should be complex or real valued. The default is False.
    seed : int, optional
        Seed for the random number generator. The default is no seed.
        
    Returns
    -------
    A : np.ndarray
        The generated matrice.
    """
    if seed is not None:
        np.random.seed(seed)

    if is_complex:
        return np.random.rand(matrice_size, matrice_size)\
               + 1j * np.random.rand(matrice_size, matrice_size)
    else:
        return np.random.rand(matrice_size, matrice_size)


def generateBandedDiagonalMatrix(matrice_size: int,
                                 matrice_bandwidth: int, 
                                 is_complex: bool = False, 
                                 seed: int = None) -> np.ndarray:
    """ Generate a banded diagonal matrix of shape: (matrice_size x matrice_size)
    with a bandwidth of "bandwidth", filled with random numbers.

    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
    is_complex : bool, optional
        Whether the matrice should be complex or real valued. The default is False.
    seed : int, optional
        Seed for the random number generator. The default is no seed.
        
    Returns
    -------
    A : np.ndarray
        The generated matrice.
    """

    A = generateRandomNumpyMat(matrice_size, is_complex, seed)
    
    for i in range(matrice_size):
        for j in range(matrice_size):
            if i - j > matrice_bandwidth or j - i > matrice_bandwidth:
                A[i, j] = 0

    return A


def convertDenseToBlockTridiag(A: np.ndarray, 
                               blocksize: int):
    """ Converte a numpy dense matrix to 3 numpy arrays containing the diagonal,
    upper diagonal and lower diagonal blocks.
    
    Parameters
    ----------
    A : np.ndarray
        The matrix to convert.
    blocksize : int
        The size of the blocks.
        
    Returns
    -------
    A_bloc_diag : np.ndarray
        The diagonal blocks.
    A_bloc_upper : np.ndarray
        The upper diagonal blocks.
    A_bloc_lower : np.ndarray
        The lower diagonal blocks.
    """
    
    nblocks = int(np.ceil(A.shape[0]/blocksize))

    A_bloc_diag  = np.zeros((nblocks, blocksize, blocksize), dtype=A.dtype)
    A_bloc_upper = np.zeros((nblocks-1, blocksize, blocksize), dtype=A.dtype)
    A_bloc_lower = np.zeros((nblocks-1, blocksize, blocksize), dtype=A.dtype)

    for i in range(nblocks):
        A_bloc_diag[i, ] = A[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize]
        if i < nblocks-1:
            A_bloc_upper[i, ] = A[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize]
            A_bloc_lower[i, ] = A[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize]

    return A_bloc_diag, A_bloc_upper, A_bloc_lower


def convertBlockTridiagToDense(A_bloc_diag: np.ndarray, 
                               A_bloc_upper: np.ndarray, 
                               A_bloc_lower: np.ndarray):
    """ Convert a block tridiagonal matrix to a dense matrix.
    
    Parameters
    ----------
    A_bloc_diag : np.ndarray
        The diagonal blocks.
    A_bloc_upper : np.ndarray
        The upper diagonal blocks.
    A_bloc_lower : np.ndarray
        The lower diagonal blocks.
        
    Returns
    -------
    A : np.ndarray
        The dense matrix.
    """
    
    nblocks   = A_bloc_diag.shape[0]
    blocksize = A_bloc_diag.shape[1]

    A = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A_bloc_diag.dtype)

    for i in range(nblocks):
        A[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = A_bloc_diag[i, ]
        if i < nblocks-1:
            A[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize] = A_bloc_upper[i, ]
            A[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] = A_bloc_lower[i, ]

    return A


def block_tridiag_lusolve(A: np.ndarray, 
                            blocksize: int) -> np.ndarray:
    """ Block tridiagonal solver using non pivoting LU decomposition/solving.

    Parameters
    ----------
    A : np.ndarray
        Block tridiagonal matrix
    blocksize : int
        Block matrice_size
        
    Returns
    -------
    G : np.ndarray
        Inverse of A
    """

    matrice_size = A.shape[0]
    nblocks = matrice_size // blocksize

    G = np.zeros((matrice_size, matrice_size), dtype=A.dtype)
    L = np.zeros((matrice_size, matrice_size), dtype=A.dtype)
    U = np.zeros((matrice_size, matrice_size), dtype=A.dtype)
    D = np.zeros((matrice_size, matrice_size), dtype=A.dtype)
    
        
    # Initialisation of forward recurence
    D[0:blocksize, 0:blocksize] = A[0:blocksize, 0:blocksize]
    D_inv_0 = np.linalg.inv(D[0:blocksize, 0:blocksize])
    L[blocksize:2*blocksize, 0:blocksize] = A[blocksize:2*blocksize, 0:blocksize] @ D_inv_0
    U[0:blocksize, blocksize:2*blocksize] = D_inv_0 @ A[0:blocksize, blocksize:2*blocksize]
    
    # Forward recurence
    for i in range(1, nblocks):
        b_im1 = (i-1)*blocksize
        b_i   = i*blocksize
        b_ip1 = (i+1)*blocksize

        D_inv_i = np.linalg.inv(D[b_im1:b_i, b_im1:b_i])

        D[b_i:b_ip1, b_i:b_ip1] = A[b_i:b_ip1, b_i:b_ip1] - A[b_i:b_ip1, b_im1:b_i] @ D_inv_i @ A[b_im1:b_i, b_i:b_ip1]
        L[b_i:b_ip1, b_im1:b_i] = A[b_i:b_ip1, b_im1:b_i] @ D_inv_i
        U[b_im1:b_i, b_i:b_ip1] = D_inv_i @ A[b_im1:b_i, b_i:b_ip1]

    # Initialisation of backward recurence
    b_n   = (nblocks-1)*blocksize
    b_np1 = nblocks*blocksize

    G[b_n:b_np1, b_n:b_np1] = np.linalg.inv(D[b_n:b_np1, b_n:b_np1])
    
    # Backward recurence
    for i in range(nblocks-1, -1, -1):
        b_im1 = (i-1)*blocksize
        b_i   = i*blocksize
        b_ip1 = (i+1)*blocksize

        G[b_i:b_ip1, b_im1:b_i] = -G[b_i:b_ip1, b_i:b_ip1] @ L[b_i:b_ip1, b_im1:b_i]
        G[b_im1:b_i, b_i:b_ip1] = -U[b_im1:b_i, b_i:b_ip1] @ G[b_i:b_ip1, b_i:b_ip1]
        G[b_im1:b_i, b_im1:b_i] = np.linalg.inv(D[b_im1:b_i, b_im1:b_i]) + U[b_im1:b_i, b_i:b_ip1] @ G[b_i:b_ip1, b_i:b_ip1] @ L[b_i:b_ip1, b_im1:b_i]

    
    return G


if __name__ == "__main__":
    matrice_size = 128
    blocksize    = 32
    bandwidth    = np.ceil(blocksize/2)
    
    isComplex = True
    seed = 63
    A = generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = convertDenseToBlockTridiag(A_refsol, blocksize)
    
    A_block_tridiag_lusolve = block_tridiag_lusolve(A, blocksize)
    A_block_tridiag_lusolve_bloc_diag,\
        A_block_tridiag_lusolve_bloc_upper,\
        A_block_tridiag_lusolve_bloc_lower = convertDenseToBlockTridiag(A_block_tridiag_lusolve, blocksize)
        
    assert np.allclose(A_refsol_bloc_diag, A_block_tridiag_lusolve_bloc_diag)\
                and np.allclose(A_refsol_bloc_upper, A_block_tridiag_lusolve_bloc_upper)\
                and np.allclose(A_refsol_bloc_lower, A_block_tridiag_lusolve_bloc_lower)
