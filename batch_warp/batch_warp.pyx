# distutils: language = c
# cython: language_level=3

import numpy as np
cimport numpy as cnp
cnp.import_array()

# Forward declaration of the batch function (now uses plain arrays)
cdef extern:
    int BatchWarpPerspective(
        void* pSrcArray,
        void* pDstArray,
        float* srcPtsArray,
        int N,
        int width,
        int height,
        int numThreads,
        int isFloat
    )

# Forward declaration of the batch function (now uses plain arrays)
cdef extern:
    int BatchWarpAffine(
        void* pSrcArray,
        void* pDstArray,
        float* srcPtsArray,
        int N,
        int width,
        int height,
        int numThreads,
        int isFloat
    )

def batch_warp_transform(src_images, dst_images, src_points, affine=False, num_threads=1):
    """
    Perform batch warp transformation (perspective or affine) on multiple images using Intel IPP.
    Supports both uint8 and float32 image types.

    Parameters
    ----------
    src_images : numpy array
        3D array of N source images (shape: [N, height, width], dtype: uint8 or float32)
    dst_images : numpy array
        3D array of N pre-allocated destination images (shape: [N, height, width], same dtype as src_images)
    src_points : numpy array
        3D array of shape [N, 4, 2] with source points for each image (float32)
        For affine transforms, only the first 3 points are used
    affine : bool, optional
        If True, use affine transformation (3 points). If False, use perspective transformation (4 points). (default: False)
    num_threads : int, optional
        Number of threads to use for parallel processing (default: 1)

    Returns
    -------
    None
        Results are written directly to dst_images

    Examples
    --------
    >>> import batch_warp
    >>> import numpy as np
    >>>
    >>> # Create batch of images as 3D array (uint8)
    >>> src_images = np.random.randint(0, 256, (2, 1080, 1920), dtype=np.uint8)
    >>> dst_images = np.zeros((2, 1080, 1920), dtype=np.uint8)
    >>>
    >>> # Or use float32
    >>> # src_images = np.random.rand(2, 1080, 1920).astype(np.float32)
    >>> # dst_images = np.zeros((2, 1080, 1920), dtype=np.float32)
    >>>
    >>> # Define source points as [N, 4, 2] array
    >>> src_pts = np.array([
    >>>     [[200, 100], [1720, 120], [1700, 980], [220, 960]],  # Image 1
    >>>     [[210, 110], [1710, 130], [1690, 970], [230, 950]]   # Image 2
    >>> ], dtype=np.float32)
    >>>
    >>> # Perform batch perspective warp
    >>> batch_warp.batch_warp_transform(src_images, dst_images, src_pts, num_threads=4, affine=False)
    >>>
    >>> # Or perform batch affine warp (uses only first 3 points)
    >>> batch_warp.batch_warp_transform(src_images, dst_images, src_pts, num_threads=4, affine=True)
    """
    # Validate inputs
    if not isinstance(src_images, np.ndarray) or src_images.ndim != 3:
        raise ValueError("src_images must be a 3D numpy array (N, height, width)")
    if not isinstance(dst_images, np.ndarray) or dst_images.ndim != 3:
        raise ValueError("dst_images must be a 3D numpy array (N, height, width)")

    cdef int N = src_images.shape[0]
    cdef int height = src_images.shape[1]
    cdef int width = src_images.shape[2]

    if dst_images.shape[0] != N:
        raise ValueError("Number of source images and destination images must match")

    if dst_images.shape[1] != height or dst_images.shape[2] != width:
        raise ValueError(f"dst_images shape {dst_images.shape} doesn't match src_images shape {src_images.shape}")

    if not isinstance(src_points, np.ndarray) or src_points.dtype != np.float32:
        raise ValueError("src_points must be a numpy array of dtype float32")

    if src_points.ndim != 3 or src_points.shape[0] != N or src_points.shape[1] != 4 or src_points.shape[2] != 2:
        raise ValueError(f"src_points must have shape [N, 4, 2] = [{N}, 4, 2], got {src_points.shape}")

    # Determine data type
    if N == 0:
        raise ValueError("Image array cannot be empty")

    cdef bint is_float32 = (src_images.dtype == np.float32)
    cdef bint is_uint8 = (src_images.dtype == np.uint8)

    if not is_float32 and not is_uint8:
        raise ValueError(f"Images must be float32 or uint8, got {src_images.dtype}")

    if src_images.dtype != dst_images.dtype:
        raise ValueError("src_images and dst_images must have the same dtype")

    cdef int isFloat = 1 if is_float32 else 0
    cdef int status

    # Check if arrays are already contiguous, if not raise error to avoid copy
    if not src_images.flags['C_CONTIGUOUS']:
        raise ValueError("src_images must be C-contiguous. Use np.ascontiguousarray() before calling this function.")
    if not dst_images.flags['C_CONTIGUOUS']:
        raise ValueError("dst_images must be C-contiguous. Use np.ascontiguousarray() before calling this function.")
    if not src_points.flags['C_CONTIGUOUS']:
        raise ValueError("src_points must be C-contiguous. Use np.ascontiguousarray() before calling this function.")

    # Get pointers directly without copying
    cdef float* pSrcPtsArray = <float*>cnp.PyArray_DATA(src_points)
    cdef void* pSrcData = <void*>cnp.PyArray_DATA(src_images)
    cdef void* pDstData = <void*>cnp.PyArray_DATA(dst_images)

    # Call the appropriate batch warp function based on affine flag
    if affine:
        status = BatchWarpAffine(
            pSrcData, pDstData,
            pSrcPtsArray,
            N, width, height, num_threads, isFloat
        )
    else:
        status = BatchWarpPerspective(
            pSrcData, pDstData,
            pSrcPtsArray,
            N, width, height, num_threads, isFloat
        )

    if status != 0:
        transform_type = "affine" if affine else "perspective"
        raise RuntimeError(f"Batch warp {transform_type} failed with status: {status}")
