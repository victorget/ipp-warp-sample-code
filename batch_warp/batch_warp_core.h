#ifndef BATCH_WARP_CORE_H
#define BATCH_WARP_CORE_H

// Batch WarpPerspective function using plain arrays (not array of pointers)
// pSrcArray and pDstArray are plain contiguous arrays containing all images
int BatchWarpPerspective(
    void* pSrcArray,
    void* pDstArray,
    float* srcPtsArray,
    int N,
    int width,
    int height,
    int numThreads,
    int isFloat
);

// Batch WarpAffine function using plain arrays (not array of pointers)
// pSrcArray and pDstArray are plain contiguous arrays containing all images
// Uses first 3 points from each 4-point group in srcPtsArray
int BatchWarpAffine(
    void* pSrcArray,
    void* pDstArray,
    float* srcPtsArray,
    int N,
    int width,
    int height,
    int numThreads,
    int isFloat
);

#endif // BATCH_WARP_CORE_H
