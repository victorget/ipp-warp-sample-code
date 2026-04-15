#include <ipp.h>
#include <stdio.h>
#include "batch_warp_core.h"

#define IPP_SAFE_CALL_LOOP(pFunc, finalStatus, ...) \
    status = pFunc(__VA_ARGS__); \
    if (status != ippStsNoErr) { \
        _Pragma("omp critical") \
        finalStatus = status; \
        continue; }

#define IPP_SAFE_CALL(pFunc, ...) \
    status = pFunc(__VA_ARGS__); \
    if (status != ippStsNoErr) { return status; }

// Helper function to calculate buffer sizes for warp perspective
static IppStatus GetPerspectiveBufferSizes(
    IppiSize roiSize,
    IppiRect roi,
    IppDataType dataType,
    float* srcPtsArray,
    double borderValue,
    int* specSize,
    int* bufSize
) {
    IppStatus status = ippStsNoErr;
    Ipp64f H[3][3] = {{1,0,0},{0,1,0},{0,0,1}};

    const double quad[4][2] = {
        { srcPtsArray[0], srcPtsArray[1] },
        { srcPtsArray[2], srcPtsArray[3] },
        { srcPtsArray[4], srcPtsArray[5] },
        { srcPtsArray[6], srcPtsArray[7] }
    };

    IPP_SAFE_CALL(ippiGetPerspectiveTransform, roi, quad, H);

    // Get required spec and buffer size (same for all images if the ROI is the same for entire batch (uniform batch))
    IPP_SAFE_CALL(ippiWarpPerspectiveGetSize,
        roiSize, roi, roiSize, dataType, H,
        ippLinear, ippWarpBackward, ippBorderConst,
        specSize, bufSize);

    IppiWarpSpec* pSpec = (IppiWarpSpec*)ippMalloc_L(*specSize);
    if (pSpec == NULL) {
        return ippStsMemAllocErr;
    }

    // Initialize warp structure for this transform
    IPP_SAFE_CALL(ippiWarpPerspectiveLinearInit,
        roiSize, roi, roiSize, dataType, H,
        ippWarpBackward, 1, ippBorderConst, &borderValue, 0, pSpec);

    IPP_SAFE_CALL(ippiWarpGetBufferSize, pSpec, roiSize, bufSize);
    ippFree(pSpec);

    return ippStsNoErr;
}

// Helper function to calculate buffer sizes for warp affine
static IppStatus GetAffineBufferSizes(
    IppiSize srcSize,
    IppiSize dstSize,
    IppiRect roi,
    IppDataType dataType,
    float* srcPtsArray,
    double borderValue,
    int* specSize,
    int* bufSize
) {
    IppStatus status = ippStsNoErr;
    double coeffs[2][3] = {{1,0,0},{0,1,0}};

    double quad[4][2] = {
        { srcPtsArray[0], srcPtsArray[1] },
        { srcPtsArray[2], srcPtsArray[3] },
        { srcPtsArray[4], srcPtsArray[5] },
        { srcPtsArray[6], srcPtsArray[7] }
    };
    quad[3][0] = quad[0][0] + (quad[2][0] - quad[1][0]);
    quad[3][1] = quad[0][1] + (quad[2][1] - quad[1][1]);

    IPP_SAFE_CALL(ippiGetAffineTransform, roi, quad, coeffs);

    // Get required spec and buffer size
    IPP_SAFE_CALL(ippiWarpAffineGetSize,
        srcSize, dstSize, dataType, coeffs,
        ippLinear, ippWarpBackward, ippBorderConst, specSize, bufSize);

    IppiWarpSpec* pSpec = (IppiWarpSpec*)ippMalloc_L(*specSize);
    if (pSpec == NULL) {
        return ippStsMemAllocErr;
    }

    // Initialize warp structure for this transform
    IPP_SAFE_CALL(ippiWarpAffineLinearInit,
        srcSize, dstSize, dataType, coeffs,
        ippWarpBackward, 1, ippBorderConst, &borderValue, 0, pSpec);

    IPP_SAFE_CALL(ippiWarpGetBufferSize, pSpec, dstSize, bufSize);
    ippFree(pSpec);

    return ippStsNoErr;
}

// Batch WarpPerspective C implementation using plain arrays
// This is the core function that will be called from Cython
int BatchWarpPerspective(
    void* pSrcArray,
    void* pDstArray,
    float* srcPtsArray,
    int N,
    int width,
    int height,
    int numThreads,
    int isFloat
) {
    if(pSrcArray == NULL || pDstArray == NULL || srcPtsArray == NULL) {
        return ippStsNullPtrErr;
    }
    if(numThreads <= 0 || N <= 1 || width <= 1 || height <= 1) {
        return ippStsBadArgErr;
    }
    IppStatus status = ippStsNoErr;
    IppStatus finalStatus = ippStsNoErr;
    IppDataType dataType = isFloat ? ipp32f : ipp8u;
    int srcStep = isFloat ? width * sizeof(Ipp32f) : width * sizeof(Ipp8u);
    int dstStep = srcStep;

    // Get buffer sizes once (same for all images if the ROI is the same for entire batch)
    int bufSize = 0, specSize = 0;
    IppiPoint roiOffset = {0, 0};
    IppiSize roiSize = {width, height};
    IppiRect roi = {0, 0, width, height};
    double borderValue = 128.0; // border value

    // Calculate buffer sizes using helper function
    IPP_SAFE_CALL(GetPerspectiveBufferSizes, roiSize, roi, dataType, srcPtsArray, borderValue, &specSize, &bufSize);

    #pragma omp parallel num_threads(numThreads) shared(finalStatus)
    {
        // Thread-local buffers
        IppiWarpSpec* pSpec = (IppiWarpSpec*)ippMalloc_L(specSize);
        Ipp8u* pBuf = (Ipp8u*)ippMalloc_L(bufSize);

        if (pSpec == NULL || pBuf == NULL) {
            #pragma omp critical
            finalStatus = ippStsMemAllocErr;
        }

        #pragma omp for
        for (int i = 0; i < N; i++) {
            if (finalStatus != ippStsNoErr) continue;

            // Get pointers to the points for this image (8 floats = 4 points x 2 coords)
            float *srcPts = (float*)(&srcPtsArray[i * 8]);
            const double quad[4][2] = {
                { srcPts[0], srcPts[1] },
                { srcPts[2], srcPts[3] },
                { srcPts[4], srcPts[5] },
                { srcPts[6], srcPts[7] }
            };

            Ipp64f H[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
            IPP_SAFE_CALL_LOOP(ippiGetPerspectiveTransform, finalStatus, roi, quad, H);

            // Initialize warp structure for this transform
            IPP_SAFE_CALL_LOOP(ippiWarpPerspectiveLinearInit, finalStatus,
                roiSize, roi, roiSize, dataType, H,
                ippWarpBackward, 1, ippBorderConst, &borderValue, 0, pSpec);

            // Calculate offset to current image in the plain array
            size_t imageOffset = (size_t)i * width * height;

            // Perform the warp transformation
            if (isFloat) {
                IPP_SAFE_CALL_LOOP(ippiWarpPerspectiveLinear_32f_C1R, finalStatus,
                    ((Ipp32f*)pSrcArray) + imageOffset, srcStep,
                    ((Ipp32f*)pDstArray) + imageOffset, dstStep,
                    roiOffset, roiSize,
                    pSpec, pBuf
                );
            } else {
                IPP_SAFE_CALL_LOOP(ippiWarpPerspectiveLinear_8u_C1R, finalStatus,
                    ((Ipp8u*)pSrcArray) + imageOffset, srcStep,
                    ((Ipp8u*)pDstArray) + imageOffset, dstStep,
                    roiOffset, roiSize,
                    pSpec, pBuf
                );
            }
        }

        // Free thread-local buffers
        if (pBuf != NULL) {
            ippFree(pBuf);
        }
        if (pSpec != NULL) {
            ippFree(pSpec);
        }
    }

    return finalStatus;
}

// Batch WarpAffine C implementation using plain arrays
// This is the core function that will be called from Cython
int BatchWarpAffine(
    void* pSrcArray,
    void* pDstArray,
    float* srcPtsArray,
    int N,
    int width,
    int height,
    int numThreads,
    int isFloat
) {
    if(pSrcArray == NULL || pDstArray == NULL || srcPtsArray == NULL) {
        return ippStsNullPtrErr;
    }

    IppStatus status = ippStsNoErr;
    IppStatus finalStatus = ippStsNoErr;
    IppDataType dataType = isFloat ? ipp32f : ipp8u;
    int srcStep = isFloat ? width * sizeof(Ipp32f) : width * sizeof(Ipp8u);
    int dstStep = srcStep;

    // Get buffer sizes once (same for all images if the ROI is the same for entire batch)
    int bufSize = 0, specSize = 0;
    IppiPoint roiOffset = {0, 0};
    IppiSize roiSize = {width, height};
    IppiRect roi = {0, 0, width, height};
    double borderValue = 128.0; // border value

    // Calculate buffer sizes using helper function
    IPP_SAFE_CALL(GetAffineBufferSizes, roiSize, roiSize, roi, dataType, srcPtsArray, borderValue, &specSize, &bufSize);

    // Process each image in the batch - each thread gets its own buffers
    #pragma omp parallel num_threads(numThreads) shared(finalStatus)
    {
        // Thread-local buffers
        IppiWarpSpec* pSpec = (IppiWarpSpec*)ippMalloc_L(specSize);
        Ipp8u* pBuf = (Ipp8u*)ippMalloc_L(bufSize);

        if (pSpec == NULL || pBuf == NULL) {
            #pragma omp critical
            finalStatus = ippStsMemAllocErr;
        }

        #pragma omp for
        for (int i = 0; i < N; i++) {
            if (finalStatus != ippStsNoErr) continue;

            // Get pointers to the points for this image (use first 3 of 4 points = 6 floats)
            float *srcPts = (float*)(&srcPtsArray[i * 8]);
            double quad[4][2] = {
                { srcPts[0], srcPts[1] },
                { srcPts[2], srcPts[3] },
                { srcPts[4], srcPts[5] },
                { srcPts[6], srcPts[7] }
            };
            quad[3][0] = quad[0][0] + (quad[2][0] - quad[1][0]);
            quad[3][1] = quad[0][1] + (quad[2][1] - quad[1][1]);

            double coeffs[2][3];
            IPP_SAFE_CALL_LOOP(ippiGetAffineTransform, finalStatus, roi, quad, coeffs);
            // Initialize warp structure for this transform
            IPP_SAFE_CALL_LOOP(ippiWarpAffineLinearInit, finalStatus,
                roiSize, roiSize, dataType, coeffs,
                ippWarpBackward, 1, ippBorderConst, &borderValue, 0, pSpec);

            // Calculate offset to current image in the plain array
            size_t imageOffset = (size_t)i * width * height;

            // Perform the warp transformation
            if (isFloat) {
                IPP_SAFE_CALL_LOOP(ippiWarpAffineLinear_32f_C1R, finalStatus,
                    ((Ipp32f*)pSrcArray) + imageOffset, srcStep,
                    ((Ipp32f*)pDstArray) + imageOffset, dstStep,
                    roiOffset, roiSize,
                    pSpec, pBuf
                );
            } else {
                IPP_SAFE_CALL_LOOP(ippiWarpAffineLinear_8u_C1R, finalStatus,
                    ((Ipp8u*)pSrcArray) + imageOffset, srcStep,
                    ((Ipp8u*)pDstArray) + imageOffset, dstStep,
                    roiOffset, roiSize,
                    pSpec, pBuf
                );
            }
        }

        // Free thread-local buffers
        if (pBuf != NULL) {
            ippFree(pBuf);
        }
        if (pSpec != NULL) {
            ippFree(pSpec);
        }
    }

    return finalStatus;
}


