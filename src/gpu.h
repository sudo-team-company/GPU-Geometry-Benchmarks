#ifndef GPU_IMPL
#define GPU_IMPL

#include "structs.h"

#include <CL/cl.h>
#include <cstdint>
#include <cuda_runtime.h>

void ThrowOnCudaError(cudaError_t err, char const *file, int line);
#define CHECK_CUDA_STATUS(x) ThrowOnCudaError(x, __FILE__, __LINE__)

void runIntersectAllPointsInBucketFullBucket(
  uint32_t blocks, uint32_t threads, const float2* vertices, const uint32_t* polygonOffsets,
  const uint32_t* bucketPolygons, const uint32_t* bucketPolygonsOffsets, const float2* allPoints,
  const uint32_t* bucketPointsOffsets, uint32_t bucketsNum, uint32_t* intersection);

void runIntersectAllPointsInBucket(uint32_t blocks, uint32_t threads, const float2 *vertices,
                                   const uint32_t *polygonOffsets, const uint32_t *bucketPolygons,
                                   const uint32_t *bucketPolygonsOffsets, const float2 *allPoints,
                                   const uint32_t *pointBucket, uint32_t pointsNum,
                                   uint32_t bucketsNum, uint32_t *intersection);

void runIntersectAllPointsInBucketNoDiv(
    uint32_t blocks, uint32_t threads, const float2 *vertices, const uint32_t *polygonOffsets,
    const uint32_t *bucketPolygons, const uint32_t *bucketPolygonsOffsets, const float2 *allPoints,
    const uint32_t *pointBucket, uint32_t pointsNum, uint32_t bucketsNum, uint32_t *intersection);


void ocl_intersectInBuckets_bucket(Device device, const Buckets &buckets, uint32_t *intersection);
void ocl_intersectInBuckets(Device device, const Buckets &buckets, uint32_t *intersection);
void ocl_intersectInBuckets_noDiv(Device device, const Buckets &buckets, uint32_t *intersection);


void cuda_intersectInBuckets_bucket(Device device, const Buckets &buckets, uint32_t *intersection);
void cuda_intersectInBuckets(Device device, const Buckets &buckets, uint32_t *intersection);
void cuda_intersectInBuckets_noDiv(Device device, const Buckets &buckets, uint32_t *intersection);

#endif // GPU_IMPL
