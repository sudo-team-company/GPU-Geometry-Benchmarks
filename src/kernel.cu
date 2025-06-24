#ifdef __INTELLISENSE__
#define __CUDACC__
#include <cuda_runtime.h>
#endif
#include <cstdint>

#define RES_SIZE (4)
#define RES_NONE (0xffffffff);

////////////////////////////////////////////////////////////////////////////////
//Bucket by thread                                                            //
////////////////////////////////////////////////////////////////////////////////

static __device__ bool isPointInsidePolygonFullBucket(const float2 *polygon, uint32_t polygonSize, float2 point) {
  int intersections = 0;
  for (size_t i = 0; i < polygonSize; i++) {
    float2 a = polygon[i];
    size_t j = i + 1;
    if (polygonSize == j) {
      j = 0;
    }
    float2 b = polygon[j];
    if (point.y >= min(a.y, b.y) && point.y < max(a.y, b.y)) {
      float dy = a.y - b.y;
      if (dy != 0 && (point.x < (point.y - b.y) * (a.x - b.x) / (a.y - b.y) + b.x)) {
        // Horizontal ray to the right intersects the edge.
        intersections++;
      }
    }
  }
  return intersections % 2 != 0;
}

static __device__ void intersectPointFullBucket(const float2 *vertices,     // [In] Polygons vertices
  const uint32_t *polygonOffsets, // [In] Polygons start offsets
  const uint32_t *polygons,       // [In] Bucket Polygons indices
  const uint32_t polygonsSize,             // [In] Bucket Polygons number
  const float2 point,                  // [In] Current Point
  uint32_t *intersection          // [Out] Result
) {
  for (size_t i = 0; i < RES_SIZE; i++) {
    intersection[i] = RES_NONE;
  }
  for (size_t i = 0, j = 0; i < polygonsSize && j < RES_SIZE; i++) {
    uint32_t curIdx = polygons[i];
    const float2 *polygon = vertices + polygonOffsets[curIdx];
    uint32_t size = polygonOffsets[curIdx + 1] - polygonOffsets[curIdx];
    if (isPointInsidePolygonFullBucket(polygon, size, point)) {
      intersection[j++] = curIdx;
    }
  }
}

static __device__ void intersectPointsFullBucket(const float2 *vertices,     // [In] Polygons vertices
  const uint32_t *polygonOffsets, // [In] Polygons start offsets
  const uint32_t *polygons,       // [In] Bucket Polygons indices
  const uint32_t polygonsSize,             // [In] Bucket Polygons number
  const float2 *points,       // [In] Bucket Points
  const uint32_t pointsSize,               // [In] Bucket Points number
  uint32_t *intersection          // [Out] Result
) {
  for (uint32_t i = 0; i < pointsSize; i++) {
    intersectPointFullBucket(vertices, polygonOffsets, polygons, polygonsSize, points[i],
      intersection + i * RES_SIZE);
  }
}

__global__ void intersectAllPointsInBucketFullBucket(
  const float2 *vertices,                // [In] Polygons vertices
  const uint32_t *polygonOffsets,        // [In] Polygons start offsets
  const uint32_t *bucketPolygons,        // [In] Bucket Polygons indices
  const uint32_t *bucketPolygonsOffsets, // [In] Bucket Polygons start offsets
  const float2 *allPoints,               // [In] Points
  const uint32_t *bucketPointsOffsets,   // [In] Bucket points start offsets
  uint32_t bucketsNum,                   // [In] Number of polygons
  uint32_t *intersection                 // [Out] Result
) {
  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < bucketsNum) {
    const uint32_t *polygons = bucketPolygons + bucketPolygonsOffsets[id];
    const uint32_t polygonsSize = bucketPolygonsOffsets[id + 1] - bucketPolygonsOffsets[id];
    const float2 *points = allPoints + bucketPointsOffsets[id];
    const uint32_t pointsSize = bucketPointsOffsets[id + 1] - bucketPointsOffsets[id];
    uint32_t *bucketIntersection = intersection + bucketPointsOffsets[id] * RES_SIZE;
    intersectPointsFullBucket(vertices, polygonOffsets, polygons, polygonsSize, points, pointsSize,
      bucketIntersection);
  }
}

void runIntersectAllPointsInBucketFullBucket(
  uint32_t blocks,
  uint32_t threads,
  const float2 *vertices,
  const uint32_t *polygonOffsets,
  const uint32_t *bucketPolygons,
  const uint32_t *bucketPolygonsOffsets,
  const float2 *allPoints,
  const uint32_t *bucketPointsOffsets,
  uint32_t bucketsNum,
  uint32_t *intersection
) {
  intersectAllPointsInBucketFullBucket<<<blocks, threads>>>(vertices, polygonOffsets, bucketPolygons,
    bucketPolygonsOffsets, allPoints, bucketPointsOffsets, bucketsNum, intersection);
}


////////////////////////////////////////////////////////////////////////////////
//Point by thread                                                             //
////////////////////////////////////////////////////////////////////////////////

static __device__ bool isPointInsidePolygon(const float2 *polygon, uint32_t polygonSize, float2 point) {
  int intersections = 0;
  for (size_t i = 0; i < polygonSize; i++) {
    float2 a = polygon[i];
    size_t j = i + 1;
    if (polygonSize == j) {
      j = 0;
    }
    float2 b = polygon[j];
    if (point.y >= min(a.y, b.y) && point.y < max(a.y, b.y)) {
      float dy = a.y - b.y;
      if (dy != 0 && (point.x < (point.y - b.y) * (a.x - b.x) / dy + b.x)) {
        // Horizontal ray to the right intersects the edge.
        intersections++;
      }
    }
  }
  return intersections % 2 != 0;
}

static __device__ void intersectPoint(const float2 *vertices,     // [In] Polygons vertices
  const uint32_t *polygonOffsets, // [In] Polygons start offsets
  const uint32_t *polygons,       // [In] Bucket Polygons indices
  const uint32_t polygonsSize,             // [In] Bucket Polygons number
  const float2 point,                  // [In] Current Point
  uint32_t *intersection          // [Out] Result
) {
  for (size_t i = 0; i < RES_SIZE; i++) {
    intersection[i] = RES_NONE;
  }
  for (size_t i = 0, j = 0; i < polygonsSize && j < RES_SIZE; i++) {
    uint32_t curIdx = polygons[i];
    const float2 *polygon = vertices + polygonOffsets[curIdx];
    uint32_t size = polygonOffsets[curIdx + 1] - polygonOffsets[curIdx];
    if (isPointInsidePolygon(polygon, size, point)) {
      intersection[j++] = curIdx;
    }
  }
}

__global__ void intersectAllPointsInBucket(
  const float2 *vertices,                // [In] Polygons vertices
  const uint32_t *polygonOffsets,        // [In] Polygons start offsets
  const uint32_t *bucketPolygons,        // [In] Bucket Polygons indices
  const uint32_t *bucketPolygonsOffsets, // [In] Bucket Polygons start offsets
  const float2 *allPoints,               // [In] Points
  const uint32_t *pointBucket,           // [In] Bucket number for point
  uint32_t pointsNum,                    // [In] Number of points
  uint32_t bucketsNum,                   // [In] Number of polygons
  uint32_t *intersection                 // [Out] Result
) {
  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < pointsNum && pointBucket[id] < bucketsNum) {
    const uint32_t *polygons = bucketPolygons + bucketPolygonsOffsets[pointBucket[id]];
    const uint32_t polygonsSize = bucketPolygonsOffsets[pointBucket[id] + 1] - bucketPolygonsOffsets[pointBucket[id]];
    uint32_t *bucketIntersection = intersection + id * RES_SIZE;
    intersectPoint(vertices, polygonOffsets, polygons, polygonsSize, allPoints[id], bucketIntersection);
  }
}

void runIntersectAllPointsInBucket(
  uint32_t blocks,
  uint32_t threads,
  const float2 *vertices,
  const uint32_t *polygonOffsets,
  const uint32_t *bucketPolygons,
  const uint32_t *bucketPolygonsOffsets,
  const float2 *allPoints,
  const uint32_t *pointBucket,
  uint32_t pointsNum,
  uint32_t bucketsNum,
  uint32_t *intersection
) {
  intersectAllPointsInBucket<<<blocks, threads>>>(vertices, polygonOffsets, bucketPolygons,
    bucketPolygonsOffsets, allPoints, pointBucket, pointsNum, bucketsNum, intersection);
}

////////////////////////////////////////////////////////////////////////////////
//Point by thread + avoid division                                            //
////////////////////////////////////////////////////////////////////////////////

static __device__ bool isPointInsidePolygonNoDiv(const float2 *polygon, uint32_t polygonSize, float2 point) {
  int intersections = 0;
  for (size_t i = 0; i < polygonSize; i++) {
    float2 a = polygon[i];
    size_t j = i + 1;
    if (polygonSize == j) {
      j = 0;
    }
    float2 b = polygon[j];
    if (a.y < b.y) {
      float t = a.x;
      a.x = b.x;
      b.x = t;
      t = a.y;
      a.y = b.y;
      b.y = t;
    }
    if (b.y <= point.y && point.y < a.y) {
      float dy = a.y - b.y;
      if (dy != 0 && (point.x - b.x) * dy < (point.y - b.y) * (a.x - b.x)) {
        // Horizontal ray to the right intersects the edge.
        intersections++;
      }
    }
  }
  return intersections % 2 != 0;
}

static __device__ void intersectPointNoDiv(const float2 *vertices,     // [In] Polygons vertices
  const uint32_t *polygonOffsets, // [In] Polygons start offsets
  const uint32_t *polygons,       // [In] Bucket Polygons indices
  const uint32_t polygonsSize,             // [In] Bucket Polygons number
  const float2 point,                  // [In] Current Point
  uint32_t *intersection          // [Out] Result
) {
  for (size_t i = 0; i < RES_SIZE; i++) {
    intersection[i] = RES_NONE;
  }
  for (size_t i = 0, j = 0; i < polygonsSize && j < RES_SIZE; i++) {
    uint32_t curIdx = polygons[i];
    const float2 *polygon = vertices + polygonOffsets[curIdx];
    uint32_t size = polygonOffsets[curIdx + 1] - polygonOffsets[curIdx];
    if (isPointInsidePolygonNoDiv(polygon, size, point)) {
      intersection[j++] = curIdx;
    }
  }
}

__global__ void intersectAllPointsInBucketNoDiv(
  const float2 *vertices,                // [In] Polygons vertices
  const uint32_t *polygonOffsets,        // [In] Polygons start offsets
  const uint32_t *bucketPolygons,        // [In] Bucket Polygons indices
  const uint32_t *bucketPolygonsOffsets, // [In] Bucket Polygons start offsets
  const float2 *allPoints,               // [In] Points
  const uint32_t *pointBucket,           // [In] Bucket number for point
  uint32_t pointsNum,                    // [In] Number of points
  uint32_t bucketsNum,                   // [In] Number of polygons
  uint32_t *intersection                 // [Out] Result
) {
  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < pointsNum && pointBucket[id] < bucketsNum) {
    const uint32_t *polygons = bucketPolygons + bucketPolygonsOffsets[pointBucket[id]];
    const uint32_t polygonsSize = bucketPolygonsOffsets[pointBucket[id] + 1] - bucketPolygonsOffsets[pointBucket[id]];
    uint32_t *bucketIntersection = intersection + id * RES_SIZE;
    intersectPointNoDiv(vertices, polygonOffsets, polygons, polygonsSize, allPoints[id], bucketIntersection);
  }
}

void runIntersectAllPointsInBucketNoDiv(
  uint32_t blocks,
  uint32_t threads,
  const float2 *vertices,
  const uint32_t *polygonOffsets,
  const uint32_t *bucketPolygons,
  const uint32_t *bucketPolygonsOffsets,
  const float2 *allPoints,
  const uint32_t *pointBucket,
  uint32_t pointsNum,
  uint32_t bucketsNum,
  uint32_t *intersection
) {
  intersectAllPointsInBucketNoDiv<<<blocks, threads>>>(vertices, polygonOffsets, bucketPolygons,
    bucketPolygonsOffsets, allPoints, pointBucket, pointsNum, bucketsNum, intersection);
}
