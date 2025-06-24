#define RES_SIZE (4)
#define RES_NONE (0xffffffff)

////////////////////////////////////////////////////////////////////////////////
//Bucket by thread                                                            //
////////////////////////////////////////////////////////////////////////////////

static bool isPointInsidePolygonFullBucket(__global const float2 *polygon, uint polygonSize, float2 point) {
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

static void intersectPointFullBucket(__global const float2 *vertices,     // [In] Polygons vertices
                                     __global const uint *polygonOffsets, // [In] Polygons start offsets
                                     __global const uint *polygons,       // [In] Bucket Polygons indices
                                     const uint polygonsSize,             // [In] Bucket Polygons number
                                     const float2 point,                  // [In] Current Point
                                     __global uint *intersection          // [Out] Result
) {
  for (size_t i = 0; i < RES_SIZE; i++) {
    intersection[i] = RES_NONE;
  }
  for (size_t i = 0, j = 0; i < polygonsSize && j < RES_SIZE; i++) {
    uint curIdx = polygons[i];
    __global const float2 *polygon = vertices + polygonOffsets[curIdx];
    uint size = polygonOffsets[curIdx + 1] - polygonOffsets[curIdx];
    if (isPointInsidePolygonFullBucket(polygon, size, point)) {
      intersection[j++] = curIdx;
    }
  }
}

static void intersectPointsFullBucket(__global const float2 *vertices,     // [In] Polygons vertices
                                      __global const uint *polygonOffsets, // [In] Polygons start offsets
                                      __global const uint *polygons,       // [In] Bucket Polygons indices
                                      const uint polygonsSize,             // [In] Bucket Polygons number
                                      __global const float2 *points,       // [In] Bucket Points
                                      const uint pointsSize,               // [In] Bucket Points number
                                      __global uint *intersection          // [Out] Result
) {
  for (uint i = 0; i < pointsSize; i++) {
    intersectPointFullBucket(vertices, polygonOffsets, polygons, polygonsSize, points[i], intersection + i * RES_SIZE);
  }
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void intersectAllPointsInBucketFullBucket(
    __global const float2 *vertices,            // [In] Polygons vertices
    __global const uint *polygonOffsets,        // [In] Polygons start offsets
    __global const uint *bucketPolygons,        // [In] Bucket Polygons indices
    __global const uint *bucketPolygonsOffsets, // [In] Bucket Polygons start offsets
    __global const float2 *allPoints,           // [In] Points
    __global const uint *bucketPointsOffsets,   // [In] Bucket points start offsets
    uint bucketsNum,                            // [In] Number of polygons
    __global uint *intersection                 // [Out] Result
) {
  uint id = get_global_id(0);
  if (id < bucketsNum) {
    __global const uint *polygons = bucketPolygons + bucketPolygonsOffsets[id];
    const uint polygonsSize = bucketPolygonsOffsets[id + 1] - bucketPolygonsOffsets[id];
    __global const float2 *points = allPoints + bucketPointsOffsets[id];
    const uint pointsSize = bucketPointsOffsets[id + 1] - bucketPointsOffsets[id];
    __global uint *bucketIntersection = intersection + bucketPointsOffsets[id] * RES_SIZE;
    intersectPointsFullBucket(vertices, polygonOffsets, polygons, polygonsSize, points, pointsSize,
                    bucketIntersection);
  }
}

////////////////////////////////////////////////////////////////////////////////
//Point by thread                                                             //
////////////////////////////////////////////////////////////////////////////////

static bool isPointInsidePolygon(__global const float2 *polygon, uint polygonSize, float2 point) {
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

static void intersectPoint(__global const float2 *vertices,     // [In] Polygons vertices
                           __global const uint *polygonOffsets, // [In] Polygons start offsets
                           __global const uint *polygons,       // [In] Bucket Polygons indices
                           const uint polygonsSize,             // [In] Bucket Polygons number
                           const float2 point,                  // [In] Current Point
                           __global uint *intersection          // [Out] Result
) {
  for (size_t i = 0; i < RES_SIZE; i++) {
    intersection[i] = RES_NONE;
  }
  for (size_t i = 0, j = 0; i < polygonsSize && j < RES_SIZE; i++) {
    uint curIdx = polygons[i];
    __global const float2 *polygon = vertices + polygonOffsets[curIdx];
    uint size = polygonOffsets[curIdx + 1] - polygonOffsets[curIdx];
    if (isPointInsidePolygon(polygon, size, point)) {
      intersection[j++] = curIdx;
    }
  }
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void intersectAllPointsInBucket(
    __global const float2 *vertices,            // [In] Polygons vertices
    __global const uint *polygonOffsets,        // [In] Polygons start offsets
    __global const uint *bucketPolygons,        // [In] Bucket Polygons indices
    __global const uint *bucketPolygonsOffsets, // [In] Bucket Polygons start offsets
    __global const float2 *allPoints,           // [In] Points
    __global const uint *pointBucket,           // [In] Bucket number for point
    uint pointsNum,                             // [In] Number of points
    uint bucketsNum,                            // [In] Number of polygons
    __global uint *intersection                 // [Out] Result
) {
  uint id = get_global_id(0);
  if (id < pointsNum && pointBucket[id] < bucketsNum) {
    __global const uint *polygons = bucketPolygons + bucketPolygonsOffsets[pointBucket[id]];
    const uint polygonsSize = bucketPolygonsOffsets[pointBucket[id] + 1] - bucketPolygonsOffsets[pointBucket[id]];
    __global uint *bucketIntersection = intersection + id * RES_SIZE;
    intersectPoint(vertices, polygonOffsets, polygons, polygonsSize, allPoints[id], bucketIntersection);
  }
}

////////////////////////////////////////////////////////////////////////////////
//Point by thread + avoid division                                            //
////////////////////////////////////////////////////////////////////////////////

static bool isPointInsidePolygonNoDiv(__global const float2 *polygon, uint polygonSize, float2 point) {
  int intersections = 0;
  for (size_t i = 0; i < polygonSize; i++) {
    float2 a = polygon[i];
    size_t j = i + 1;
    if (polygonSize == j) {
      j = 0;
    }
    float2 b = polygon[j];    if (a.y < b.y) {
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

static void intersectPointNoDiv(__global const float2 *vertices,     // [In] Polygons vertices
                                __global const uint *polygonOffsets, // [In] Polygons start offsets
                                __global const uint *polygons,       // [In] Bucket Polygons indices
                                const uint polygonsSize,             // [In] Bucket Polygons number
                                const float2 point,                  // [In] Current Point
                                __global uint *intersection          // [Out] Result
) {
  for (size_t i = 0; i < RES_SIZE; i++) {
    intersection[i] = RES_NONE;
  }
  for (size_t i = 0, j = 0; i < polygonsSize && j < RES_SIZE; i++) {
    uint curIdx = polygons[i];
    __global const float2 *polygon = vertices + polygonOffsets[curIdx];
    uint size = polygonOffsets[curIdx + 1] - polygonOffsets[curIdx];
    if (isPointInsidePolygonNoDiv(polygon, size, point)) {
      intersection[j++] = curIdx;
    }
  }
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void intersectAllPointsInBucketNoDiv(
    __global const float2 *vertices,            // [In] Polygons vertices
    __global const uint *polygonOffsets,        // [In] Polygons start offsets
    __global const uint *bucketPolygons,        // [In] Bucket Polygons indices
    __global const uint *bucketPolygonsOffsets, // [In] Bucket Polygons start offsets
    __global const float2 *allPoints,           // [In] Points
    __global const uint *pointBucket,           // [In] Bucket number for point
    uint pointsNum,                             // [In] Number of points
    uint bucketsNum,                            // [In] Number of polygons
    __global uint *intersection                 // [Out] Result
) {
  uint id = get_global_id(0);
  if (id < pointsNum && pointBucket[id] < bucketsNum) {
    __global const uint *polygons = bucketPolygons + bucketPolygonsOffsets[pointBucket[id]];
    const uint polygonsSize = bucketPolygonsOffsets[pointBucket[id] + 1] - bucketPolygonsOffsets[pointBucket[id]];
    __global uint *bucketIntersection = intersection + id * RES_SIZE;
    intersectPointNoDiv(vertices, polygonOffsets, polygons, polygonsSize, allPoints[id], bucketIntersection);
  }
}
