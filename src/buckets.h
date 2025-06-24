#ifndef BUCKETS
#define BUCKETS

#include "structs.h"

Buckets makeBuckets(const std::vector<Polygon> &polygons, const std::vector<Coord> &points);

/// Generates random buckets. All polygons are triangles. Each polygon is inside only one bucket.
Buckets generateRandomBuckets(const RandomParams &params, bool enableChecks = false);

#endif // BUCKETS
