#include "cpu.h"

#include <algorithm>
#include <iostream>
#include <vector>

static bool isPointInsidePolygon_old(const Polygon &polygon, const Coord &point) {
  int intersections = 0;
  for (size_t i = 0; i < polygon.size(); i++) {
    float ax = polygon[i].x;
    float ay = polygon[i].y;
    size_t j = i + 1;
    if (polygon.size() == j) {
      j = 0;
    }
    float bx = polygon[j].x;
    float by = polygon[j].y;
    if (point.y >= std::min(ay, by) && point.y < std::max(ay, by)) {
      float dy = ay - by;
      if (dy != 0 && (point.x < (point.y - by) * (ax - bx) / (ay - by) + bx)) {
        // Horizontal ray to the right intersects the edge.
        intersections++;
      }
    }
  }
  return intersections % 2 != 0;
}

static void intersectPoint_old(const std::vector<Polygon> &raw, const std::vector<size_t> &polygons,
                               const Coord &point, uint32_t *intersection) {
  for (size_t i = 0; i < RES_SIZE; i++) {
    intersection[i] = RES_NONE;
  }
  for (size_t i = 0, j = 0; i < polygons.size() && j < RES_SIZE; i++) {
    if (isPointInsidePolygon_old(raw[polygons[i]], point)) {
      intersection[j++] = polygons[i];
    }
  }
}

static void intersectAllPointsInBucket_old(const std::vector<Polygon> &raw,
                                           const std::vector<size_t> &polygons,
                                           const std::vector<Coord> &points,
                                           uint32_t *intersection) {
  for (size_t i = 0; i < points.size(); i++) {
    intersectPoint_old(raw, polygons, points[i], intersection + i * RES_SIZE);
  }
}

void cpu_intersectInBuckets(Device, const Buckets &buckets, uint32_t *intersection) {
  std::cout << "CPU:  Running on cpu" << std::endl;
  uint32_t *bucket_intersection = intersection;
  for (size_t i = 0; i < buckets.polygons.size(); i++) {
    std::vector<size_t> polygons = buckets.polygons[i];
    std::vector<Coord> points = buckets.points[i];
    intersectAllPointsInBucket_old(buckets.raw, polygons, points, bucket_intersection);
    bucket_intersection += points.size() * RES_SIZE;
  }
}


////////////////////////////////////////////////////////////////////////////////


static bool isPointInsidePolygon(const Polygon &polygon, const Coord &p) {
  int intersections = 0;
  size_t n = polygon.size();
  for (size_t i = 0, j = 1; i < n; i++, j = i + 1) {
    if (n == j) {
      j = 0;
    }
    float ax = polygon[i].x;
    float ay = polygon[i].y;
    float bx = polygon[j].x;
    float by = polygon[j].y;
    if (std::min(ay, by) <= p.y && p.y < std::max(ay, by)) {
      if ((ay - by) != 0 && (p.x < (p.y - by) * (ax - bx) / (ay - by) + bx)) {
        // Horizontal ray to the right intersects the edge.
        intersections++;
      }
    }
  }
  return intersections % 2 != 0;
}

static void intersectPoint(const std::vector<Polygon> &raw, const std::vector<size_t> &polygons,
                           const Coord &point, uint32_t *intersection) {
  for (size_t i = 0; i < RES_SIZE; i++) {
    intersection[i] = RES_NONE;
  }
  for (size_t i = 0, j = 0; i < polygons.size() && j < RES_SIZE; i++) {
    if (isPointInsidePolygon(raw[polygons[i]], point)) {
      intersection[j++] = polygons[i];
    }
  }
}

static void intersectAllPointsInBucket(const std::vector<Polygon> &raw,
                                       const std::vector<size_t> &polygons,
                                       const std::vector<Coord> &points, uint32_t *intersection) {
  for (size_t i = 0; i < points.size(); i++) {
    intersectPoint(raw, polygons, points[i], intersection + i * RES_SIZE);
  }
}

void cpu_intersectInBuckets_omp(Device, const Buckets &buckets, uint32_t *intersection) {
  std::cout << "CPU:  Running on cpu (OpenMP)" << std::endl;
  std::vector<uint32_t *> bucket_intersections(buckets.polygons.size());
  bucket_intersections[0] = intersection;
  for (int i = 1; i < (int)buckets.polygons.size(); i++) {
    bucket_intersections[i] = bucket_intersections[i - 1] + buckets.points[i - 1].size() * RES_SIZE;
  }
#pragma omp parallel for
  for (int i = 0; i < (int)buckets.polygons.size(); i++) {
    const std::vector<size_t> &polygons = buckets.polygons[i];
    const std::vector<Coord> &points = buckets.points[i];
    intersectAllPointsInBucket(buckets.raw, polygons, points, bucket_intersections[i]);
  }
}
