#include "buckets.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

Buckets makeBuckets(const std::vector<Polygon> &polygons, const std::vector<Coord> &points) {
  Buckets buckets = {polygons, {}, {}};

  buckets.polygons.push_back({0, 1, 2, 3});
  buckets.points.push_back(points);

  buckets.polygons.push_back({1, 2, 3});
  buckets.points.push_back(points);

  buckets.polygons.push_back({2, 3});
  buckets.points.push_back(points);

  buckets.polygons.push_back({3});
  buckets.points.push_back(points);

  buckets.polygons.push_back({1});
  buckets.points.push_back(points);

  while (buckets.polygons.size() < 64) {
    buckets.polygons.push_back({0, 1, 2, 3});
    buckets.points.push_back(points);
  }

  if (buckets.polygons.size() != buckets.points.size()) {
    throw std::runtime_error("Incorrect buckets data.");
  }
  return buckets;
}


constexpr float MIN_SEGMENT_LENGTH = 0.1f;        // (squared)
constexpr float MIN_POINT_TO_SEGMENT_DIST = 0.1f; // (squared)

constexpr inline float distSquared(Coord A, Coord B) {
  float dx = B.x - A.x;
  float dy = B.y - A.y;
  return dx * dx + dy * dy;
}

constexpr inline float distSquared(Coord A, Coord B, Coord P) {
  float dx = B.x - A.x;
  float dy = B.y - A.y;
  float d2 = dx * dx + dy * dy;
  float c = dy * P.x - dx * P.y + B.x * A.y - B.y * A.x;
  return c * c / d2;
}

constexpr inline bool isGoodTriangle(Coord A, Coord B, Coord C) {
  return distSquared(A, B) > MIN_SEGMENT_LENGTH && distSquared(B, C) > MIN_SEGMENT_LENGTH &&
         distSquared(C, A) > MIN_SEGMENT_LENGTH;
}

constexpr inline bool isGoodPoint(Coord A, Coord B, Coord P) {
  float AP = distSquared(A, P);
  if (AP < MIN_POINT_TO_SEGMENT_DIST) {
    return false;
  }
  float BP = distSquared(B, P);
  if (BP < MIN_POINT_TO_SEGMENT_DIST) {
    return false;
  }
  float minX = std::min(A.x, B.x) - MIN_POINT_TO_SEGMENT_DIST;
  if (P.x < minX) {
    return true;
  }
  float maxX = std::max(A.x, B.x) + MIN_POINT_TO_SEGMENT_DIST;
  if (maxX < P.x) {
    return true;
  }
  float minY = std::min(A.y, B.y) - MIN_POINT_TO_SEGMENT_DIST;
  if (P.y < minY) {
    return true;
  }
  float maxY = std::max(A.y, B.y) + MIN_POINT_TO_SEGMENT_DIST;
  if (maxY < P.y) {
    return true;
  }
  return distSquared(A, B, P) > MIN_POINT_TO_SEGMENT_DIST;
}

inline bool isGoodPoint(const std::vector<Polygon> &raw, const std::vector<size_t> &polygons,
                        Coord P) {
  for (size_t i : polygons) {
    if (!isGoodPoint(raw[i][0], raw[i][1], P)) {
      return false;
    }
    if (!isGoodPoint(raw[i][1], raw[i][2], P)) {
      return false;
    }
    if (!isGoodPoint(raw[i][2], raw[i][0], P)) {
      return false;
    }
  }
  return true;
}

Buckets generateRandomBuckets(const RandomParams &params, bool enableChecks) {
  std::cout << "Random generation params:" << std::endl;
  std::cout << "\tNumber of buckets: " << params.xNumBuckets * params.yNumBuckets << " | "
            << params.xNumBuckets << " x " << params.yNumBuckets << std::endl;
  std::cout << "\tFull OX axis space from " << params.minX << " to " << params.maxX << std::endl;
  std::cout << "\tFull OY axis space from " << params.minY << " to " << params.maxY << std::endl;
  std::cout << "\tNumber of polygons in bucket from " << params.minPolygons << " to "
            << params.maxPolygons << std::endl;
  std::cout << "\tNumber of points in bucket from " << params.minPoints << " to "
            << params.maxPoints << std::endl;


  std::vector<Polygon> raw;
  std::vector<std::vector<size_t>> polygons(params.xNumBuckets * params.yNumBuckets);
  std::vector<std::vector<Coord>> points(params.xNumBuckets * params.yNumBuckets);

  float dX = (params.maxX - params.minX) / params.xNumBuckets;
  float dY = (params.maxY - params.minY) / params.yNumBuckets;

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<size_t> distPolygonsNum(params.minPolygons, params.maxPolygons);
  std::uniform_int_distribution<size_t> distPointsNum(params.minPoints, params.maxPoints);
  std::uniform_real_distribution<float> distX(0.0f, dX);
  std::uniform_real_distribution<float> distY(0.0f, dY);

  for (size_t i = 0; i < params.yNumBuckets; i++) {
    for (size_t j = 0; j < params.xNumBuckets; j++) {
      size_t bucketIdx = i * params.xNumBuckets + j;
      float curMinX = params.minX + i * dX;
      float curMinY = params.minY + j * dY;

      size_t polygonsNum = distPolygonsNum(gen);
      for (size_t k = 0; k < polygonsNum; k++) {
        Polygon polygon;
        for (size_t l = 0; l < 3; l++) {
          float x = curMinX + distX(gen);
          float y = curMinY + distY(gen);
          polygon.push_back(Coord{x, y});
        }
        if (enableChecks && !isGoodTriangle(polygon[0], polygon[1], polygon[2])) {
          k--;
          continue;
        }
        polygons[bucketIdx].push_back(raw.size());
        raw.push_back(polygon);
      }

      size_t pointsNum = distPointsNum(gen);
      for (size_t k = 0; k < pointsNum; k++) {
        float x = curMinX + distX(gen);
        float y = curMinY + distY(gen);
        Coord p{x, y};
        if (enableChecks && !isGoodPoint(raw, polygons[bucketIdx], p)) {
          k--;
          continue;
        }
        points[bucketIdx].push_back(p);
      }
    }
  }

  if (polygons.size() != points.size()) {
    throw std::runtime_error("Incorrect buckets data.");
  }


  std::cout << std::endl;
  std::cout << "\tGenerated total polygons: " << raw.size() << std::endl;
  size_t counter = 0;
  for (auto bucket : points) {
    counter += bucket.size();
  }
  std::cout << "\tGenerated total points: " << counter << std::endl;
  std::cout << std::endl;


  return Buckets{raw, polygons, points};
}
