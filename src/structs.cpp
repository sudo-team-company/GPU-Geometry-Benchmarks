#include "structs.h"

#include <iomanip>

std::ostream &operator<<(std::ostream &os, const Coord &coord) {
  os << std::fixed << std::setprecision(2) << "(" << coord.x << ", " << coord.y << ")";
  return os;
}


std::ostream &operator<<(std::ostream &os, const std::vector<Coord> &points) {
  os << "[ ";
  for (Coord point : points) {
    os << point << " ";
  }
  os << "]";
  return os;
}


std::ostream &operator<<(std::ostream &os, const Buckets &buckets) {
  os << "Buckets:" << std::endl;
  os << "\tPolygons:" << std::endl;
  for (const Polygon &polygon : buckets.raw) {
    os << "\t\t" << polygon << std::endl;
  }
  os << "\tBuckets:" << std::endl;
  for (size_t i = 0; i < buckets.polygons.size(); i++) {
    os << "\t\tPolygons: [ ";
    for (size_t polygon : buckets.polygons[i]) {
      os << polygon << " ";
    }
    os << "]" << std::endl;
    os << "\t\tPoints: " << buckets.points[i] << std::endl;
    os << std::endl;
  }
  return os;
}


std::ostream &operator<<(std::ostream &os, const std::vector<uint32_t> &result) {
  os << "[ ";
  for (size_t i = 0; i < result.size(); i++) {
    if (i != 0 && i % RES_SIZE == 0) {
      os << "| ";
    }
    if (result[i] != RES_NONE) {
      os << result[i];
    } else {
      os << "_";
    }
    os << " ";
  }
  os << "]";
  return os;
}
