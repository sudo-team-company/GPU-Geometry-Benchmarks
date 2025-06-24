#ifndef STRUCTS
#define STRUCTS

#include <CL/cl.h>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <variant>
#include <vector>

struct Coord {
  float x;
  float y;
};

std::ostream &operator<<(std::ostream &os, const Coord &coord);

static_assert(sizeof(Coord) == 8);
static_assert(offsetof(Coord, x) == 0);
static_assert(offsetof(Coord, y) == 4);


using Polygon = std::vector<Coord>;

std::ostream &operator<<(std::ostream &os, const std::vector<Coord> &points);


struct Buckets {
  std::vector<Polygon> raw;
  std::vector<std::vector<size_t>> polygons;
  std::vector<std::vector<Coord>> points;
};

std::ostream &operator<<(std::ostream &os, const Buckets &buckets);


constexpr size_t RES_SIZE = 4;
constexpr uint32_t RES_NONE = 0xffffffff;

std::ostream &operator<<(std::ostream &os, const std::vector<uint32_t> &result);


struct RandomParams {
  size_t xNumBuckets;
  size_t yNumBuckets;
  float minX;
  float maxX;
  float minY;
  float maxY;
  size_t minPolygons;
  size_t maxPolygons;
  size_t minPoints;
  size_t maxPoints;
};


class CpuDevice {};
using Device = std::variant<CpuDevice, cl_device_id, int>;

#endif // STRUCTS
