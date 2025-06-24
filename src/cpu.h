#ifndef CPU_IMPL
#define CPU_IMPL

#include "structs.h"

#include <cstdint>

void cpu_intersectInBuckets(Device device, const Buckets &buckets, uint32_t *intersection);

void cpu_intersectInBuckets_omp(Device device, const Buckets &buckets, uint32_t *intersection);

#endif // CPU_IMPL
