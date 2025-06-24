#include "buckets.h"
#include "clUtils.h"
#include "cpu.h"
#include "gpu.h"
#include "structs.h"
#include "timer.h"
#include <CL/opencl.h>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>


using Solution = void (*)(Device, const Buckets &, uint32_t *);


TEST(test, simple) {
  Polygon square = {{0, 0}, {0, 10}, {10, 10}, {10, 0}};
  Polygon triangle = {{-10, 0}, {0, 20}, {10, 0}};
  Polygon pentagon = {{0, 0}, {4, 0}, {6, 4}, {2, 7}, {-2, 4}};
  Polygon arrow = {{0, 0}, {5, 5}, {10, 0}, {7, 0}, {7, -5}, {3, -5}, {3, 0}};
  std::vector<Polygon> polygons = {square, triangle, pentagon, arrow};


  std::vector<Coord> points = {{4, 2}, {-4, 0}, {2, 5}, {10, 10}, {9, 9}, {5, -2}, {-1, 4}, {2, 1}};
  uint32_t expected[8 * RES_SIZE] = {
      0, 1,        2,        3,        1,        RES_NONE, RES_NONE, RES_NONE,
      0, 1,        2,        RES_NONE, RES_NONE, RES_NONE, RES_NONE, RES_NONE,
      0, RES_NONE, RES_NONE, RES_NONE, 3,        RES_NONE, RES_NONE, RES_NONE,
      1, 2,        RES_NONE, RES_NONE, 0,        1,        2,        3,
  };


  Buckets buckets = makeBuckets(polygons, points);


  size_t intersection_size = 0;
  for (size_t i = 0; i < buckets.points.size(); i++) {
    intersection_size += buckets.points[i].size();
  }
  intersection_size *= RES_SIZE;


  std::vector<uint32_t> intersection_cpu(intersection_size, RES_NONE);
  cpu_intersectInBuckets_omp(CpuDevice(), buckets, intersection_cpu.data());


  std::vector<std::vector<uint32_t>> intersection_gpus;

  // CL
  std::vector<cl_device_id> devices = getClDevices();
  for (cl_device_id device : devices) {
    std::vector<uint32_t> intersection1(intersection_size, RES_NONE);
    ocl_intersectInBuckets_bucket(device, buckets, intersection1.data());
    intersection_gpus.push_back(intersection1);

    std::vector<uint32_t> intersection2(intersection_size, RES_NONE);
    ocl_intersectInBuckets(device, buckets, intersection2.data());
    intersection_gpus.push_back(intersection2);

    std::vector<uint32_t> intersection3(intersection_size, RES_NONE);
    ocl_intersectInBuckets_noDiv(device, buckets, intersection3.data());
    intersection_gpus.push_back(intersection3);
  }

  // CUDA
  int nDevices;
  cudaError_t err = cudaGetDeviceCount(&nDevices);
  if (err == cudaErrorNoDevice) {
    nDevices = 0;
  } else {
    CHECK_CUDA_STATUS(err);
  }
  for (int i = 0; i < nDevices; i++) {
    std::vector<cl_uint> intersection1(intersection_size, RES_NONE);
    cuda_intersectInBuckets_bucket(i, buckets, intersection1.data());
    intersection_gpus.push_back(intersection1);

    std::vector<cl_uint> intersection2(intersection_size, RES_NONE);
    cuda_intersectInBuckets(i, buckets, intersection2.data());
    intersection_gpus.push_back(intersection2);

    std::vector<cl_uint> intersection3(intersection_size, RES_NONE);
    cuda_intersectInBuckets_noDiv(i, buckets, intersection3.data());
    intersection_gpus.push_back(intersection3);
  }


  // Check first bucket only for now
  for (size_t i = 0; i < points.size() * RES_SIZE; i++) {
    if (expected[i] != intersection_cpu[i]) {
      throw std::runtime_error("CPU and precomputed solutions differ.");
    }
    if (i != 0) {
      std::cout << " ";
    }
    if (intersection_cpu[i] == RES_NONE) {
      std::cout << "_";
    } else {
      std::cout << intersection_cpu[i];
    }
  }
  std::cout << std::endl;

  EXPECT_FALSE(intersection_gpus.empty());

  // Check only first gpu result for now
  for (size_t i = 0; i < points.size() * RES_SIZE; i++) {
    EXPECT_EQ(expected[i], intersection_gpus[0][i]);
    if (i != 0) {
      std::cout << " ";
    }
    if (intersection_gpus[0][i] == RES_NONE) {
      std::cout << "_";
    } else {
      std::cout << intersection_gpus[0][i];
    }
  }
  std::cout << std::endl;


  // Compare cpu and gpu results
  for (size_t i = 0; i < intersection_gpus.size(); i++) {
    EXPECT_EQ(intersection_cpu, intersection_gpus[i]);
  }
}


std::vector<uint32_t> runSolutionSmall(Solution solve, Device device, const Buckets &buckets,
                                       size_t resultSize) {
  Timer timer;
  std::vector<uint32_t> result(resultSize, RES_NONE);
  timer.start();
  solve(device, buckets, result.data());
  timer.end();
  std::cout << timer << std::endl;
  std::cout << result << std::endl;
  std::cout << "--------------------------------------------------------------------------------"
            << std::endl;
  return result;
}

TEST(test, randomSmall) {
  size_t numBuckets = 3;
  float spaceCoeff = static_cast<float>(numBuckets) / 2.0f * 1000.0f;
  RandomParams rParams{
      numBuckets,  // xNumBuckets
      numBuckets,  // yNumBuckets
      -spaceCoeff, // minX
      spaceCoeff,  // maxX
      -spaceCoeff, // minY
      spaceCoeff,  // maxY
      4,           // minPolygons
      5,           // maxPolygons
      2,           // minPoints
      3,           // maxPoints
  };

  Timer timer;

  timer.start();
  Buckets buckets = generateRandomBuckets(rParams, true);
  size_t resultSize = 0;
  for (size_t i = 0; i < buckets.points.size(); i++) {
    resultSize += buckets.points[i].size();
  }
  resultSize *= RES_SIZE;
  timer.end();
  std::cout << timer << std::endl;
  std::cout << buckets << std::endl;
  std::cout << "--------------------------------------------------------------------------------"
            << std::endl;


  std::vector<cl_device_id> clDevices = getClDevices();
  int cudaDevices;
  cudaError_t err = cudaGetDeviceCount(&cudaDevices);
  if (err == cudaErrorNoDevice) {
    cudaDevices = 0;
  } else {
    CHECK_CUDA_STATUS(err);
  }


  std::vector<std::vector<uint32_t>> resultGpus;
  for (cl_device_id device : clDevices) {
    // Warm-up
    resultGpus.push_back(runSolutionSmall(ocl_intersectInBuckets, device, buckets, resultSize));

    resultGpus.push_back(
        runSolutionSmall(ocl_intersectInBuckets_bucket, device, buckets, resultSize));
    resultGpus.push_back(runSolutionSmall(ocl_intersectInBuckets, device, buckets, resultSize));
    resultGpus.push_back(
        runSolutionSmall(ocl_intersectInBuckets_noDiv, device, buckets, resultSize));
  }
  for (int i = 0; i < cudaDevices; i++) {
    // Warm-up
    resultGpus.push_back(runSolutionSmall(cuda_intersectInBuckets, i, buckets, resultSize));

    resultGpus.push_back(runSolutionSmall(cuda_intersectInBuckets_bucket, i, buckets, resultSize));
    resultGpus.push_back(runSolutionSmall(cuda_intersectInBuckets, i, buckets, resultSize));
    resultGpus.push_back(runSolutionSmall(cuda_intersectInBuckets_noDiv, i, buckets, resultSize));
  }
  std::vector<uint32_t> resultCpu =
      runSolutionSmall(cpu_intersectInBuckets_omp, CpuDevice(), buckets, resultSize);


  // Compare cpu and gpu results
  for (const std::vector<uint32_t> &resultGpu : resultGpus) {
    EXPECT_EQ(resultCpu, resultGpu);
  }


  size_t counter = 0;
  for (size_t i = 0; i < resultCpu.size(); i++) {
    if (resultCpu[i] != RES_NONE) {
      counter++;
    }
  }
  std::cout << "Found " << counter << " intersections" << std::endl;
}


std::vector<uint32_t> runSolution(Solution solve, Device device, const Buckets &buckets,
                                  size_t resultSize) {
  Timer timer;
  std::vector<uint32_t> result(resultSize, RES_NONE);
  timer.start();
  solve(device, buckets, result.data());
  timer.end();
  std::cout << timer << std::endl;
  std::cout << "--------------------------------------------------------------------------------"
            << std::endl;
  return result;
}

TEST(test, random) {
  size_t numBuckets = 256;
  float spaceCoeff = static_cast<float>(numBuckets) / 2.0f * 1000.0f;
  RandomParams rParams{
      numBuckets,  // xNumBuckets
      numBuckets,  // yNumBuckets
      -spaceCoeff, // minX
      spaceCoeff,  // maxX
      -spaceCoeff, // minY
      spaceCoeff,  // maxY
      1000,        // minPolygons
      1000,        // maxPolygons
      2000,        // minPoints
      2000,        // maxPoints
  };
  Timer timer;


  timer.start();
  Buckets buckets = generateRandomBuckets(rParams);
  size_t resultSize = 0;
  for (size_t i = 0; i < buckets.points.size(); i++) {
    resultSize += buckets.points[i].size();
  }
  resultSize *= RES_SIZE;
  timer.end();
  std::cout << timer << std::endl;
  std::cout << "--------------------------------------------------------------------------------"
            << std::endl;


  std::vector<cl_device_id> clDevices = getClDevices();
  int cudaDevices;
  cudaError_t err = cudaGetDeviceCount(&cudaDevices);
  if (err == cudaErrorNoDevice) {
    cudaDevices = 0;
  } else {
    CHECK_CUDA_STATUS(err);
  }


  std::vector<std::vector<uint32_t>> resultGpus;
  for (cl_device_id device : clDevices) {
    // Warm-up
    resultGpus.push_back(runSolution(ocl_intersectInBuckets, device, buckets, resultSize));

    resultGpus.push_back(runSolution(ocl_intersectInBuckets_bucket, device, buckets, resultSize));
    resultGpus.push_back(runSolution(ocl_intersectInBuckets, device, buckets, resultSize));
    resultGpus.push_back(runSolution(ocl_intersectInBuckets_noDiv, device, buckets, resultSize));
  }
  for (int i = 0; i < cudaDevices; i++) {
    // Warm-up
    resultGpus.push_back(runSolution(cuda_intersectInBuckets, i, buckets, resultSize));

    resultGpus.push_back(runSolution(cuda_intersectInBuckets_bucket, i, buckets, resultSize));
    resultGpus.push_back(runSolution(cuda_intersectInBuckets, i, buckets, resultSize));
    resultGpus.push_back(runSolution(cuda_intersectInBuckets_noDiv, i, buckets, resultSize));
  }
  std::vector<uint32_t> resultCpu =
      runSolution(cpu_intersectInBuckets_omp, CpuDevice(), buckets, resultSize);


  // Compare cpu and gpu results
  for (const std::vector<uint32_t> &resultGpu : resultGpus) {
    EXPECT_EQ(resultCpu, resultGpu);
  }


  size_t counter = 0;
  for (size_t i = 0; i < resultCpu.size(); i++) {
    if (resultCpu[i] != RES_NONE) {
      counter++;
    }
  }
  std::cout << "Found " << counter << " intersections" << std::endl;
}
