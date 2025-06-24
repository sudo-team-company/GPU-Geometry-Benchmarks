#include "gpu.h"

#include "clDeviceManager.h"
#include "clUtils.h"
#include "structs.h"
#include "timer.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void ThrowOnCudaError(cudaError_t err, char const *file, int line) {
  if (err == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " +
                           cudaGetErrorString(err) + " (" + cudaGetErrorName(err) + ")");
}

static std::string loadKernelSource(const std::string &filename) {
  std::ifstream file(filename);
  std::stringstream ss;
  ss << file.rdbuf();
  return ss.str();
}


void ocl_intersectInBuckets_bucket(Device device_, const Buckets &buckets, uint32_t *intersection) {
  std::cout << "CL:   Bucket by thread" << std::endl;
  cl_device_id device = std::get<cl_device_id>(device_);
  cl_int clStatus = CL_SUCCESS;

  clDeviceManager dev(device);
  const auto deviceName = dev.getDeviceName();
  if (deviceName.find("Simulator") != std::string::npos) {
    std::cout << "Device " << deviceName << " is not supported" << std::endl;
    return;
  }
  std::cout << "CL:   Running on " << deviceName << std::endl;

  std::string kernelSource = loadKernelSource("src/kernel.cl");
  cl_kernel kernel = dev.loadKernel(kernelSource, "intersectAllPointsInBucketFullBucket");

  cl_uint bucketsNum = buckets.points.size();

  size_t localSizes[3] = {0};
  CHECK_OPENCL_CALL(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                             sizeof(localSizes), localSizes, nullptr));
  size_t localSize = localSizes[0];
  if (1 != localSizes[1] || 1 != localSizes[2]) {
    throw std::runtime_error("Bad local sizes of kernel.");
  }

  size_t totalPointsNum = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    totalPointsNum += buckets.points[i].size();
  }

  size_t globalSize = align(bucketsNum, localSize);

  size_t totalVerticesNum = 0;
  for (size_t i = 0; i < buckets.raw.size(); i++) {
    totalVerticesNum += buckets.raw[i].size();
  }

  std::vector<cl_float2> vertices(totalVerticesNum);
  for (size_t i = 0, offset = 0; i < buckets.raw.size(); i++) {
    std::memcpy(vertices.data() + offset, buckets.raw[i].data(),
                sizeof(Coord) * buckets.raw[i].size());
    offset += buckets.raw[i].size();
  }
  cl_mem verticesBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_float2) * totalVerticesNum, vertices.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  std::vector<cl_uint> polygonOffsets(buckets.raw.size() + 1);
  polygonOffsets[0] = 0;
  for (size_t i = 0; i < buckets.raw.size(); i++) {
    polygonOffsets[i + 1] = polygonOffsets[i] + buckets.raw[i].size();
  }
  cl_mem polygonOffsetsBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_uint) * polygonOffsets.size(), polygonOffsets.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  size_t totalBucketPolygonsNum = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    totalBucketPolygonsNum += buckets.polygons[i].size();
  }

  std::vector<cl_uint> bucketPolygons(totalBucketPolygonsNum);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    for (size_t j = 0; j < buckets.polygons[i].size(); j++) {
      bucketPolygons[offset + j] = buckets.polygons[i][j];
    }
    offset += buckets.polygons[i].size();
  }
  cl_mem bucketPolygonsBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_uint) * bucketPolygons.size(), bucketPolygons.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  std::vector<cl_uint> bucketPolygonsOffsets(bucketsNum + 1);
  bucketPolygonsOffsets[0] = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    bucketPolygonsOffsets[i + 1] = bucketPolygonsOffsets[i] + buckets.polygons[i].size();
  }
  cl_mem bucketPolygonsOffsetsBuf = clCreateBuffer(
      dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(cl_uint) * bucketPolygonsOffsets.size(), bucketPolygonsOffsets.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  std::vector<cl_float2> allPoints(totalPointsNum);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    std::memcpy(allPoints.data() + offset, buckets.points[i].data(),
                sizeof(Coord) * buckets.points[i].size());
    offset += buckets.points[i].size();
  }
  cl_mem allPointsBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_float2) * allPoints.size(), allPoints.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  std::vector<cl_uint> bucketPointsOffsets(bucketsNum + 1);
  bucketPointsOffsets[0] = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    bucketPointsOffsets[i + 1] = bucketPointsOffsets[i] + buckets.points[i].size();
  }
  cl_mem bucketPointsOffsetsBuf = clCreateBuffer(
      dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(cl_uint) * bucketPointsOffsets.size(), bucketPointsOffsets.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);


  size_t intersectionSize = allPoints.size() * RES_SIZE;
  cl_mem intersectionBuf = clCreateBuffer(dev.getContext(), CL_MEM_READ_WRITE,
                                          sizeof(cl_uint) * intersectionSize, nullptr, &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  int arg = 0;
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &verticesBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &polygonOffsetsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &bucketPolygonsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &bucketPolygonsOffsetsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &allPointsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &bucketPointsOffsetsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_uint), &bucketsNum));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &intersectionBuf));

  CHECK_OPENCL_CALL(clFinish(dev.getQueue()));

  Timer timer;
  timer.start();
  CHECK_OPENCL_CALL(clEnqueueNDRangeKernel(dev.getQueue(), kernel, 1, nullptr, &globalSize,
                                           &localSize, 0, nullptr, nullptr));
  CHECK_OPENCL_CALL(clEnqueueReadBuffer(dev.getQueue(), intersectionBuf, CL_TRUE, 0,
                                        sizeof(cl_uint) * intersectionSize, intersection, 0,
                                        nullptr, nullptr));
  timer.end();
  std::cout << "[KERNEL] " << timer << std::endl;

  CHECK_OPENCL_CALL(clReleaseMemObject(intersectionBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(bucketPointsOffsetsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(allPointsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(bucketPolygonsOffsetsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(bucketPolygonsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(polygonOffsetsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(verticesBuf));

  CHECK_OPENCL_CALL(clReleaseKernel(kernel));
}

void ocl_intersectInBuckets(Device device_, const Buckets &buckets, uint32_t *intersection) {
  std::cout << "CL:   Point by thread" << std::endl;
  cl_device_id device = std::get<cl_device_id>(device_);
  cl_int clStatus = CL_SUCCESS;

  clDeviceManager dev(device);
  const auto deviceName = dev.getDeviceName();
  if (deviceName.find("Simulator") != std::string::npos) {
    std::cout << "Device " << deviceName << " is not supported" << std::endl;
    return;
  }
  std::cout << "CL:   Running on " << deviceName << std::endl;

  std::string kernelSource = loadKernelSource("src/kernel.cl");
  cl_kernel kernel = dev.loadKernel(kernelSource, "intersectAllPointsInBucket");

  cl_uint bucketsNum = buckets.points.size();

  size_t localSizes[3] = {0};
  CHECK_OPENCL_CALL(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                             sizeof(localSizes), localSizes, nullptr));
  size_t localSize = localSizes[0];
  if (1 != localSizes[1] || 1 != localSizes[2]) {
    throw std::runtime_error("Bad local sizes of kernel.");
  }

  size_t extendedPointsSize = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    extendedPointsSize += align(buckets.points[i].size(), localSize);
  }

  size_t globalSize = align(extendedPointsSize, localSize);

  size_t totalVerticesNum = 0;
  for (size_t i = 0; i < buckets.raw.size(); i++) {
    totalVerticesNum += buckets.raw[i].size();
  }

  std::vector<cl_float2> vertices(totalVerticesNum);
  for (size_t i = 0, offset = 0; i < buckets.raw.size(); i++) {
    std::memcpy(vertices.data() + offset, buckets.raw[i].data(),
                sizeof(Coord) * buckets.raw[i].size());
    offset += buckets.raw[i].size();
  }
  cl_mem verticesBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_float2) * totalVerticesNum, vertices.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  std::vector<cl_uint> polygonOffsets(buckets.raw.size() + 1);
  polygonOffsets[0] = 0;
  for (size_t i = 0; i < buckets.raw.size(); i++) {
    polygonOffsets[i + 1] = polygonOffsets[i] + buckets.raw[i].size();
  }
  cl_mem polygonOffsetsBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_uint) * polygonOffsets.size(), polygonOffsets.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  size_t totalBucketPolygonsNum = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    totalBucketPolygonsNum += buckets.polygons[i].size();
  }

  std::vector<cl_uint> bucketPolygons(totalBucketPolygonsNum);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    for (size_t j = 0; j < buckets.polygons[i].size(); j++) {
      bucketPolygons[offset + j] = buckets.polygons[i][j];
    }
    offset += buckets.polygons[i].size();
  }
  cl_mem bucketPolygonsBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_uint) * bucketPolygons.size(), bucketPolygons.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  std::vector<cl_uint> bucketPolygonsOffsets(bucketsNum + 1);
  bucketPolygonsOffsets[0] = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    bucketPolygonsOffsets[i + 1] = bucketPolygonsOffsets[i] + buckets.polygons[i].size();
  }
  cl_mem bucketPolygonsOffsetsBuf = clCreateBuffer(
      dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(cl_uint) * bucketPolygonsOffsets.size(), bucketPolygonsOffsets.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  std::vector<cl_float2> allPoints(extendedPointsSize);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    std::memcpy(allPoints.data() + offset, buckets.points[i].data(),
                sizeof(Coord) * buckets.points[i].size());
    offset += align(buckets.points[i].size(), localSize);
  }
  cl_mem allPointsBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_float2) * allPoints.size(), allPoints.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  std::vector<cl_uint> pointBucket(extendedPointsSize);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    cl_uint *curOff = pointBucket.data() + offset;
    cl_uint curSize = buckets.points[i].size();
    std::fill(curOff, curOff + curSize, i);
    std::fill(curOff + curSize, curOff + align(curSize, localSize), bucketsNum);
    offset += align(buckets.points[i].size(), localSize);
  }
  cl_mem pointBucketBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_uint) * pointBucket.size(), pointBucket.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);


  size_t intersectionSize = allPoints.size() * RES_SIZE;
  cl_mem intersectionBuf = clCreateBuffer(dev.getContext(), CL_MEM_READ_WRITE,
                                          sizeof(cl_uint) * intersectionSize, nullptr, &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  int arg = 0;
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &verticesBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &polygonOffsetsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &bucketPolygonsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &bucketPolygonsOffsetsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &allPointsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &pointBucketBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_uint), &extendedPointsSize));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_uint), &bucketsNum));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &intersectionBuf));

  std::vector<cl_uint> intersectionTmp(intersectionSize);
  CHECK_OPENCL_CALL(clFinish(dev.getQueue()));

  Timer timer;
  timer.start();
  CHECK_OPENCL_CALL(clEnqueueNDRangeKernel(dev.getQueue(), kernel, 1, nullptr, &globalSize,
                                           &localSize, 0, nullptr, nullptr));
  CHECK_OPENCL_CALL(clEnqueueReadBuffer(dev.getQueue(), intersectionBuf, CL_TRUE, 0,
                                        sizeof(cl_uint) * intersectionSize, intersectionTmp.data(),
                                        0, nullptr, nullptr));
  timer.end();
  std::cout << "[KERNEL] " << timer << std::endl;

  for (size_t i = 0, offset1 = 0, offset2 = 0; i < bucketsNum; i++) {
    std::memcpy(intersection + offset1, intersectionTmp.data() + offset2,
                sizeof(cl_uint) * buckets.points[i].size() * RES_SIZE);
    offset1 += buckets.points[i].size() * RES_SIZE;
    offset2 += align(buckets.points[i].size(), localSize) * RES_SIZE;
  }

  CHECK_OPENCL_CALL(clReleaseMemObject(intersectionBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(pointBucketBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(allPointsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(bucketPolygonsOffsetsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(bucketPolygonsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(polygonOffsetsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(verticesBuf));

  CHECK_OPENCL_CALL(clReleaseKernel(kernel));
}

void ocl_intersectInBuckets_noDiv(Device device_, const Buckets &buckets, uint32_t *intersection) {
  std::cout << "CL:   Point by thread + avoid div optimization" << std::endl;
  cl_device_id device = std::get<cl_device_id>(device_);
  cl_int clStatus = CL_SUCCESS;

  clDeviceManager dev(device);
  const auto deviceName = dev.getDeviceName();
  if (deviceName.find("Simulator") != std::string::npos) {
    std::cout << "Device " << deviceName << " is not supported" << std::endl;
    return;
  }
  std::cout << "CL:   Running on " << deviceName << std::endl;

  std::string kernelSource = loadKernelSource("src/kernel.cl");
  cl_kernel kernel = dev.loadKernel(kernelSource, "intersectAllPointsInBucketNoDiv");

  cl_uint bucketsNum = buckets.points.size();

  size_t localSizes[3] = {0};
  CHECK_OPENCL_CALL(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                             sizeof(localSizes), localSizes, nullptr));
  size_t localSize = localSizes[0];
  if (1 != localSizes[1] || 1 != localSizes[2]) {
    throw std::runtime_error("Bad local sizes of kernel.");
  }

  size_t extendedPointsSize = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    extendedPointsSize += align(buckets.points[i].size(), localSize);
  }

  size_t globalSize = align(extendedPointsSize, localSize);

  size_t totalVerticesNum = 0;
  for (size_t i = 0; i < buckets.raw.size(); i++) {
    totalVerticesNum += buckets.raw[i].size();
  }

  std::vector<cl_float2> vertices(totalVerticesNum);
  for (size_t i = 0, offset = 0; i < buckets.raw.size(); i++) {
    std::memcpy(vertices.data() + offset, buckets.raw[i].data(),
                sizeof(Coord) * buckets.raw[i].size());
    offset += buckets.raw[i].size();
  }
  cl_mem verticesBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_float2) * totalVerticesNum, vertices.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  std::vector<cl_uint> polygonOffsets(buckets.raw.size() + 1);
  polygonOffsets[0] = 0;
  for (size_t i = 0; i < buckets.raw.size(); i++) {
    polygonOffsets[i + 1] = polygonOffsets[i] + buckets.raw[i].size();
  }
  cl_mem polygonOffsetsBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_uint) * polygonOffsets.size(), polygonOffsets.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  size_t totalBucketPolygonsNum = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    totalBucketPolygonsNum += buckets.polygons[i].size();
  }

  std::vector<cl_uint> bucketPolygons(totalBucketPolygonsNum);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    for (size_t j = 0; j < buckets.polygons[i].size(); j++) {
      bucketPolygons[offset + j] = buckets.polygons[i][j];
    }
    offset += buckets.polygons[i].size();
  }
  cl_mem bucketPolygonsBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_uint) * bucketPolygons.size(), bucketPolygons.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  std::vector<cl_uint> bucketPolygonsOffsets(bucketsNum + 1);
  bucketPolygonsOffsets[0] = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    bucketPolygonsOffsets[i + 1] = bucketPolygonsOffsets[i] + buckets.polygons[i].size();
  }
  cl_mem bucketPolygonsOffsetsBuf = clCreateBuffer(
      dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(cl_uint) * bucketPolygonsOffsets.size(), bucketPolygonsOffsets.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  std::vector<cl_float2> allPoints(extendedPointsSize);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    std::memcpy(allPoints.data() + offset, buckets.points[i].data(),
                sizeof(Coord) * buckets.points[i].size());
    offset += align(buckets.points[i].size(), localSize);
  }
  cl_mem allPointsBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_float2) * allPoints.size(), allPoints.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  std::vector<cl_uint> pointBucket(extendedPointsSize);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    cl_uint *curOff = pointBucket.data() + offset;
    cl_uint curSize = buckets.points[i].size();
    std::fill(curOff, curOff + curSize, i);
    std::fill(curOff + curSize, curOff + align(curSize, localSize), bucketsNum);
    offset += align(buckets.points[i].size(), localSize);
  }
  cl_mem pointBucketBuf =
      clCreateBuffer(dev.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_uint) * pointBucket.size(), pointBucket.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);


  size_t intersectionSize = allPoints.size() * RES_SIZE;
  cl_mem intersectionBuf = clCreateBuffer(dev.getContext(), CL_MEM_READ_WRITE,
                                          sizeof(cl_uint) * intersectionSize, nullptr, &clStatus);
  CHECK_OPENCL_STATUS(clStatus);

  int arg = 0;
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &verticesBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &polygonOffsetsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &bucketPolygonsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &bucketPolygonsOffsetsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &allPointsBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &pointBucketBuf));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_uint), &extendedPointsSize));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_uint), &bucketsNum));
  CHECK_OPENCL_CALL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &intersectionBuf));

  std::vector<cl_uint> intersectionTmp(intersectionSize);
  CHECK_OPENCL_CALL(clFinish(dev.getQueue()));

  Timer timer;
  timer.start();
  CHECK_OPENCL_CALL(clEnqueueNDRangeKernel(dev.getQueue(), kernel, 1, nullptr, &globalSize,
                                           &localSize, 0, nullptr, nullptr));
  CHECK_OPENCL_CALL(clEnqueueReadBuffer(dev.getQueue(), intersectionBuf, CL_TRUE, 0,
                                        sizeof(cl_uint) * intersectionSize, intersectionTmp.data(),
                                        0, nullptr, nullptr));
  timer.end();
  std::cout << "[KERNEL] " << timer << std::endl;

  for (size_t i = 0, offset1 = 0, offset2 = 0; i < bucketsNum; i++) {
    std::memcpy(intersection + offset1, intersectionTmp.data() + offset2,
                sizeof(cl_uint) * buckets.points[i].size() * RES_SIZE);
    offset1 += buckets.points[i].size() * RES_SIZE;
    offset2 += align(buckets.points[i].size(), localSize) * RES_SIZE;
  }

  CHECK_OPENCL_CALL(clReleaseMemObject(intersectionBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(pointBucketBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(allPointsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(bucketPolygonsOffsetsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(bucketPolygonsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(polygonOffsetsBuf));
  CHECK_OPENCL_CALL(clReleaseMemObject(verticesBuf));

  CHECK_OPENCL_CALL(clReleaseKernel(kernel));
}


void cuda_intersectInBuckets_bucket(Device device_, const Buckets &buckets, uint32_t *intersection){
  std::cout << "CUDA: Bucket by thread" << std::endl;
  int device = std::get<int>(device_);
  CHECK_CUDA_STATUS(cudaSetDevice(device));
  cudaDeviceProp prop;
  CHECK_CUDA_STATUS(cudaGetDeviceProperties(&prop, device));
  std::cout << "CUDA: Running on " << prop.name << std::endl;

  uint32_t bucketsNum = (uint32_t)buckets.points.size();

  uint32_t threads = 64;

  size_t totalPointsNum = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    totalPointsNum += buckets.points[i].size();
  }
  uint32_t blocks = (uint32_t)divUp(bucketsNum, threads);

  size_t totalVerticesNum = 0;
  for (size_t i = 0; i < buckets.raw.size(); i++) {
    totalVerticesNum += buckets.raw[i].size();
  }

  std::vector<cl_float2> vertices(totalVerticesNum);
  for (size_t i = 0, offset = 0; i < buckets.raw.size(); i++) {
    std::memcpy(vertices.data() + offset, buckets.raw[i].data(),
                sizeof(Coord) * buckets.raw[i].size());
    offset += buckets.raw[i].size();
  }
  float2* verticesBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&verticesBuf, sizeof(cl_float2) * totalVerticesNum));
  CHECK_CUDA_STATUS(cudaMemcpy(verticesBuf, vertices.data(),
    sizeof(cl_float2) * totalVerticesNum, cudaMemcpyHostToDevice));

  std::vector<cl_uint> polygonOffsets(buckets.raw.size() + 1);
  polygonOffsets[0] = 0;
  for (size_t i = 0; i < buckets.raw.size(); i++) {
    polygonOffsets[i + 1] = polygonOffsets[i] + buckets.raw[i].size();
  }
  uint32_t* polygonOffsetsBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&polygonOffsetsBuf, sizeof(cl_uint) * polygonOffsets.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(polygonOffsetsBuf, polygonOffsets.data(),
    sizeof(cl_uint) * polygonOffsets.size(), cudaMemcpyHostToDevice));

  size_t totalBucketPolygonsNum = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    totalBucketPolygonsNum += buckets.polygons[i].size();
  }

  std::vector<cl_uint> bucketPolygons(totalBucketPolygonsNum);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    for (size_t j = 0; j < buckets.polygons[i].size(); j++) {
      bucketPolygons[offset + j] = buckets.polygons[i][j];
    }
    offset += buckets.polygons[i].size();
  }
  uint32_t* bucketPolygonsBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&bucketPolygonsBuf, sizeof(cl_uint) * bucketPolygons.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(bucketPolygonsBuf, bucketPolygons.data(),
    sizeof(cl_uint) * bucketPolygons.size(), cudaMemcpyHostToDevice));

  std::vector<cl_uint> bucketPolygonsOffsets(bucketsNum + 1);
  bucketPolygonsOffsets[0] = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    bucketPolygonsOffsets[i + 1] = bucketPolygonsOffsets[i] + buckets.polygons[i].size();
  }
  uint32_t* bucketPolygonsOffsetsBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&bucketPolygonsOffsetsBuf, sizeof(cl_uint) * bucketPolygonsOffsets.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(bucketPolygonsOffsetsBuf, bucketPolygonsOffsets.data(),
    sizeof(cl_uint) * bucketPolygonsOffsets.size(), cudaMemcpyHostToDevice));

  std::vector<cl_float2> allPoints(totalPointsNum);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    std::memcpy(allPoints.data() + offset, buckets.points[i].data(),
                sizeof(Coord) * buckets.points[i].size());
    offset += buckets.points[i].size();
  }
  float2* allPointsBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&allPointsBuf, sizeof(cl_float2) * allPoints.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(allPointsBuf, allPoints.data(),
    sizeof(cl_float2) * allPoints.size(), cudaMemcpyHostToDevice));

  std::vector<cl_uint> bucketPointsOffsets(bucketsNum + 1);
  bucketPointsOffsets[0] = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    bucketPointsOffsets[i + 1] = bucketPointsOffsets[i] + buckets.points[i].size();
  }
  uint32_t *bucketPointsOffsetsBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&bucketPointsOffsetsBuf, sizeof(cl_uint) * bucketPointsOffsets.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(bucketPointsOffsetsBuf, bucketPointsOffsets.data(),
    sizeof(cl_uint) * bucketPointsOffsets.size(), cudaMemcpyHostToDevice));


  size_t intersectionSize = allPoints.size() * RES_SIZE;
  uint32_t *intersectionBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&intersectionBuf, sizeof(cl_uint) * intersectionSize));
  CHECK_CUDA_STATUS(cudaDeviceSynchronize());

  Timer timer;
  timer.start();
  runIntersectAllPointsInBucketFullBucket(blocks, threads, verticesBuf, polygonOffsetsBuf, bucketPolygonsBuf,
    bucketPolygonsOffsetsBuf, allPointsBuf, bucketPointsOffsetsBuf, bucketsNum, intersectionBuf);
  CHECK_CUDA_STATUS(cudaMemcpy(intersection, intersectionBuf, sizeof(cl_uint) * intersectionSize, cudaMemcpyDeviceToHost));
  timer.end();
  std::cout << "[KERNEL] " << timer << std::endl;

  CHECK_CUDA_STATUS(cudaFree(intersectionBuf));
  CHECK_CUDA_STATUS(cudaFree(bucketPointsOffsetsBuf));
  CHECK_CUDA_STATUS(cudaFree(allPointsBuf));
  CHECK_CUDA_STATUS(cudaFree(bucketPolygonsOffsetsBuf));
  CHECK_CUDA_STATUS(cudaFree(bucketPolygonsBuf));
  CHECK_CUDA_STATUS(cudaFree(polygonOffsetsBuf));
  CHECK_CUDA_STATUS(cudaFree(verticesBuf));
}

void cuda_intersectInBuckets(Device device_, const Buckets &buckets, uint32_t *intersection) {
  std::cout << "CUDA: Point by thread" << std::endl;
  int device = std::get<int>(device_);
  CHECK_CUDA_STATUS(cudaSetDevice(device));
  cudaDeviceProp prop;
  CHECK_CUDA_STATUS(cudaGetDeviceProperties(&prop, device));
  std::cout << "CUDA: Running on " << prop.name << std::endl;

  uint32_t bucketsNum = (uint32_t)buckets.points.size();

  uint32_t threads = 64;

  size_t extendedPointsSize = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    extendedPointsSize += align(buckets.points[i].size(), threads);
  }
  size_t blocks = (uint32_t)align(extendedPointsSize, threads);

  size_t totalVerticesNum = 0;
  for (size_t i = 0; i < buckets.raw.size(); i++) {
    totalVerticesNum += buckets.raw[i].size();
  }

  std::vector<cl_float2> vertices(totalVerticesNum);
  for (size_t i = 0, offset = 0; i < buckets.raw.size(); i++) {
    std::memcpy(vertices.data() + offset, buckets.raw[i].data(),
                sizeof(Coord) * buckets.raw[i].size());
    offset += buckets.raw[i].size();
  }
  float2 *verticesBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&verticesBuf, sizeof(cl_float2) * totalVerticesNum));
  CHECK_CUDA_STATUS(cudaMemcpy(verticesBuf, vertices.data(), sizeof(cl_float2) * totalVerticesNum,
                               cudaMemcpyHostToDevice));

  std::vector<cl_uint> polygonOffsets(buckets.raw.size() + 1);
  polygonOffsets[0] = 0;
  for (size_t i = 0; i < buckets.raw.size(); i++) {
    polygonOffsets[i + 1] = polygonOffsets[i] + buckets.raw[i].size();
  }
  uint32_t *polygonOffsetsBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&polygonOffsetsBuf, sizeof(cl_uint) * polygonOffsets.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(polygonOffsetsBuf, polygonOffsets.data(),
                               sizeof(cl_uint) * polygonOffsets.size(), cudaMemcpyHostToDevice));

  size_t totalBucketPolygonsNum = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    totalBucketPolygonsNum += buckets.polygons[i].size();
  }

  std::vector<cl_uint> bucketPolygons(totalBucketPolygonsNum);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    for (size_t j = 0; j < buckets.polygons[i].size(); j++) {
      bucketPolygons[offset + j] = buckets.polygons[i][j];
    }
    offset += buckets.polygons[i].size();
  }
  uint32_t *bucketPolygonsBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&bucketPolygonsBuf, sizeof(cl_uint) * bucketPolygons.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(bucketPolygonsBuf, bucketPolygons.data(),
                               sizeof(cl_uint) * bucketPolygons.size(), cudaMemcpyHostToDevice));

  std::vector<cl_uint> bucketPolygonsOffsets(bucketsNum + 1);
  bucketPolygonsOffsets[0] = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    bucketPolygonsOffsets[i + 1] = bucketPolygonsOffsets[i] + buckets.polygons[i].size();
  }
  uint32_t *bucketPolygonsOffsetsBuf;
  CHECK_CUDA_STATUS(
      cudaMalloc(&bucketPolygonsOffsetsBuf, sizeof(cl_uint) * bucketPolygonsOffsets.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(bucketPolygonsOffsetsBuf, bucketPolygonsOffsets.data(),
                               sizeof(cl_uint) * bucketPolygonsOffsets.size(),
                               cudaMemcpyHostToDevice));

  std::vector<cl_float2> allPoints(extendedPointsSize);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    std::memcpy(allPoints.data() + offset, buckets.points[i].data(),
                sizeof(Coord) * buckets.points[i].size());
    offset += align(buckets.points[i].size(), threads);
  }
  float2 *allPointsBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&allPointsBuf, sizeof(cl_float2) * allPoints.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(allPointsBuf, allPoints.data(), sizeof(cl_float2) * allPoints.size(),
                               cudaMemcpyHostToDevice));

  std::vector<cl_uint> pointBucket(extendedPointsSize);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    cl_uint *curOff = pointBucket.data() + offset;
    cl_uint curSize = buckets.points[i].size();
    std::fill(curOff, curOff + curSize, i);
    std::fill(curOff + curSize, curOff + align(curSize, threads), bucketsNum);
    offset += align(buckets.points[i].size(), threads);
  }
  uint32_t *pointBucketBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&pointBucketBuf, sizeof(cl_uint) * pointBucket.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(pointBucketBuf, pointBucket.data(),
                               sizeof(cl_uint) * pointBucket.size(), cudaMemcpyHostToDevice));

  size_t intersectionSize = allPoints.size() * RES_SIZE;
  uint32_t *intersectionBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&intersectionBuf, sizeof(cl_uint) * intersectionSize));
  CHECK_CUDA_STATUS(cudaDeviceSynchronize());

  std::vector<cl_uint> intersectionTmp(intersectionSize);
  Timer timer;
  timer.start();
  runIntersectAllPointsInBucket(blocks, threads, verticesBuf, polygonOffsetsBuf, bucketPolygonsBuf,
                                bucketPolygonsOffsetsBuf, allPointsBuf, pointBucketBuf,
                                extendedPointsSize, bucketsNum, intersectionBuf);
  CHECK_CUDA_STATUS(cudaMemcpy(intersectionTmp.data(), intersectionBuf,
                               sizeof(cl_uint) * intersectionSize, cudaMemcpyDeviceToHost));
  timer.end();
  std::cout << "[KERNEL] " << timer << std::endl;

  for (size_t i = 0, offset1 = 0, offset2 = 0; i < bucketsNum; i++) {
    std::memcpy(intersection + offset1, intersectionTmp.data() + offset2,
                sizeof(cl_uint) * buckets.points[i].size() * RES_SIZE);
    offset1 += buckets.points[i].size() * RES_SIZE;
    offset2 += align(buckets.points[i].size(), threads) * RES_SIZE;
  }

  CHECK_CUDA_STATUS(cudaFree(intersectionBuf));
  CHECK_CUDA_STATUS(cudaFree(pointBucketBuf));
  CHECK_CUDA_STATUS(cudaFree(allPointsBuf));
  CHECK_CUDA_STATUS(cudaFree(bucketPolygonsOffsetsBuf));
  CHECK_CUDA_STATUS(cudaFree(bucketPolygonsBuf));
  CHECK_CUDA_STATUS(cudaFree(polygonOffsetsBuf));
  CHECK_CUDA_STATUS(cudaFree(verticesBuf));
}

void cuda_intersectInBuckets_noDiv(Device device_, const Buckets &buckets, uint32_t *intersection) {
  std::cout << "CUDA: Point by thread + avoid div optimization" << std::endl;
  int device = std::get<int>(device_);
  CHECK_CUDA_STATUS(cudaSetDevice(device));
  cudaDeviceProp prop;
  CHECK_CUDA_STATUS(cudaGetDeviceProperties(&prop, device));
  std::cout << "CUDA: Running on " << prop.name << std::endl;

  uint32_t bucketsNum = (uint32_t)buckets.points.size();

  uint32_t threads = 64;

  size_t extendedPointsSize = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    extendedPointsSize += align(buckets.points[i].size(), threads);
  }
  size_t blocks = (uint32_t)align(extendedPointsSize, threads);

  size_t totalVerticesNum = 0;
  for (size_t i = 0; i < buckets.raw.size(); i++) {
    totalVerticesNum += buckets.raw[i].size();
  }

  std::vector<cl_float2> vertices(totalVerticesNum);
  for (size_t i = 0, offset = 0; i < buckets.raw.size(); i++) {
    std::memcpy(vertices.data() + offset, buckets.raw[i].data(),
                sizeof(Coord) * buckets.raw[i].size());
    offset += buckets.raw[i].size();
  }
  float2 *verticesBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&verticesBuf, sizeof(cl_float2) * totalVerticesNum));
  CHECK_CUDA_STATUS(cudaMemcpy(verticesBuf, vertices.data(), sizeof(cl_float2) * totalVerticesNum,
                               cudaMemcpyHostToDevice));

  std::vector<cl_uint> polygonOffsets(buckets.raw.size() + 1);
  polygonOffsets[0] = 0;
  for (size_t i = 0; i < buckets.raw.size(); i++) {
    polygonOffsets[i + 1] = polygonOffsets[i] + buckets.raw[i].size();
  }
  uint32_t *polygonOffsetsBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&polygonOffsetsBuf, sizeof(cl_uint) * polygonOffsets.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(polygonOffsetsBuf, polygonOffsets.data(),
                               sizeof(cl_uint) * polygonOffsets.size(), cudaMemcpyHostToDevice));

  size_t totalBucketPolygonsNum = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    totalBucketPolygonsNum += buckets.polygons[i].size();
  }

  std::vector<cl_uint> bucketPolygons(totalBucketPolygonsNum);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    for (size_t j = 0; j < buckets.polygons[i].size(); j++) {
      bucketPolygons[offset + j] = buckets.polygons[i][j];
    }
    offset += buckets.polygons[i].size();
  }
  uint32_t *bucketPolygonsBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&bucketPolygonsBuf, sizeof(cl_uint) * bucketPolygons.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(bucketPolygonsBuf, bucketPolygons.data(),
                               sizeof(cl_uint) * bucketPolygons.size(), cudaMemcpyHostToDevice));

  std::vector<cl_uint> bucketPolygonsOffsets(bucketsNum + 1);
  bucketPolygonsOffsets[0] = 0;
  for (size_t i = 0; i < bucketsNum; i++) {
    bucketPolygonsOffsets[i + 1] = bucketPolygonsOffsets[i] + buckets.polygons[i].size();
  }
  uint32_t *bucketPolygonsOffsetsBuf;
  CHECK_CUDA_STATUS(
      cudaMalloc(&bucketPolygonsOffsetsBuf, sizeof(cl_uint) * bucketPolygonsOffsets.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(bucketPolygonsOffsetsBuf, bucketPolygonsOffsets.data(),
                               sizeof(cl_uint) * bucketPolygonsOffsets.size(),
                               cudaMemcpyHostToDevice));

  std::vector<cl_float2> allPoints(extendedPointsSize);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    std::memcpy(allPoints.data() + offset, buckets.points[i].data(),
                sizeof(Coord) * buckets.points[i].size());
    offset += align(buckets.points[i].size(), threads);
  }
  float2 *allPointsBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&allPointsBuf, sizeof(cl_float2) * allPoints.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(allPointsBuf, allPoints.data(), sizeof(cl_float2) * allPoints.size(),
                               cudaMemcpyHostToDevice));

  std::vector<cl_uint> pointBucket(extendedPointsSize);
  for (size_t i = 0, offset = 0; i < bucketsNum; i++) {
    cl_uint *curOff = pointBucket.data() + offset;
    cl_uint curSize = buckets.points[i].size();
    std::fill(curOff, curOff + curSize, i);
    std::fill(curOff + curSize, curOff + align(curSize, threads), bucketsNum);
    offset += align(buckets.points[i].size(), threads);
  }
  uint32_t *pointBucketBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&pointBucketBuf, sizeof(cl_uint) * pointBucket.size()));
  CHECK_CUDA_STATUS(cudaMemcpy(pointBucketBuf, pointBucket.data(),
                               sizeof(cl_uint) * pointBucket.size(), cudaMemcpyHostToDevice));

  size_t intersectionSize = allPoints.size() * RES_SIZE;
  uint32_t *intersectionBuf;
  CHECK_CUDA_STATUS(cudaMalloc(&intersectionBuf, sizeof(cl_uint) * intersectionSize));
  CHECK_CUDA_STATUS(cudaDeviceSynchronize());

  std::vector<cl_uint> intersectionTmp(intersectionSize);
  Timer timer;
  timer.start();
  runIntersectAllPointsInBucketNoDiv(
      blocks, threads, verticesBuf, polygonOffsetsBuf, bucketPolygonsBuf, bucketPolygonsOffsetsBuf,
      allPointsBuf, pointBucketBuf, extendedPointsSize, bucketsNum, intersectionBuf);
  CHECK_CUDA_STATUS(cudaMemcpy(intersectionTmp.data(), intersectionBuf,
                               sizeof(cl_uint) * intersectionSize, cudaMemcpyDeviceToHost));
  timer.end();
  std::cout << "[KERNEL] " << timer << std::endl;

  for (size_t i = 0, offset1 = 0, offset2 = 0; i < bucketsNum; i++) {
    std::memcpy(intersection + offset1, intersectionTmp.data() + offset2,
                sizeof(cl_uint) * buckets.points[i].size() * RES_SIZE);
    offset1 += buckets.points[i].size() * RES_SIZE;
    offset2 += align(buckets.points[i].size(), threads) * RES_SIZE;
  }

  CHECK_CUDA_STATUS(cudaFree(intersectionBuf));
  CHECK_CUDA_STATUS(cudaFree(pointBucketBuf));
  CHECK_CUDA_STATUS(cudaFree(allPointsBuf));
  CHECK_CUDA_STATUS(cudaFree(bucketPolygonsOffsetsBuf));
  CHECK_CUDA_STATUS(cudaFree(bucketPolygonsBuf));
  CHECK_CUDA_STATUS(cudaFree(polygonOffsetsBuf));
  CHECK_CUDA_STATUS(cudaFree(verticesBuf));
}
