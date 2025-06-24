#include "clUtils.h"

#include <iostream>
#include <vector>

void printBuildLog(cl_program program, cl_device_id device) {
  size_t size = 0;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
  if (size > 2) {
    std::vector<char> buildLog(size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, buildLog.data(), nullptr);
    std::cerr << "Build log:\n";
    std::cerr << "============\n";
    std::cerr << buildLog.data();
    std::cerr << "============";
  } else {
    std::cerr << "Build log is empty";
  }
  std::cerr << "\n";
}


std::vector<cl_device_id> getClDevices() {
  cl_uint nPlatforms = 0;
  std::vector<cl_device_id> allDevices;
  cl_int clRet = clGetPlatformIDs(0, nullptr, &nPlatforms);
  if (CL_SUCCESS != clRet || 0 == nPlatforms) {
    std::cerr << "No compatible OpenCL platforms found" << std::endl;
    return allDevices;
  }
  std::vector<cl_platform_id> platforms(nPlatforms);
  clGetPlatformIDs(nPlatforms, platforms.data(), nullptr);
  for (cl_uint i = 0; i < nPlatforms; i++) {
    cl_uint nDevices = 0;
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &nDevices);
    std::vector<cl_device_id> devices(nDevices);
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, nDevices, devices.data(), 0);
    allDevices.insert(allDevices.end(), devices.begin(), devices.end());
  }
  return allDevices;
}


size_t divUp(size_t numerator, size_t denominator) {
  if (0 == denominator) {
    throw std::runtime_error("Denominator is zero.");
  }
  return (numerator + denominator - 1) / denominator;
}

size_t align(size_t value, size_t alignment) {
  if (0 == alignment) {
    throw std::runtime_error("Alignment is zero.");
  }
  return divUp(value, alignment) * alignment;
}
