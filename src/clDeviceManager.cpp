#include "clDeviceManager.h"

#include "clUtils.h"

clDeviceManager::clDeviceManager(cl_device_id device) {
  cl_int clStatus = CL_SUCCESS;
  device_ = device;
  context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &clStatus);
  CHECK_OPENCL_STATUS(clStatus);
  queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &clStatus);
  CHECK_OPENCL_STATUS(clStatus);
}

clDeviceManager::~clDeviceManager() {
  if (nullptr != queue_) {
    clReleaseCommandQueue(queue_);
    queue_ = nullptr;
  }
  if (nullptr != context_) {
    clReleaseContext(context_);
    context_ = nullptr;
  }
  device_ = nullptr;
}

std::string clDeviceManager::getDeviceName() const {
  if (nullptr == device_) {
    return "";
  }
  size_t nameSize = 0;
  CHECK_OPENCL_CALL(clGetDeviceInfo(device_, CL_DEVICE_NAME, 0, nullptr, &nameSize));
  std::vector<char> name(nameSize);
  CHECK_OPENCL_CALL(clGetDeviceInfo(device_, CL_DEVICE_NAME, name.size(), name.data(), nullptr));
  return std::string(name.data());
}

cl_program clDeviceManager::loadProgram(std::string &source) const {
  cl_int clStatus = CL_SUCCESS;
  const char *str = reinterpret_cast<char *>(source.data());
  const size_t strSize = source.size();
  cl_program program = clCreateProgramWithSource(context_, 1, &str, &strSize, &clStatus);
  CHECK_OPENCL_STATUS(clStatus);
  clStatus = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
  if (CL_SUCCESS != clStatus) {
    printBuildLog(program, device_);
  }
  CHECK_OPENCL_STATUS(clStatus);
  return program;
}

cl_kernel clDeviceManager::loadKernel(cl_program program, const std::string &kernelName) const {
  cl_int clStatus = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, kernelName.data(), &clStatus);
  CHECK_OPENCL_STATUS(clStatus);
  return kernel;
}

cl_kernel clDeviceManager::loadKernel(std::string &source, const std::string &kernelName) const {
  cl_program program = loadProgram(source);
  cl_kernel kernel = loadKernel(program, kernelName);
  CHECK_OPENCL_CALL(clReleaseProgram(program));
  return kernel;
}
