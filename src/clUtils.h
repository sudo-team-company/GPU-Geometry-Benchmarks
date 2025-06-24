#ifndef CL_UTILS
#define CL_UTILS

#include <CL/opencl.h>

#include <functional>
#include <sstream>

/// Check OpenCL status and throw exception if it failed.
#define CHECK_OPENCL_STATUS(ret)                                                                   \
  do {                                                                                             \
    cl_int clRet = ret;                                                                            \
    if (clRet != CL_SUCCESS) {                                                                     \
      std::stringstream ss;                                                                        \
      ss << "OpenCL call error " << clRet << "(" << __LINE__ << ")";                               \
      throw std::runtime_error(ss.str());                                                          \
    }                                                                                              \
  } while (false)


/// Check OpenCL call status and throw exception if it failed.
#define CHECK_OPENCL_CALL(ret)                                                                     \
  do {                                                                                             \
    cl_int clRet = ret;                                                                            \
    if (clRet != CL_SUCCESS) {                                                                     \
      std::stringstream ss;                                                                        \
      ss << "OpenCL call failed: (" << __LINE__ << ")" << #ret << " returned " << clRet;           \
      throw std::runtime_error(ss.str());                                                          \
    }                                                                                              \
  } while (false)


/// Prints OpenCL program build log.
void printBuildLog(cl_program program, cl_device_id device);


/// Returns all OpenCL devices.
std::vector<cl_device_id> getClDevices();


size_t divUp(size_t numerator, size_t denominator);

/// Aligns value with alignment
size_t align(size_t value, size_t alignment);

#endif // CL_UTILS
