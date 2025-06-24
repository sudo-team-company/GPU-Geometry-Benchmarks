#ifndef CL_DEVICE_MANAGER
#define CL_DEVICE_MANAGER

#include <CL/cl.h>
#include <string>

class clDeviceManager {
public:
  clDeviceManager(cl_device_id device);
  ~clDeviceManager();

  cl_device_id getDevice() { return device_; }
  cl_context getContext() { return context_; }
  cl_command_queue getQueue() { return queue_; }

  std::string getDeviceName() const;

  cl_program loadProgram(std::string &source) const;
  cl_kernel loadKernel(cl_program program, const std::string &kernelName) const;
  cl_kernel loadKernel(std::string &source, const std::string &kernelName) const;

private:
  cl_device_id device_;
  cl_context context_;
  cl_command_queue queue_;
};

#endif // CL_DEVICE_MANAGER
