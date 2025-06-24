#include "timer.h"

std::ostream &operator<<(std::ostream &os, const Timer &timer) {
  os << "Execution time: " << timer.seconds() << " s | " << timer.milliseconds() << " ms | "
     << timer.microseconds() << " μs";
  return os;
}
