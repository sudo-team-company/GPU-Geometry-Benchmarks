#ifndef TIMER
#define TIMER

#include <chrono>
#include <ostream>

class Timer {
public:
  void start() { start_ = std::chrono::steady_clock::now(); }
  void end() { end_ = std::chrono::steady_clock::now(); }

  long seconds() const {
    return std::chrono::duration_cast<std::chrono::seconds>(end_ - start_).count();
  }

  long milliseconds() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
  }

  long microseconds() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count();
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::chrono::time_point<std::chrono::steady_clock> end_;
};

std::ostream &operator<<(std::ostream &os, const Timer &timer);

#endif // TIMER
