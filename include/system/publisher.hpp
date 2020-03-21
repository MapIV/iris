#pragma once
#include "system/database.hpp"
#include <mutex>

namespace vllm
{
// thread safe publisher
class Publisher
{
private:
  Database pub[2];
  std::mutex mtx;
  int id = 0;
  bool flag[2] = {false, false};

public:
  void push(const Database& p)
  {
    pub[id] = p;
    flag[id] = true;

    std::lock_guard lock(mtx);
    id = (id + 1) % 2;
  }

  bool pop(Database& p)
  {
    std::lock_guard lock(mtx);
    if (flag[(id + 1) % 2] == false) {
      return false;
    }

    p = pub[(id + 1) % 2];
    flag[(id + 1) % 2] = false;
    return true;
  }
};
}  // namespace vllm