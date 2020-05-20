#pragma once
#include "vllm/core/types.hpp"

namespace vllm
{
struct KeypointsWithNormal {
  pcXYZ::Ptr cloud;
  pcNormal::Ptr normals;

  KeypointsWithNormal() : cloud(new pcXYZ),
                          normals(new pcNormal)
  {
  }
};
}  // namespace vllm