#pragma once
#include "core/types.hpp"

namespace iris
{
struct KeypointsWithNormal {
  pcXYZ::Ptr cloud;
  pcNormal::Ptr normals;

  KeypointsWithNormal() : cloud(new pcXYZ),
                          normals(new pcNormal)
  {
  }
};
}  // namespace iris