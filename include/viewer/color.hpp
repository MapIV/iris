#pragma once
#include <Eigen/Dense>

namespace vllm
{
namespace viewer
{
struct Color {
  float r;
  float g;
  float b;
  float size;
  Color() { r = g = b = size = 1.0f; }
  Color(float r, float g, float b, float s) : r(r), g(g), b(b), size(s) {}
};

// h[0,360],s[0,1],v[0,1]
Eigen::Vector3f convertRGB(Eigen::Vector3f hsv);

}  // namespace viewer
}  // namespace vllm