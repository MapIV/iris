#pragma once

namespace vllm
{
struct Color {
  float r;
  float g;
  float b;
  float size;
  Color() { r = g = b = size = 1.0f; }
  Color(float r, float g, float b, float s) : r(r), g(g), b(b), size(s) {}
};
}  // namespace vllm