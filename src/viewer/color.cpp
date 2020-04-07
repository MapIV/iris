#include "vllm/viewer/color.hpp"

namespace vllm
{
namespace viewer
{
Eigen::Vector3f convertRGB(Eigen::Vector3f hsv)
{
  const float max = hsv(2);
  const float min = max * (1 - hsv(1));
  const float H = hsv(0);
  const float D = max - min;
  if (H < 60) return {max, H / 60 * D + min, min};
  if (H < 120) return {(120 - H) / 60 * D + min, max, min};
  if (H < 180) return {min, max, (H - 120) / 60 * D + min};
  if (H < 240) return {min, (240 - H) / 60 * D + min, max};
  if (H < 300) return {(H - 240) / 60 * D + min, min, max};
  if (H < 360) return {max, min, (360 - H) / 60 * D + min};
  return {255, 255, 255};
}

}  // namespace viewer
}  // namespace vllm