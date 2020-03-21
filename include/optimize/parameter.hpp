#pragma once

namespace vllm
{
namespace optimize
{
struct Parameter {
  float scale_gain = 0;
  float latitude_gain = 0;
  float altitude_gain = 0;
  float smooth_gain = 0;
};
}  // namespace optimize
}  // namespace vllm