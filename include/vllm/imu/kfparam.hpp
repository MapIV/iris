#pragma once

namespace vllm
{
struct KFParam {
  float initial_cov_p;      // [m]
  float initial_cov_v;      // [m/s]
  float initial_cov_theta;  // [rad]
  float initial_cov_grad;   // [m/s^2]
  float initial_cov_bias;   // [m/s^2]

  float drive_cov_v;      // [m/s]
  float drive_cov_theta;  // [rad]
  float drive_cov_bias;   // [m/s^2]

  float observe_cov_p;      // [m]
  float observe_cov_theta;  // [rad]
};

}  // namespace vllm