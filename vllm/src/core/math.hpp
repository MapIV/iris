#pragma once
#include <Eigen/Dense>

namespace vllm
{
namespace so3
{
Eigen::Matrix3f hat(const Eigen::Vector3f& xi);

Eigen::Vector3f log(const Eigen::Matrix3f& R);

Eigen::Matrix3f exp(const Eigen::Vector3f& xi);
}  // namespace so3
}  // namespace vllm