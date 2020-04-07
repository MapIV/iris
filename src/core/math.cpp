#include "vllm/core/math.hpp"

namespace vllm
{
namespace so3
{
Eigen::Matrix3f hat(const Eigen::Vector3f& xi)
{
  Eigen::Matrix3f S;
  // clang-format off
  S <<  0,   -xi(2), xi(1),
       xi(2), 0,    -xi(0),
      -xi(1), xi(0), 0;
  // clang-format on
  return S;
}

Eigen::Vector3f log(const Eigen::Matrix3f& R)
{
  Eigen::Vector3f xi = Eigen::Vector3f::Zero();
  float w_length = static_cast<float>(std::acos((R.trace() - 1.0f) * 0.5f));
  if (w_length > 1e-6f) {
    Eigen::Vector3f tmp;
    tmp << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);
    xi = 1.0f / (2.0f * static_cast<float>(std::sin(w_length))) * tmp * w_length;
  }
  return xi;
}

Eigen::Matrix3f exp(const Eigen::Vector3f& xi)
{
  float theta = xi.norm();
  Eigen::Vector3f axis = xi.normalized();

  float cos = std::cos(theta);
  float sin = std::sin(theta);

  auto tmp1 = cos * Eigen::Matrix3f::Identity();
  auto tmp2 = (1 - cos) * (axis * axis.transpose());
  auto tmp3 = sin * hat(axis);
  return tmp1 + tmp2 + tmp3;
}
}  // namespace so3
}  // namespace vllm