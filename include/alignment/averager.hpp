#pragma once
#include "core/util.hpp"
#include <Eigen/Dense>

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

Eigen::Vector3f calcAverageTransform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t, int n)
{
  Eigen::Matrix3f A = Eigen::Matrix3f::Zero();

  for (int i = 0; i < n; i++) {
    Eigen::Matrix3f tmp = Eigen::Matrix3f::Identity();
    for (int j = 0; j < i; j++) {
      tmp = R * tmp;
    }
    A += tmp;
  }

  return A.inverse() * t;
}

Eigen::Matrix4f calcVelocity(const std::list<Eigen::Matrix4f>& poses)
{
  Eigen::Matrix4f V = Eigen::Matrix4f::Identity();

  const int dt = static_cast<int>(poses.size()) - 2;

  Eigen::Matrix4f T0 = getNormalizedPose(*std::next(poses.begin()));
  Eigen::Matrix4f Tn = getNormalizedPose(*std::prev(poses.end()));

  V = T0 * Tn.inverse();
  Eigen::Matrix3f R = V.topLeftCorner(3, 3);
  Eigen::Vector3f t = V.topRightCorner(3, 1);
  Eigen::Matrix3f root_R = so3::exp(so3::log(R) / dt);
  V.topLeftCorner(3, 3) = root_R;
  V.topRightCorner(3, 1) = calcAverageTransform(root_R, t, dt);
  return V;
}
}  // namespace vllm