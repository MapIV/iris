#pragma once
#include "util.hpp"
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

Eigen::Matrix4f calcVelocity(const std::list<Eigen::Matrix4f>& poses)
{
  Eigen::Matrix4f V = Eigen::Matrix4f::Identity();

  const int dt = poses.size() - 2;
  std::cout << "###### calcVelocity ##### dt= " << dt << std::endl;


  Eigen::Matrix4f T0 = *std::next(poses.begin());
  Eigen::Matrix4f Tn = *std::prev(poses.end());
  std::cout << T0 << "\n"
            << Tn << std::endl;

  Eigen::Matrix3f R0 = vllm::getNormalizedRotation(T0.topLeftCorner(3, 3));
  Eigen::Matrix3f Rn = vllm::getNormalizedRotation(Tn.topLeftCorner(3, 3));

  Eigen::Vector3f t0 = T0.topRightCorner(3, 1);
  Eigen::Vector3f tn = Tn.topRightCorner(3, 1);

  V.topRightCorner(3, 1) = (t0 - tn) / dt;
  V.topLeftCorner(3, 3) = so3::exp((so3::log(R0) - so3::log(Rn)) / dt);

  std::cout << V << std::endl;
  return V;
}

Eigen::Matrix4f entrywiseProduct(const Eigen::Matrix4f& T1, const Eigen::Matrix4f& T2)
{
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.topLeftCorner(3, 3) = T1.topLeftCorner(3, 3) * T2.topLeftCorner(3, 3);
  T.topRightCorner(3, 1) = T1.topRightCorner(3, 1) + T2.topRightCorner(3, 1);
  return T;
}

}  // namespace vllm