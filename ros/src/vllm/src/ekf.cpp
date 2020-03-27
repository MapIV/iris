#include "ekf.hpp"

namespace vllm
{

Eigen::Quaternionf EKF::exp(const Eigen::Vector3f& v)
{
  float norm = v.norm();
  float c = std::cos(norm / 2);
  float s = std::sin(norm / 2);
  Eigen::Vector3f n = s * v.normalized();
  return Eigen::Quaternionf(c, n.x(), n.y(), n.z());
}


}  // namespace vllm