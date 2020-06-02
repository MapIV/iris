#pragma once
#include <Eigen/Dense>

namespace iris
{
namespace map
{
struct Info {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  float x;
  float y;
  float theta;

  Info() {}
  Info(float x, float y, float theta) : x(x), y(y), theta(theta) {}

  Eigen::Vector2f xy() const { return Eigen::Vector2f(x, y); }

  std::string toString() const
  {
    return std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(theta);
  }

  bool isEqual(const Info& a, const Info& b) const
  {
    const float epsilon = 1e-6f;
    if (std::fabs(a.x - b.x) > epsilon)
      return false;
    if (std::fabs(a.y - b.y) > epsilon)
      return false;
    if (std::fabs(a.theta - b.theta) > epsilon)
      return false;
    return true;
  }

  bool operator==(const Info& other) const
  {
    return isEqual(*this, other);
  }
  bool operator!=(const Info& other) const
  {
    return !isEqual(*this, other);
  }
};

}  // namespace map
}  // namespace iris