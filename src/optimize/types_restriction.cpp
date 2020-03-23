#include "optimize/types_restriction.hpp"

namespace vllm
{
namespace optimize
{
void Edge_Scale_Restriction::computeError()
{
  const VertexSim3Expmap* vp0 = static_cast<const VertexSim3Expmap*>(_vertices[0]);
  double scale = vp0->estimate().scale();

  Eigen::Vector2d scales = measurement();

  double ds = scale - scales(0);
  if (ds < 0.01)
    _error(0) = 10 * ds + gain * (scale - 2 * scales(0) + scales(1));
  else
    _error(0) = gain * (scale - 2 * scales(0) + scales(1));
}

void Edge_Altitude_Restriction::computeError()
{
  const VertexSim3Expmap* vp0 = static_cast<const VertexSim3Expmap*>(_vertices[0]);
  Eigen::Vector3d now = vp0->estimate().map(measurement());
  _error(0) = gain * now.z();
}

void Edge_Latitude_Restriction::computeError()
{
  const VertexSim3Expmap* vp0 = static_cast<const VertexSim3Expmap*>(_vertices[0]);
  Eigen::Matrix3d R = vp0->estimate().rotation().toRotationMatrix();
  Eigen::Vector3d ez(0, 0, 1);

  double swing = (ez - R * ez).norm();
  if (swing > 0.2)
    _error(0) = 1e4;
  else
    _error(0) = gain * swing;
}

void Edge_Smooth_Restriction::computeError()
{
  const VertexSim3Expmap* vp0 = static_cast<const VertexSim3Expmap*>(_vertices[0]);

  Eigen::Vector3d older = measurement().older_pos;
  Eigen::Vector3d old = measurement().old_pos;
  Eigen::Vector3d now = vp0->estimate().map(measurement().camera_pos);

  Eigen::Vector3d dx = now - old - old + older;
  _error(0) = gain * dx.norm();
}

}  // namespace optimize
}  // namespace vllm