#include "vllm/optimize/types_restriction.hpp"

namespace vllm
{
namespace optimize
{
void Edge_Scale_Restriction::computeError()
{
  const VertexSim3Expmap* vp0 = static_cast<const VertexSim3Expmap*>(_vertices[0]);
  double scale = vp0->estimate().scale();
  const double ref_scale = measurement();

  // TODO:
  _error(0) = gain * (ref_scale - scale);
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

  double swing = (R * ez).z();
  if (swing > 0.25)
    _error(0) = 1e4;
  else
    _error(0) = gain * swing;
}

// void Edge_Smooth_Restriction::computeError()
// {
// TODO:
// const VertexSim3Expmap* vp0 = static_cast<const VertexSim3Expmap*>(_vertices[0]);

// double min=
// Eigen::Vector3f t0 = measurement().at(0);
// for (size_t i = 1, N = measurement().size(); i < N - 1; i++) {
//   Eigen::Vector3f t1 = measurement().at(i);

//   t0 = t1;
// }
// Eigen::Vector3d older = measurement().older_pos;
// Eigen::Vector3d old = measurement().old_pos;
// Eigen::Vector3d now = vp0->estimate().map(measurement().camera_pos);

// double old_vel = (old - older).norm();
// double vel = (now - old).norm();

// double thr = 0.1;
// double dv = vel - old_vel;
// // Usually, the larger the velocity, the better
// if (dv < thr)
//   _error(0) = gain * (thr - dv) + 0.5 * (vel - old_vel);
// else
//   _error(0) = 0.5 * (vel - old_vel);
// }

}  // namespace optimize
}  // namespace vllm