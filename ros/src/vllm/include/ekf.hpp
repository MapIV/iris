#pragma once
#include <Eigen/Dense>

namespace vllm
{
// error state extended kalman filter
class EKF
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EKF() : gravity(0, 0, -9.8)
  {
    cov = Eigen::MatrixXf::Identity(9, 9);
    pos.setZero();
    vel.setZero();
    qua.setIdentity();

    V = Eigen::MatrixXf::Zero(9, 9);
    V.topLeftCorner(3, 3) = 0.1 * Eigen::Matrix3f::Identity(3, 3);
    V.block(3, 3, 3, 3) = 0.1 * Eigen::Matrix3f::Identity(3, 3);
    V.bottomRightCorner(3, 3) = 0.1 * Eigen::Matrix3f::Identity(3, 3);
  }
  EKF(const Eigen::Matrix4f& T) : EKF()
  {
    init(T);
  }

  void init(const Eigen::Matrix4f& T, const Eigen::Vector3f& v = Eigen::Vector3f::Zero())
  {
    Eigen::Matrix3f R = T.topLeftCorner(3, 3);
    pos = T.topRightCorner(3, 1);
    qua = Eigen::Quaternionf(R);
    vel = v;

    cov = 0.5 * Eigen::MatrixXf::Identity(9, 9);
  }

  void predict(const Eigen::Vector3f& acc, const Eigen::Vector3f& omega, float dt)
  {
    Eigen::Matrix3f R = qua.toRotationMatrix();
    Eigen::Quaternionf dq = exp(R.transpose() * omega * dt);

    pos += vel * dt + 0.5 * (R * acc - gravity) * dt * dt;
    vel += (R * acc - gravity) * dt;
    qua = qua * dq;

    cov += V * dt;
  }

  void observe(const Eigen::Matrix4f& T, int time_machine = 0)
  {
  }

  Eigen::Matrix4f getState()
  {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.topLeftCorner(3, 3) = qua.toRotationMatrix();
    T.topRightCorner(3, 1) = pos;
    return T;
  }

private:
  Eigen::Quaternionf exp(const Eigen::Vector3f& v);

  const Eigen::Vector3f gravity;

  // drive noise
  Eigen::MatrixXf V;

  //  nominal state
  Eigen::Vector3f pos, vel;
  Eigen::Quaternionf qua;

  // variance covariance matrix
  Eigen::MatrixXf cov;  // 9x9
};
}  // namespace vllm