#pragma once
#include <Eigen/Dense>

namespace vllm
{
// error state extended kalman filter
class EKF
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EKF() : gravity(0, 0, -9.8f)
  {
    // state variable
    pos.setZero();
    vel.setZero();
    qua.setIdentity();
    // variance covariance matrix
    P.setIdentity(9, 9);

    // drive noise
    V.setZero(9, 9);
    V.topLeftCorner(3, 3) = 0.1 * Eigen::Matrix3f::Identity(3, 3);      // position noise
    V.block(3, 3, 3, 3) = 0.1 * Eigen::Matrix3f::Identity(3, 3);        // velocity noise
    V.bottomRightCorner(3, 3) = 0.1 * Eigen::Matrix3f::Identity(3, 3);  // rotation noise

    // observe noise
    W.setZero(6, 6);
    W.topLeftCorner(3, 3) = 0.1 * Eigen::Matrix3f::Identity(3, 3);      // position noise
    W.bottomRightCorner(3, 3) = 0.1 * Eigen::Matrix3f::Identity(3, 3);  // rotation noise
  }
  EKF(const Eigen::Matrix4f& T) : EKF()
  {
    init(T);
  }

  Eigen::Matrix4f getState();

  void init(const Eigen::Matrix4f& T, const Eigen::Vector3f& v = Eigen::Vector3f::Zero());

  void predict(const Eigen::Vector3f& acc, const Eigen::Vector3f& omega, float dt);

  void observe(const Eigen::Matrix4f& T, int time_machine = 0);

private:
  Eigen::Quaternionf exp(const Eigen::Vector3f& v);

  Eigen::VectorXf toVec(const Eigen::Vector3f& p, const Eigen::Quaternionf& q);

  Eigen::MatrixXf calcH(const Eigen::Quaternionf& q);

  const Eigen::Vector3f gravity;

  // drive noise
  Eigen::MatrixXf V;
  // observe noise
  Eigen::MatrixXf W;

  // nominal state
  Eigen::Vector3f pos, vel;
  Eigen::Quaternionf qua;

  // variance covariance matrix
  Eigen::MatrixXf P;  // 9x9
};
}  // namespace vllm
