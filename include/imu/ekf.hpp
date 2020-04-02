#pragma once
#include <Eigen/Dense>
#include <iostream>

namespace vllm
{
// error state extended kalman filter
class EKF
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EKF() : gravity(0, 0, 9.8f)
  {
    last_ns = 0;

    // state variable
    pos.setZero();
    vel.setZero();
    qua.setIdentity();
    // variance covariance matrix
    P.setIdentity(9, 9);

    // drive noise
    Eigen::MatrixXf L, Q;
    L.setZero(9, 6);
    L.block(3, 0, 3, 3).setIdentity();
    L.bottomRightCorner(3, 3).setIdentity();
    Q.setZero(6, 6);
    Q.topLeftCorner(3, 3) = Eigen::Matrix3f::Identity() * 1.0;      // velocity variance [m/s]
    Q.bottomRightCorner(3, 3) = Eigen::Matrix3f::Identity() * 1.0;  // rotation variance [rad]
    LQL = L * Q * L.transpose();
    std::cout << "LQL\n"
              << LQL << std::endl;

    // observe noise
    W.setZero(7, 7);
    W.topLeftCorner(3, 3) = 0.05 * Eigen::Matrix3f::Identity();      // position variance [m]
    W.bottomRightCorner(4, 4) = 0.05 * Eigen::Matrix4f::Identity();  // rotation variance [rad]
  }
  EKF(const Eigen::Matrix4f& T) : EKF()
  {
    init(T);
  }

  Eigen::Matrix4f getState();

  void init(const Eigen::Matrix4f& T, const Eigen::Vector3f& v = Eigen::Vector3f::Zero());

  void predict(const Eigen::Vector3f& acc, const Eigen::Vector3f& omega, unsigned long ns);

  void observe(const Eigen::Matrix4f& T, unsigned long ns);

private:
  Eigen::Quaternionf exp(const Eigen::Vector3f& v);

  Eigen::VectorXf toVec(const Eigen::Vector3f& p, const Eigen::Quaternionf& q);

  Eigen::MatrixXf calcH(const Eigen::Quaternionf& q);

  Eigen::MatrixXf calcF(const Eigen::Quaternionf& q, const Eigen::Vector3f& acc, float dt);

  Eigen::Matrix3f hat(const Eigen::Vector3f& vec);

  bool isUpadatable() { return (last_ns != 0); }

  const Eigen::Vector3f gravity;

  unsigned long last_ns;

  float scale = 1.0;

  // drive noise
  Eigen::MatrixXf LQL;
  // observe noise
  Eigen::MatrixXf W;

  // nominal state
  Eigen::Vector3f pos, vel;
  Eigen::Quaternionf qua;

  // variance covariance matrix
  Eigen::MatrixXf P;  // 9x9
};
}  // namespace vllm
