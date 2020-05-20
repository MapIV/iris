#pragma once
#include "vllm/imu/kfparam.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace vllm
{
class EKF
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EKF(const KFParam& param,
      const Eigen::Matrix4f& T,
      const Eigen::Vector3f& initial_vel = Eigen::Vector3f::Zero()) : param(param)
  {
    last_ns = 0;

    // state variable
    Eigen::Matrix3f R = T.topLeftCorner(3, 3);
    pos = T.topRightCorner(3, 1);
    qua = Eigen::Quaternionf(R).normalized();  // NOTE: must normalize here
    vel = initial_vel;
    gravity << 0, 0, 9.8f;
    bias.setZero();

    // variance covariance matrix
    P.setZero(15, 15);
    P.block(0, 0, 3, 3) = param.initial_cov_p * Eigen::Matrix3f::Identity();
    P.block(3, 3, 3, 3) = param.initial_cov_v * Eigen::Matrix3f::Identity();
    P.block(6, 6, 3, 3) = param.initial_cov_theta * Eigen::Matrix3f::Identity();
    P.block(9, 9, 3, 3) = param.initial_cov_grad * Eigen::Matrix3f::Identity();
    P.block(12, 12, 3, 3) = param.initial_cov_bias * Eigen::Matrix3f::Identity();

    // driving noise
    Eigen::MatrixXf Q;
    Q.setZero(9, 9);
    Q.block(0, 0, 3, 3) = param.drive_cov_v * Eigen::Matrix3f::Identity();      // velocity variance [m/s]
    Q.block(3, 3, 3, 3) = param.drive_cov_theta * Eigen::Matrix3f::Identity();  // rotation variance [rad]
    Q.block(6, 6, 3, 3) = param.drive_cov_bias * Eigen::Matrix3f::Identity();   // bias variance [m/s^2]

    Eigen::MatrixXf L;
    L.setZero(15, 9);
    L.block(3, 0, 3, 3).setIdentity();
    L.block(6, 3, 3, 3).setIdentity();
    L.block(12, 6, 3, 3).setIdentity();
    LQL = L * Q * L.transpose();  // (15x9)*(9x9)*(9x15)=(15x15)

    // observe noise
    W.setZero(7, 7);
    W.block(0, 0, 3, 3) = param.observe_cov_p * Eigen::Matrix3f::Identity();      // position variance [m]
    W.block(0, 0, 4, 4) = param.observe_cov_theta * Eigen::Matrix4f::Identity();  // rotation variance [rad]
    slow_start = 10.0;
  }

  Eigen::Matrix4f getState();

  void predict(const Eigen::Vector3f& acc, const Eigen::Vector3f& omega, unsigned long ns);

  void observe(const Eigen::Matrix4f& T, unsigned long ns);

private:
  KFParam param;

  Eigen::Quaternionf exp(const Eigen::Vector3f& v);

  Eigen::VectorXf toVec(const Eigen::Vector3f& p, const Eigen::Quaternionf& q);

  Eigen::MatrixXf calcH(const Eigen::Quaternionf& q);

  Eigen::MatrixXf calcF(const Eigen::Quaternionf& q, const Eigen::Vector3f& acc, float dt);

  Eigen::Matrix3f hat(const Eigen::Vector3f& vec);

  bool isUpadatable() { return (last_ns != 0); }

  unsigned long last_ns;

  float slow_start;
  float scale = 1.0;

  // drive noise
  Eigen::MatrixXf LQL;
  // observe noise
  Eigen::MatrixXf W;

  // nominal state
  Eigen::Vector3f bias;
  Eigen::Vector3f gravity;
  Eigen::Vector3f pos, vel;
  Eigen::Quaternionf qua;

  // variance covariance matrix
  Eigen::MatrixXf P;  // 9x9
};
}  // namespace vllm
