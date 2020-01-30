#pragma once
#include <pcl/common/pca.h>

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

// Local Point Distribution
struct LPD {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LPD() : N(0), sigma(Eigen::Vector3f::Zero()), T(Eigen::Matrix4f::Identity()) {}

  LPD(const pcXYZ::Ptr& cloud, float gain) : N(0), sigma(Eigen::Vector3f::Zero()), T(Eigen::Matrix4f::Identity())
  {
    init(cloud, gain);
  }

  void init(const pcXYZ::Ptr& cloud, float gain)
  {
    N = cloud->size();
    if (N < 20)
      return;

    // primary component analysis
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud);
    Eigen::Matrix3f _R = correctRotationMatrix(pca.getEigenVectors());
    Eigen::Vector3f _sigma = correctSigma(pca.getEigenValues());
    _sigma = _sigma.array().sqrt() * gain;

    // compute centroid
    pcl::PointXYZ point;
    pcl::computeCentroid(*cloud, point);
    Eigen::Vector3f _mu = point.getArray3fMap();

    // set member variables
    sigma = _sigma;
    T = makeT(_R, _mu);
  }

  size_t N;
  Eigen::Vector3f sigma;
  Eigen::Matrix4f T;

  Eigen::Matrix3f R() { return T.topLeftCorner(3, 3); }
  Eigen::Vector3f t() { return T.topRightCorner(3, 1); }

  Eigen::Matrix3f invR() { return R().transpose(); }
  Eigen::Vector3f invt() { return -invR() * t(); }
  Eigen::Matrix4f invT() { return makeT(invR(), invt()); }

private:
  Eigen::Matrix4f makeT(const Eigen::Matrix3f& R, const Eigen::Vector3f& mu)
  {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.topLeftCorner(3, 3) = R;
    T.topRightCorner(3, 1) = mu;
    return T;
  }

  Eigen::Matrix3f correctRotationMatrix(const Eigen::Matrix3f& R)
  {
    if (R.determinant() < 0) {
      Eigen::Matrix3f A = Eigen::Matrix3f::Identity();
      A(2, 2) = -1;
      return R * A;
    } else {
      return R;
    }
  }

  Eigen::Vector3f correctSigma(const Eigen::Vector3f& sigma)
  {
    return Eigen::Vector3f(std::abs(sigma.x()), std::abs(sigma.y()), std::abs(sigma.z()));
  }
};

}  // namespace vllm