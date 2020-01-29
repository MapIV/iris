#pragma once
#include <pcl/common/pca.h>

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

// Local Point Distribution
struct LPD {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LPD() : N(0), sigma(Eigen::Vector3f::Zero()), T(Eigen::Matrix4f::Identity()) {}

  LPD(size_t N, const Eigen::Matrix3f& R, const Eigen::Vector3f& mu, const Eigen::Vector3f& sigma)
      : N(N), sigma(sigma), T(makeT(R, mu)) {}

  LPD(size_t N, const Eigen::Matrix4f& T, const Eigen::Vector3f& sigma)
      : N(N), sigma(sigma), T(T) {}

  size_t N;
  Eigen::Vector3f sigma;
  Eigen::Matrix4f T;

  void show()
  {
    std::cout << "N= " << N << " ,sigma= " << sigma.transpose() << "\n"
              << T << std::endl;
  }

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
};

class LpdAnalyzer
{
public:
  LPD compute(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
  {
    const size_t N = cloud->size();
    if (N < 5)
      return LPD{};

    // primary component analysis
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud);
    Eigen::Matrix3f R = correctRotationMatrix(pca.getEigenVectors());
    Eigen::Vector3f sigma = correctSigma(pca.getEigenValues());
    sigma = sigma.array().sqrt() / 8;

    // compute centroid
    pcl::PointXYZ point;
    pcl::computeCentroid(*cloud, point);
    Eigen::Vector3f mu = point.getArray3fMap();

    return LPD(N, R, mu, sigma);
  }

private:
  Eigen::Matrix3f correctRotationMatrix(const Eigen::Matrix3f& R)
  {
    if (R.trace() < 0) {
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