#pragma once
#include "global_point_distribution.hpp"

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

class CorrespondenceRejectorLpd
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CorrespondenceRejectorLpd(GPD gpd, float gain) : gpd(gpd), gain(gain) {}

  pcl::Correspondences refineCorrespondences(const pcl::Correspondences& correspondences, const pcXYZ::Ptr& cloud)
  {
    pcl::Correspondences refined;
    for (size_t i = 0, N = correspondences.size(); i < N; i++) {
      const pcl::Correspondence pair = correspondences.at(i);
      pcl::PointXYZ point = cloud->at(pair.index_query);
      if (check(point)) {
        refined.push_back(pair);
      }
    }
    return refined;
  }

  bool check(const pcl::PointXYZ& point)
  {
    LPD lpd = gpd.getLPD(point);
    if (lpd.N < 20)
      return false;

    Eigen::Vector3f p = point.getVector3fMap();
    Eigen::Vector3f transformed = lpd.invR() * p + lpd.invt();
    return inBox(transformed, gain * lpd.sigma);
  }

private:
  const float gain;
  GPD gpd;

  bool inBox(const Eigen::Vector3f& point, const Eigen::Vector3f& boarder)
  {
    if (point.x() > boarder.x()
        || point.y() > boarder.y()
        || point.z() > boarder.z()) return false;

    if (point.x() < -boarder.x()
        || point.y() < -boarder.y()
        || point.z() < -boarder.z()) return false;
    return true;
  }
};
}  // namespace vllm