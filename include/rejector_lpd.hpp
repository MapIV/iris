#pragma once
#include "global_point_distribution.hpp"
#include <pcl/filters/crop_box.h>

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

class CorrespondenceRejectorLpd
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CorrespondenceRejectorLpd(GPD gpd) : gpd(gpd) {}

  pcl::Correspondences refineCorrespondences(const pcl::Correspondences& correspondences, const pcXYZ::Ptr& cloud)
  {
    pcl::Correspondences refined;
    for (size_t i = 0, N = correspondences.size(); i < N; i++) {
      const pcl::Correspondence pair = correspondences.at(i);
      pcl::PointXYZ point = cloud->at(i);
      // pcl::PointXYZ point = cloud->at(pair.index_query);
      if (check(point)) {
        refined.push_back(pair);
      }
    }
    return refined;
  }

  pcl::PointXYZ project(const pcl::PointXYZ& point)
  {
    LPD lpd = gpd.getLPD(point);
    Eigen::Vector3f p = point.getVector3fMap();
    p = lpd.invR() * p + lpd.invt();

    return pcl::PointXYZ(p.x(), p.y(), p.z());
  }


  bool check(const pcl::PointXYZ& point)
  {
    LPD lpd = gpd.getLPD(point);
    if (lpd.N < 20)
      return false;

    Eigen::Vector3f p = point.getVector3fMap();
    Eigen::Vector3f transformed = lpd.invR() * p + lpd.invt();
    return inBox(transformed, lpd.sigma);
  }

private:
  GPD gpd;

  Eigen::Vector4f toVec4f(const Eigen::Vector3f& vec)
  {
    return Eigen::Vector4f(vec.x(), vec.y(), vec.z(), 1);
  }

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