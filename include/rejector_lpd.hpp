#pragma once
#include "local_point_distribution.hpp"
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

class CorrespondenceRejectorLpd
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CorrespondenceRejectorLpd()
  {
    table.resize(N);
    for (int i = 0; i < N; ++i) {
      table[i].resize(N);
      for (int j = 0; j < N; ++j) {
        table[i][j].resize(N);
      }
    }
  }

  void init(const pcXYZ::Ptr& cloud)
  {
    pcl::PointXYZ min_point, max_point;
    pcl::getMinMax3D(*cloud, min_point, max_point);

    D = getResolution(min_point, max_point, N);
    bottom << min_point.x, min_point.y, min_point.z, 1.0f;

    pcl::CropBox<pcl::PointXYZ> clop;
    clop.setInputCloud(cloud);

    LpdAnalyzer analyzer;

    // cut out pointcloud for each voxel
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
          Eigen::Vector4f min_box, max_box;
          min_box << D.x * static_cast<float>(i), D.y * static_cast<float>(j), D.z * static_cast<float>(k), 1.0f;
          max_box << D.x * static_cast<float>(i + 1), D.y * static_cast<float>(j + 1), D.z * static_cast<float>(k + 1), 1.0f;

          pcXYZ cloud_cropped;
          clop.setMax(bottom + max_box);
          clop.setMin(bottom + min_box);
          clop.filter(cloud_cropped);

          LPD tmp = analyzer.compute(cloud_cropped.makeShared());
          table[i][j][k] = tmp;
        }
      }
    }
  }

  bool check(const pcl::PointXYZ& point)
  {
    Eigen::Vector3i index = getIndex(point);
    LPD lpd = table[index.x()][index.y()][index.z()];

    // debug
    // std::cout << index.transpose() << " " << point << std::endl;
    // lpd.show();

    if (lpd.N < 10)
      return false;

    Eigen::Vector3f p = point.getVector3fMap();
    Eigen::Vector3f transformed = lpd.R() * p + lpd.t();
    return inBox(transformed, 0.01f * static_cast<float>(lpd.N) * lpd.sigma);
  }

private:
  const int N = 2;
  std::vector<std::vector<std::vector<LPD>>> table;

  // blok size
  pcl::PointXYZ D;
  Eigen::Vector4f bottom;

  pcl::PointXYZ getResolution(pcl::PointXYZ min, pcl::PointXYZ max, int N)
  {
    float inv = 1.0f / static_cast<float>(N);
    return pcl::PointXYZ((max.x - min.x) * inv, (max.y - min.y) * inv, (max.z - min.z) * inv);
  }

  Eigen::Vector3i getIndex(const pcl::PointXYZ& point)
  {
    float x = point.x - bottom.x();
    float y = point.y - bottom.y();
    float z = point.z - bottom.z();

    x = std::max(std::min(x / D.x, static_cast<float>(N - 1)), 0.0f);
    y = std::max(std::min(y / D.y, static_cast<float>(N - 1)), 0.0f);
    z = std::max(std::min(z / D.z, static_cast<float>(N - 1)), 0.0f);

    return Eigen::Vector3i(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z));
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