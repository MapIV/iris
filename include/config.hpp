#pragma once
#include <opencv2/core/eigen.hpp>

namespace vllm
{
struct Config {
  Config() {}

  Config(const std::string& yaml_file)
  {
    init(yaml_file);
  }

  void init(const std::string& yaml_file)
  {
    cv::FileStorage fs(yaml_file, cv::FileStorage::READ);
    {
      cv::Mat trans, normal, up;
      float s;
      fs["VLLM.t_init"] >> trans;
      fs["VLLM.normal_init"] >> normal;
      fs["VLLM.up_init"] >> up;
      fs["VLLM.s_init"] >> s;

      Eigen::Vector3f n, u, t;
      Eigen::Matrix3f R = Eigen::Matrix3f::Zero();
      cv::cv2eigen(normal, n);
      cv::cv2eigen(up, u);
      cv::cv2eigen(trans, t);

      n.normalize();
      u.normalize();
      R.row(2) = n;
      R.row(1) = (n.dot(u) * n - u).normalized();
      R.row(0) = R.row(1).cross(R.row(2));

      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.topLeftCorner(3, 3) = s * R.transpose();
      T.topRightCorner(3, 1) = t;
      T_init = T;
      std::cout << T << std::endl;
    }

    fs["VLLM.normal_search_leaf"] >> normal_search_leaf;
    fs["VLLM.voxel_grid_leaf"] >> voxel_grid_leaf;
    fs["VLLM.pcd_file"] >> pcd_file;
    fs["VLLM.video_file"] >> video_file;
    fs["VLLM.gpd_size"] >> gpd_size;
    fs["VLLM.gpd_gain"] >> gpd_gain;
    fs["VLLM.iteration"] >> iteration;
    fs["VLLM.frame_skip"] >> frame_skip;
    fs["VLLM.scale_gain"] >> scale_gain;
    fs["VLLM.pitch_gain"] >> pitch_gain;
    fs["VLLM.model_gain"] >> model_gain;

    fs["VLLM.distance_min"] >> distance_min;
    fs["VLLM.distance_max"] >> distance_max;

    fs["VLLM.converge_translation"] >> converge_translation;
    fs["VLLM.converge_rotation"] >> converge_rotation;

    std::cout << "gpd_gain " << gpd_gain << std::endl;

    cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
  }

  float distance_min, distance_max;
  double scale_gain, pitch_gain, model_gain;
  int frame_skip;
  int iteration;
  float gpd_gain;
  int gpd_size;
  float normal_search_leaf;
  float voxel_grid_leaf;
  float converge_translation;
  float converge_rotation;

  std::string pcd_file;
  std::string video_file;
  Eigen::Matrix4f T_init;
};
}  // namespace vllm