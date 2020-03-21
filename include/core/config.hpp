#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>

namespace vllm
{
struct Config {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Config() {}

  Config(const std::string& yaml_file)
  {
    init(yaml_file);
  }

  void init(const std::string& yaml_file)
  {
    self_path = yaml_file;

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
      R.row(1) = (n.dot(u) * n - u).normalized();  // Gram–Schmidt orthonormalization
      R.row(0) = R.row(1).cross(R.row(2));

      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.topLeftCorner(3, 3) = s * R.transpose();
      T.topRightCorner(3, 1) = t;
      T_init = T;
      std::cout << T << std::endl;
    }

    fs["Map.normal_search_leaf"] >> normal_search_leaf;
    fs["Map.voxel_grid_leaf"] >> voxel_grid_leaf;
    fs["Map.submap_grid_leaf"] >> submap_grid_leaf;

    fs["VLLM.pcd_file"] >> pcd_file;
    fs["VLLM.video_file"] >> video_file;
    fs["VLLM.vocab_file"] >> vocab_file;
    fs["VLLM.iteration"] >> iteration;
    fs["VLLM.frame_skip"] >> frame_skip;
    fs["VLLM.scale_gain"] >> scale_gain;
    fs["VLLM.smooth_gain"] >> smooth_gain;
    fs["VLLM.latitude_gain"] >> latitude_gain;
    fs["VLLM.altitude_gain"] >> altitude_gain;

    fs["VLLM.distance_min"] >> distance_min;
    fs["VLLM.distance_max"] >> distance_max;

    fs["VLLM.converge_translation"] >> converge_translation;
    fs["VLLM.converge_rotation"] >> converge_rotation;

    cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
  }

  float distance_min, distance_max;
  float scale_gain, latitude_gain, smooth_gain, altitude_gain;
  int frame_skip;
  int iteration;

  float converge_translation;
  float converge_rotation;

  float normal_search_leaf;
  float voxel_grid_leaf;
  float submap_grid_leaf;

  std::string self_path;
  std::string pcd_file;
  std::string video_file;
  std::string vocab_file;
  Eigen::Matrix4f T_init;
};
}  // namespace vllm