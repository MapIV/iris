#include "vllm/bridge/bridge.hpp"
#include "vllm/core/config.hpp"
#include "vllm/map/map.hpp"
#include "vllm/system/system.hpp"
#include "vllm/viewer/pangolin_viewer.hpp"
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>

const std::string WIN_NAME = "☣☣ Visual Localization in 3D LiDAR Map ☣☣";

Eigen::Matrix4f processRecover(const Eigen::Matrix4f& T_);

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::cerr << "invalid arguments" << std::endl;
    std::cout << "./main [path to config.yaml]" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Initialize config
  vllm::Config config(argv[1]);

  // Load LiDAR map
  vllm::map::Parameter map_param(
      config.pcd_file, config.voxel_grid_leaf, config.normal_search_leaf, config.submap_grid_leaf);
  std::shared_ptr<vllm::map::Map> map = std::make_shared<vllm::map::Map>(map_param);

  // Initialize system
  std::shared_ptr<vllm::System> system = std::make_shared<vllm::System>(config, map);

  // Initialize viewer
  vllm::viewer::PangolinViewer pangolin_viewer(system);
  pangolin_viewer.startLoop();
  cv::namedWindow(WIN_NAME, cv::WINDOW_AUTOSIZE);

  // Setup for OpenVSLAM
  vllm::pcXYZIN::Ptr vslam_data(new vllm::pcXYZIN);
  vllm::BridgeOpenVSLAM bridge;
  bridge.setup(config);

  // video
  cv::VideoCapture video = cv::VideoCapture(config.video_file, cv::CAP_FFMPEG);

  std::chrono::system_clock::time_point m_start;
  int skipped_frame = 0;

  float accuracy = 0.5f;

  while (true) {

    // Read frame from video
    cv::Mat frame;
    if (!video.read(frame)) break;
    skipped_frame++;

    if (skipped_frame == config.frame_skip) {
      skipped_frame = 0;

      m_start = std::chrono::system_clock::now();  // start timer

      // process OpenVSLAM
      bridge.execute(frame);
      bridge.setCriteria(30, accuracy);
      bridge.getLandmarksAndNormals(vslam_data);
      // Update threshold to adjust the number of points
      if (vslam_data->size() < 300 && accuracy > 0.10f) accuracy -= 0.01f;
      if (vslam_data->size() > 500 && accuracy < 0.90f) accuracy += 0.01f;

      int vllm_state = system->execute(bridge.getState(), bridge.getCameraPose().inverse(), vslam_data);

      // stop timer
      std::cout << "time= \033[35m"
                << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count()
                << "\033[m ms" << std::endl;

      // visualize by OpenCV
      cv::imshow(WIN_NAME, bridge.getFrame());
      int key = cv::waitKey(1);
      if (key == 'q') break;
      if (key == 'f') system->specifyTWorld(processRecover(system->getTWorld()));
      if (key == 's')
        while (key == 's') key = cv::waitKey(0);
    }
  }
  return 0;
}

Eigen::Matrix4f processRecover(const Eigen::Matrix4f& T_)
{
  std::cout << "\033[2J\033[1;1H";  // new page
  std::cout << "\033[1m\033[31m current t_world: \033[0m\033[33m" << T_.topRightCorner(3, 1).transpose() << std::endl;
  std::cout << "\033[0m\033[1m\033[31m Please enter the 2D pose to recover the localization." << std::endl;
  std::cout << "The format is 'x y nx ny', where (nx,ny) is not necesary to be normalized." << std::endl;
  std::cout << "ex) 2.0  -1.45 -1.5 0.7[Enter]" << std::endl;
  std::cout << "\033[m";  // reset font
  float x, y, nx, ny;
  std::cin >> x >> y >> nx >> ny;
  Eigen::Matrix4f T = vllm::util::make3DPoseFrom2DPose(x, y, nx, ny);
  std::cout << "new Pose\n"
            << T << std::endl;
  return T;
}