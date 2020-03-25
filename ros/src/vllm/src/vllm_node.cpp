#include "core/config.hpp"
#include "map/map.hpp"
#include "system/system.hpp"
#include "viewer/pangolin_viewer.hpp"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <popl.hpp>

int main(int argc, char* argv[])
{
  // analyze arugments
  popl::OptionParser op("Allowed options");
  auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
  try {
    op.parse(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!config_file_path->is_set()) {
    std::cerr << "invalid arguments" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Initialize config
  vllm::Config config(config_file_path->value());

  // Load LiDAR map
  vllm::map::Parameter map_param(
      config.pcd_file, config.voxel_grid_leaf, config.normal_search_leaf, config.submap_grid_leaf);
  std::shared_ptr<vllm::map::Map> map = std::make_shared<vllm::map::Map>(map_param);

  // Initialize system
  std::shared_ptr<vllm::System> system = std::make_shared<vllm::System>(config, map);

  // Initialize viewer
  vllm::viewer::PangolinViewer pangolin_viewer(system);
  pangolin_viewer.startLoop();
  cv::namedWindow("VLLM", cv::WINDOW_AUTOSIZE);


  bool loop = true;
  std::chrono::system_clock::time_point m_start;

  while (loop) {
    m_start = std::chrono::system_clock::now();

    system->execute();

    std::cout << "time= \033[35m"
              << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count()
              << "\033[m ms" << std::endl;

    // visualize by OpenCV
    cv::imshow("VLLM", system->getFrame());
    int key = cv::waitKey(1);
    if (key == 'q') break;
    if (key == 's') {
      while (key == 's')
        key = cv::waitKey(0);
    }
  }

  pangolin_viewer.quitLoop();
  return 0;
}