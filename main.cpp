#include "map.hpp"
#include "pangolin_viewer.hpp"
#include "system.hpp"
#include <chrono>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
  std::shared_ptr<vllm::System> system = std::make_shared<vllm::System>(argc, argv);
  vllm::PangolinViewer pangolin_viewer(system);
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