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
    int ok = system->update();

    // visualize by OpenCV
    cv::imshow("VLLM", system->getFrame());
    int key = cv::waitKey(5);

    if (key == 'q') break;
    if (key == 's') {
      while (key == 's')
        key = cv::waitKey(0);
    }

    if (ok != 0)
      continue;

    // TODO: I want to put this iteration in the system class,
    // but now the Pangolin-viewer does not allow to do it.
    for (int i = 0; i < 5; i++) {
      if (system->optimize(i))
        break;
    }
    auto dur = std::chrono::system_clock::now() - m_start;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << " ms" << std::endl;
  }

  return 0;
}