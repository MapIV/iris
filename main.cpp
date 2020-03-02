#include "pangolin_viewer.hpp"
#include "system.hpp"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
  std::shared_ptr<vllm::System> system = std::make_shared<vllm::System>(argc, argv);
  vllm::PangolinViewer pangolin_viewer(system);
  cv::namedWindow("VLLM", cv::WINDOW_AUTOSIZE);

  bool loop = true;
  while (loop) {
    int ok = system->update();
    std::cout << "state: " << ok << std::endl;

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

    for (int i = 0; i < 5; i++) {
      system->optimize(i);

      // visualize by Pangolin
      int flag = pangolin_viewer.execute();
      if (flag == -1)
        loop = false;
    }
  }

  return 0;
}