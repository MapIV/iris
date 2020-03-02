#include "pangolin_viewer.hpp"
#include "system.hpp"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
  vllm::System system(argc, argv);

  // setup for Viewer
  // vllm::PangolinViewer pangolin_viewer;
  // cv::namedWindow("OpenCV", cv::WINDOW_AUTOSIZE);

  while (true) {
    int ok = system.execute();
    std::cout << "state: " << ok << std::endl;
  }

  return 0;
}