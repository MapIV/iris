#pragma once
#include <pangolin/pangolin.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vllm
{
namespace viewer
{
class PangolinCloud
{
public:
  PangolinCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
      : color(false), numPoints(static_cast<int>(cloud->size())), offset(4), stride(sizeof(pcl::PointXYZ))
  {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, cloud->points.size() * stride, cloud->points.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  PangolinCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud)
      : color(true), numPoints(static_cast<int>(cloud->size())), offset(4), stride(sizeof(pcl::PointXYZRGBA))
  {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, cloud->points.size() * stride, cloud->points.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  virtual ~PangolinCloud()
  {
    glDeleteBuffers(1, &vbo);
  }

  void drawPoints()
  {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, stride, 0);
    if (color)
      glColorPointer(4, GL_UNSIGNED_BYTE, stride, (void*)(sizeof(float) * offset));

    glEnableClientState(GL_VERTEX_ARRAY);
    if (color)
      glEnableClientState(GL_COLOR_ARRAY);

    glDrawArrays(GL_POINTS, 0, numPoints);

    if (color)
      glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }


private:
  const bool color;
  const int numPoints;
  const int offset;
  const int stride;
  GLuint vbo;
};
}  // namespace viewer
}  // namespace vllm