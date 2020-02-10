#pragma once
#include <pangolin/pangolin.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class PangolinCloud
{
public:
  PangolinCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
      : numPoints(static_cast<int>(cloud->size())), offset(4), stride(sizeof(pcl::PointXYZ))
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
    // Render some stuff
    glColor3f(1.0f, 1.0f, 0.0f);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, stride, 0);
    // glColorPointer(3, GL_UNSIGNED_BYTE, stride, (void*)(sizeof(float) * offset));

    glEnableClientState(GL_VERTEX_ARRAY);
    // glEnableClientState(GL_COLOR_ARRAY);

    glDrawArrays(GL_POINTS, 0, numPoints);

    // glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  const int numPoints;

private:
  const int offset;
  const int stride;
  GLuint vbo;
};