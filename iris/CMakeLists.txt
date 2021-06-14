cmake_minimum_required(VERSION 3.10)

# == Project ==
project(iris)

# == Check C++17 ==
include(CheckCXXCompilerFlag)
enable_language(CXX)
check_cxx_compiler_flag("-std=gnu++17" COMPILER_SUPPORTS_CXX17)
if(NOT ${COMPILER_SUPPORTS_CXX17})
  message(FATAL_ERROR "${CMAKE_CXX_COMPILER} doesn't support C++17\n")
endif()

# == Use C++17 ==
set(CMAKE_CXX_STANDARD 17)
message("Compiler:\n\t${CMAKE_CXX_COMPILER} (using C++17)")

# == Set default build type to release ==
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
message("Build Type:\n\t${CMAKE_BUILD_TYPE}")

# == Clear "CMAKE_CXX_FLAGS" ==
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -pipe -fopenmp -Ofast -lstdc++fs -mfpmath=both -mtune=native"
)# -mtune=native

# == Set warning flags ==
set(CXX_WARNING_FLAGS
    -Wall
    -Wextra
    -Wconversion
    -Wswitch-default
    -Wdisabled-optimization
    -Wformat
    -Winit-self
    -Woverloaded-virtual
    -Wfloat-equal
    -Wno-old-style-cast
    -Wno-pragmas)
foreach(FLAG IN LISTS CXX_WARNING_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
endforeach()
message("CMAKE_CXX_FLAGS:\n\t${CMAKE_CXX_FLAGS}")

# == Point Cloud Library ==
find_package(PCL QUIET REQUIRED COMPONENTS common io registration visualization kdtree)
include_directories(SYSTEM ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
message(STATUS "PCL version:\n\t${PCL_VERSION}")

# == OpenCV ==
find_package(OpenCV 3.2 REQUIRED)
message(STATUS "OpenCV version:\n\t${OpenCV_VERSION}")

# == Eigen3 ==
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 version:\n\t${EIGEN3_VERSION_STRING}")

# yaml-cpp
find_package(yaml-cpp REQUIRED)

# == g2o ==
find_package(g2o REQUIRED)
set(G2O_LIBS
    g2o::core
    g2o::stuff
    g2o::types_sba
    g2o::types_sim3
    g2o::solver_dense
    g2o::solver_eigen
    g2o::solver_csparse
    g2o::csparse_extension
    ${CXSPARSE_LIBRARIES}
    ${SUITESPARSE_LIBRARIES})
message(STATUS "g2o version:\n\t${g2o_VERSION}")

# == Catkin ==
find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport tf nav_msgs)
catkin_package(CATKIN_DEPENDS cv_bridge image_transport tf nav_msgs)

# == Headers and sources ==
include_directories(SYSTEM ${catkin_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)
file(GLOB SOURCES src/**/*.cpp)
message(STATUS "${SOURCES}")

# == Executable ==
add_executable(iris_node src/iris_node.cpp ${SOURCES})
target_link_libraries(iris_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS}
                      ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES})
