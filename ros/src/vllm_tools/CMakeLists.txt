cmake_minimum_required(VERSION 3.1)
project(vllm_tools)

# == Check C++17 ==
include(CheckCXXCompilerFlag)
enable_language(CXX)
check_cxx_compiler_flag("-std=gnu++17" COMPILER_SUPPORTS_CXX17)
if(NOT ${COMPILER_SUPPORTS_CXX17})
  message(FATAL_ERROR "${CMAKE_CXX_COMPILER} doesn't support C++17\n")
endif()

# == C++17 ==
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fopenmp -Ofast -lstdc++fs")

#################
# Find packages #
#################
# == OpenCV ==
find_package(OpenCV 3 REQUIRED)
message(STATUS "OpenCV version:\n\t${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

# == popl ==
set(popl_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rd/openvslam/3rd/popl/include)
include_directories(${popl_INCLUDE_DIR})

# == Eigen ==
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 version:\n\t${EIGEN3_VERSION_STRING}")

# == Find catkin macros and libraries ==
find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport tf visualization_msgs)

#################################
# catkin specific configuration #
#################################
catkin_package(CATKIN_DEPENDS cv_bridge image_transport tf visualization_msgs)

#########
# Build #
#########
# == include ==
include_directories(${catkin_INCLUDE_DIRS} include)

# === executable ===
add_executable(publisher_node src/publisher_node.cpp)
target_link_libraries(
  publisher_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)