cmake_minimum_required(VERSION 3.10)

# == Project ==
project(vllm)

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fopenmp -Ofast -lstdc++fs -mfpmath=both")
# set(CMAKE_CXX_FLAGS
#     "${CMAKE_CXX_FLAGS} -pipe -fopenmp -Ofast -lstdc++fs -mtune=native -march=native -mfpmath=both"
# )

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
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version:\n\t${OpenCV_VERSION}")

# == Eigen3 ==
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 version:\n\t${EIGEN3_VERSION_STRING}")

# == Pangolin ==
find_package(Pangolin)
include_directories(${Pangolin_INCLUDE_DIRS})

# == OpenGL==
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# == g2o ==
find_package(
  g2o REQUIRED
  COMPONENTS g2o::core
             g2o::stuff
             g2o::types_sba
             g2o::types_sim3
             g2o::solver_dense
             g2o::solver_eigen
             g2o::solver_csparse
             g2o::csparse_extension
  OPTIONAL_COMPONENTS g2o::csparse)
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

# == openvslam ==
set(BUILD_EXAMPLES OFF) # disable building examples
set(USE_PANGOLIN_VIEWER OFF) # disbale buidling viewer
add_subdirectory(3rd/openvslam)

message("CMAKE_CXX_FLAGS:\n\t${CMAKE_CXX_FLAGS}")

# == Headers and sources ==
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB SOURCES src/**/*.cpp)

# == Library ==
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_compile_definitions(${PROJECT_NAME} PUBLIC USE_DBOW2)
target_include_directories(
  ${PROJECT_NAME} SYSTEM
  PUBLIC ${PROJECT_SOURCE_DIR}/3rd/openvslam/src/
         ${PROJECT_SOURCE_DIR}/3rd/openvslam/3rd/json/include
         ${PROJECT_SOURCE_DIR}/3rd/openvslam/3rd/popl/include
         ${PROJECT_SOURCE_DIR}/3rd/openvslam/3rd/spdlog/include)
set_target_properties(
  ${PROJECT_NAME} PROPERTIES OUTPUT_NAME vllm LIBRARY_OUTPUT_DIRECTORY
                                              ${PROJECT_BINARY_DIR}/lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} ${G2O_LIBS} openvslam pangolin)

# == Executable ==
add_executable(main main.cpp)
target_link_libraries(main ${PCL_LIBRARIES} ${OpenCV_LIBS} pangolin vllm)

# # == Tool ==
# add_executable(adjuster tools/pcd_adjuster.cpp)
# target_link_libraries(adjuster ${PCL_LIBRARIES} ${OpenCV_LIBS})
