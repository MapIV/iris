cmake_minimum_required(VERSION 3.10)

# == project ==
project(vllm)

# == check C++17 ==
include(CheckCXXCompilerFlag)
enable_language(CXX)
check_cxx_compiler_flag("-std=gnu++17" COMPILER_SUPPORTS_CXX17)
if(NOT ${COMPILER_SUPPORTS_CXX17})
    message(FATAL_ERROR "${CMAKE_CXX_COMPILER} doesn't support C++17\n")
endif()

# == C++17 ==
set(CMAKE_CXX_STANDARD 17)
message("Compiler:\n\t${CMAKE_CXX_COMPILER} (using C++17)")

# == set default build type to release ==
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RELEASE")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
message("Build Type:\n\t${CMAKE_BUILD_TYPE}")

# == clear "CMAKE_CXX_FLAGS" ==
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fopenmp ")

# == set warning flags ==
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

# == Point Cloud Library (PCl) ==
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
find_package(PangolinD)
find_package(OpenGL REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
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

# == header & source ==
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

# == openvslam ==
set(BUILD_EXAMPLES 0)
add_subdirectory(3rd/openvslam)

# == main ==
add_executable(main main.cpp src/bridge.cpp)
target_include_directories(main SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/3rd/openvslam/3rd/popl/include ${PROJECT_SOURCE_DIR}/3rd/openvslam/3rd/spdlog/include)
target_link_libraries(main PRIVATE ${PCL_LIBRARIES} ${OpenCV_LIBS} openvslam)

# PangolinViewer is used on a priority basis
if(USE_PANGOLIN_VIEWER)
    message(STATUS "Use Pangolin Viewer")
    target_compile_definitions(main PRIVATE USE_PANGOLIN_VIEWER)
    target_link_libraries(main PRIVATE pangolin_viewer)
endif()

# == example file ==
# add_executable(optimize_sim3 optimize_sim3.cpp)
# target_link_libraries(optimize_sim3 PUBLIC ${G2O_LIBS})

# add_executable(icp examle/icp.cpp)
# target_link_libraries(icp PUBLIC ${PCL_LIBRARIES} aligner)

# add_executable(reject examle/reject_with_lpd.cpp)
# target_link_libraries(reject PUBLIC ${PCL_LIBRARIES})

# add_executable(reject_test examle/reject_test.cpp)
# target_link_libraries(reject_test PUBLIC ${PCL_LIBRARIES})

# add_executable(lpd_simple examle/lpd_simple.cpp)
# target_link_libraries(lpd_simple PUBLIC ${PCL_LIBRARIES})

# add_executable(viewer examle/simple_viewer.cpp)
# target_link_libraries(viewer PUBLIC ${PCL_LIBRARIES})
