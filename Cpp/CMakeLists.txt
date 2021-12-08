cmake_minimum_required(VERSION 3.15+)

project(aoc2021 LANGUAGES CXX)

# Setting up Build type
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# C++ Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native -std=c++14")
# include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
# CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
# if(COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
#    add_definitions(-DCOMPILEDWITHC11)
#    message(STATUS "Using flag -std=c++11.")
# elseif(COMPILER_SUPPORTS_CXX0X)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#    add_definitions(-DCOMPILEDWITHC0X)
#    message(STATUS "Using flag -std=c++0x.")
# else()
#    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
# endif()

# Relevant Packages
find_package(Eigen3 3.4.0 REQUIRED)
find_package(xtensor REQUIRED)

add_library(${PROJECT_NAME})

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    xtensor
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${xtensor_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_subdirectory(src)

add_executable(day1 src/day1/main.cpp)
target_link_libraries(day1 ${PROJECT_NAME})
