cmake_minimum_required (VERSION 3.0.0)
project (ceres_c_api)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# add_definitions(-DCOMPILEDWITHC11)
# message(STATUS "Using flag -std=c++11.")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})



add_library(${PROJECT_NAME} SHARED
    src/ceres_c_api.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${CERES_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)