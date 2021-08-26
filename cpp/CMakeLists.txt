##

if (NOT(EXE))
  set(EXE classify)
endif()

# if (NOT(CMAKE_BUILD_TYPE STREQUAL "Release"))
#   set(FLAG_DEBUG -DDEBUG)
# endif()
set(FLAG_DEBUG -DDEBUG)

project(${EXE})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
find_package(InferenceEngine 2.0 REQUIRED)
find_package(ngraph REQUIRED)
find_package(OpenCV 4.1 REQUIRED)

add_definitions(-DTBB_SUPPRESS_DEPRECATED_MESSAGES ${FLAG_DEBUG})

set(IE_ROOT_DIR ${InferenceEngine_DIR}/..)

include_directories(
  ${IE_ROOT_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)	

if(APPLE)
  include_directories(/opt/local/include)
  link_directories(/opt/local/lib)
endif(APPLE)

string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)

if (ARCH STREQUAL x86_64)
  set(ARCH intel64)
endif()

link_directories(${IE_ROOT_DIR}/lib/${ARCH})

if (ARCH STREQUAL "arm64")
  set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wno-deprecated-declarations")
else()
  set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -march=native")  
endif()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

file (GLOB_RECURSE SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/classify.cpp
)

add_executable (${EXE} ${SOURCE_FILES})

target_link_libraries (${EXE} PRIVATE
  ${OpenCV_LIBS}
  inference_engine inference_engine_legacy
  gflags
)


# end
