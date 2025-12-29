cmake_minimum_required(VERSION 3.22)

# For std::filesystem
# Must be a cache variable and be set before project()
# Reference: https://cmake.org/cmake/help/latest/variable/CMAKE_OSX_DEPLOYMENT_TARGET.html
# Maybe it can be a normal variable if policy CMP0126 is set to NEW?
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.15 CACHE STRING "Minimum OS X deployment version")

project(onnx_optimizer C CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/utils.cmake)

# For integration with onnxruntime_webassembly etc.
if (NOT DEFINED ONNX_TARGET_NAME)
  set(ONNX_TARGET_NAME onnx)
endif()

if(DEFINED BUILD_ONNX_PYTHON AND NOT DEFINED ONNX_BUILD_PYTHON)
  set(ONNX_BUILD_PYTHON ${BUILD_ONNX_PYTHON})
endif()

option(ONNX_OPT_USE_SYSTEM_PROTOBUF "" OFF)
if(NOT ONNX_OPT_USE_SYSTEM_PROTOBUF)
  option(protobuf_BUILD_TESTS "" OFF)
  option(protobuf_MSVC_STATIC_RUNTIME "" ${ONNX_USE_MSVC_STATIC_RUNTIME})
  add_subdirectory_if_no_target(${PROJECT_SOURCE_DIR}/third_party/protobuf libprotobuf)
endif()


set(ONNX_ROOT ${PROJECT_SOURCE_DIR}/third_party/onnx)
add_subdirectory_if_no_target(${ONNX_ROOT} ${ONNX_TARGET_NAME})

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" ONNX_OPTIMIZER_VERSION)
string(STRIP "${ONNX_OPTIMIZER_VERSION}" ONNX_OPTIMIZER_VERSION)

file(GLOB onnx_opt_srcs CONFIGURE_DEPENDS "onnxoptimizer/*.cc"
    "onnxoptimizer/*.h"
    "onnxoptimizer/passes/*.cc"
    "onnxoptimizer/passes/*.h"
    )
list(REMOVE_ITEM onnx_opt_srcs "${PROJECT_SOURCE_DIR}/onnxoptimizer/cpp2py_export.cc")

onnxopt_add_library(onnx_optimizer ${onnx_opt_srcs})
target_link_libraries(onnx_optimizer PUBLIC ${ONNX_TARGET_NAME})
target_include_directories(onnx_optimizer PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )

onnxopt_add_executable(onnx_optimizer_exec examples/onnx_optimizer_exec.cpp)
target_link_libraries(onnx_optimizer_exec onnx_optimizer)


file(GLOB onnx_opt_c_api_srcs CONFIGURE_DEPENDS "onnxoptimizer/c_api/*.cc"
  "onnxoptimizer/c_api/*.h"
    )

onnxopt_add_library(onnx_optimizer_c_api ${onnx_opt_c_api_srcs})
target_link_libraries(onnx_optimizer_c_api PRIVATE onnx_optimizer)
target_include_directories(onnx_optimizer_c_api PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )

if(ONNX_BUILD_PYTHON)
  find_package(
    Python 3
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
  add_subdirectory(third_party/nanobind EXCLUDE_FROM_ALL)
  nanobind_add_module(onnx_opt_cpp2py_export "onnxoptimizer/cpp2py_export.cc" STABLE_ABI)
  target_link_libraries(onnx_opt_cpp2py_export PRIVATE onnx_optimizer)
endif()

include(GNUInstallDirs)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/onnxoptimizer
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.h")

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/ONNXOptimizerConfigVersion.cmake.in
  ${PROJECT_BINARY_DIR}/ONNXOptimizerConfigVersion.cmake
  @ONLY)
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/ONNXOptimizerConfig.cmake.in
  ${PROJECT_BINARY_DIR}/ONNXOptimizerConfig.cmake
  @ONLY)
install(FILES
  ${PROJECT_BINARY_DIR}/ONNXOptimizerConfigVersion.cmake
  ${PROJECT_BINARY_DIR}/ONNXOptimizerConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ONNXOptimizer
  COMPONENT dev)
install(EXPORT ONNXOptimizerTargets DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ONNXOptimizer")
install(TARGETS
  onnx_optimizer onnx_optimizer_c_api
  EXPORT ONNXOptimizerTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(ONNX_BUILD_TESTS)
  enable_testing()
  include(third_party/onnx/cmake/external/googletest.cmake)
  add_executable(test_simple tests/test_simple.cc)
  target_link_libraries(test_simple onnx_optimizer gtest gtest_main) 
  add_test(NAME test_simple COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_simple)
  add_dependencies(test_simple onnx_gtests)
endif()
