include(${PROJECT_SOURCE_DIR}/third_party/onnx/cmake/Utils.cmake)

# Poor man's FetchContent
function(add_subdirectory_if_no_target dir target)
  if (NOT TARGET ${target})
    add_subdirectory(${dir})
  endif()
endfunction()

function(onnxopt_add_library)
  add_library(${ARGV})
  if (MSVC)
    add_msvc_runtime_flag(${ARGV0})
  endif()
endfunction()

function(onnxopt_add_executable)
  add_executable(${ARGV})
  if (MSVC)
    add_msvc_runtime_flag(${ARGV0})
  endif()
endfunction()
