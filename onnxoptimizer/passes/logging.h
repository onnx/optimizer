/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once
#include <algorithm>  // min
#include <cstdlib>    // abort getenv
#include <iostream>
#include <mutex>
#include <ostream>
#include <sstream>

#include "onnxoptimizer/passes/string_utils.h"

namespace ONNX_NAMESPACE {
namespace optimization {

namespace details {

static constexpr char logging_prefix[] = {'F', 'E', 'W', 'I', 'V'};

constexpr int LOG_INFO = 0;
constexpr int LOG_WARNING = 1;
constexpr int LOG_ERROR = 2;
constexpr int LOG_FATAL = 3;

static std::once_flag read_log_threshold_flag;
static int log_threshold = LOG_INFO;

static void ReadLogThresholdFromEnv() {
  char* threshold = std::getenv("LOG_THRESHOLD");
  if (!threshold) {
    return;
  }
  std::stringstream ss;
  ss << threshold;
  ss >> log_threshold;
}

class MessageControl {
 public:
  MessageControl(const char* file, const char* function, int line, int severity)
      : severity_(severity) {
    std::call_once(read_log_threshold_flag, ReadLogThresholdFromEnv);
    stream_ << "[" << logging_prefix[std::min<int>(4, LOG_FATAL - severity_)]
            << " " << StripFilename(file) << ":" << line << " " << function
            << "]: ";
  }

  ~MessageControl() {
    if (severity_ < log_threshold) {
      return;
    }
    std::cout << stream_.rdbuf() << std::endl;
    if (severity_ == LOG_FATAL) {
      std::abort();
    }
  }

  std::ostream& Stream() {
    return stream_;
  }

 private:
  int severity_;
  std::stringstream stream_;
};

}  // namespace details

#define LOG(n)                                                                \
  details::MessageControl(__FILE__, __FUNCTION__, __LINE__, details::LOG_##n) \
      .Stream()

#define VLOG(n) \
  details::MessageControl(__FILE__, __FUNCTION__, __LINE__, -n).Stream()

#define LOG_IF(n, condition) \
  if (condition)             \
  LOG(n)

#define VLOG_IF(n, condition) \
  if (condition)              \
  VLOG(n)

}  // namespace optimization
}  // namespace ONNX_NAMESPACE