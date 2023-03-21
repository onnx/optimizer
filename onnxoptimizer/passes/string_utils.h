/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace ONNX_NAMESPACE {
namespace optimization {

template <typename T, char delimiter = ','>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& datas) {
  os << "[";
  bool first = true;
  for (const auto& d : datas) {
    if (!first) {
      os << delimiter;
    } else {
      first = false;
    }
    os << d;
  }
  os << "]";
  return os;
}

inline std::string StripFilename(const std::string& filepath) {
  size_t pos = filepath.find_last_of('/');
  return (pos == std::string::npos) ? filepath : filepath.substr(pos + 1);
}

inline std::string StripFilename(const char* filepath) {
  return StripFilename(std::string(filepath));
}

namespace {
template <typename T>
struct StrTypeTrait {
  using type = const T&;
};

template <size_t N>
struct StrTypeTrait<char[N]> {
  using type = const char*;
};

inline std::ostream& _str(std::ostream& os) {
  return os;
}

template <typename T, typename... Args>
std::ostream& _str(std::ostream& os, const T& s, const Args&... args) {
  os << s;
  return _str(os, args...);
}

template <typename... Args>
struct StrWrapper {
  static std::string Call(const Args&... args) {
    std::ostringstream os;
    _str(os, args...);
    return os.str();
  }
};

template <>
struct StrWrapper<std::string> {
  static std::string Call(const std::string& s) {
    return s;
  }
};

template <>
struct StrWrapper<const char*> {
  static const char* Call(const char* s) {
    return s;
  }
};

}  // namespace

template <typename... Args>
decltype(auto) Str(const Args&... args) {
  return StrWrapper<typename StrTypeTrait<Args>::type...>::Call(args...);
}

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
