#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"
#include "onnxoptimizer/model_util.h"
#include "onnxoptimizer/optimize.h"
#include "onnxoptimizer_c_api.h"

static const char** CopyPasses(const std::vector<std::string>& passes) {
  size_t n = passes.size();
  char** res_passes = static_cast<char**>(malloc(sizeof(char*) * (n + 1)));
  if (!res_passes) {
    return NULL;
  }
  int valid_count = 0;
  for (const auto& pass : passes) {
    const auto* from = pass.c_str();
    const auto n = strlen(from);
    char* to = static_cast<char*>(malloc(n + 1));
    if (!to) {
      continue;
    } else {
      memcpy(to, from, n);
      to[n] = '\0';
      res_passes[valid_count++] = to;
    }
  }
  while (valid_count <= n) {
    res_passes[valid_count++] = NULL;
  }
  return const_cast<const char**>(res_passes);
}

const char** C_API_GetAvailablePasses() {
  return CopyPasses(ONNX_NAMESPACE::optimization::GetAvailablePasses());
}

const char** C_API_GetFuseAndEliminationPass() {
  return CopyPasses(ONNX_NAMESPACE::optimization::GetFuseAndEliminationPass());
}

void C_API_ReleasePasses(const char*** passes) {
  if (!passes) {
    return;
  }
  const char* p = passes[0][0];
  while (p) {
    void* cur = reinterpret_cast<void*>(const_cast<char*>(p));
    p++;
    free(cur);
  }
  free(passes[0]);
  passes[0] = NULL;
}

static bool SerializeProtoAndCopy(const ONNX_NAMESPACE::ModelProto& p,
                                  void** buffer, size_t* size) {
  std::string out;
  p.SerializeToString(&out);
  void* buf = malloc(sizeof(char) * out.size());
  if (!buf) {
    return false;
  }
  memcpy(buf, out.c_str(), out.size());
  *size = out.size();
  *buffer = buf;
  return true;
}

static std::pair<bool, ONNX_NAMESPACE::ModelProto> Optimize(
    const ONNX_NAMESPACE::ModelProto& proto, const char** passes,
    const bool fix_point) {
  std::vector<std::string> names;
  const char* p = passes[0];
  while (p) {
    names.push_back(std::string(p));
    p++;
  }
  if (names.empty()) {
    return std::make_pair(false, ONNX_NAMESPACE::ModelProto());
  }
  try {
    if (fix_point) {
      auto result = ONNX_NAMESPACE::optimization::OptimizeFixed(proto, names);
      return std::make_pair(true, result);
    } else {
      auto result = ONNX_NAMESPACE::optimization::Optimize(proto, names);
      return std::make_pair(true, result);
    }
  } catch (std::exception& e) {
    std::cerr << e.what();
    return std::make_pair(false, ONNX_NAMESPACE::ModelProto());
  }
}

bool C_API_Optimize(const char* mp_in_buffer, const size_t mp_in_size,
                    const char** passes, const bool fix_point,
                    void** mp_out_buffer, size_t* mp_out_size) {
  if (!mp_in_buffer || mp_in_size == 0 || !passes || !mp_out_buffer ||
      !mp_out_size) {
    return false;
  }

  ONNX_NAMESPACE::ModelProto proto{};
  if (!ONNX_NAMESPACE::ParseProtoFromBytes(&proto, mp_in_buffer, mp_in_size)) {
    return false;
  }
  bool ok = false;
  ONNX_NAMESPACE::ModelProto result{};
  std::tie(ok, result) = Optimize(proto, passes, fix_point);
  if (!ok) {
    return false;
  }
  return SerializeProtoAndCopy(result, mp_out_buffer, mp_out_size);
}

bool C_API_OtimizeFromFile(const char* import_model_path,
                           const char* export_model_path, const char** passes,
                           const bool fix_point, const bool save_external_data,
                           const char* data_file_name) {
  if (!import_model_path || !export_model_path || !passes ||
      (save_external_data && !data_file_name)) {
    return false;
  }
  try {
    ONNX_NAMESPACE::ModelProto proto{};
    ONNX_NAMESPACE::optimization::loadModel(
        &proto, std::string(import_model_path), true);
    bool ok = false;
    ONNX_NAMESPACE::ModelProto result{};
    std::tie(ok, result) = Optimize(proto, passes, fix_point);
    if (!ok) {
      return false;
    }
    ONNX_NAMESPACE::optimization::saveModel(
        &result, std::string(export_model_path), save_external_data,
        std::string(data_file_name));
    return true;
  } catch (std::exception& e) {
    std::cerr << e.what();
    return false;
  }
}