#ifndef ONNXOPTIMIZER_C_API_H
#define ONNXOPTIMIZER_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

/// caller must call C_API_ReleasePasses to free memory
const char** C_API_GetAvailablePasses();

/// caller must call C_API_ReleasePasses to free memory
const char** C_API_GetFuseAndEliminationPass();

void C_API_ReleasePasses(const char*** passes);

// caller must call free to release mp_out buffer
bool C_API_Optimize(const char* mp_in, const size_t mp_in_size,
                    const char** passes, const bool fix_point, void** mp_out,
                    size_t* mp_out_size);

bool C_API_OtimizeFromFile(const char* import_model_path,
                           const char* export_model_path, const char** passes,
                           const bool fix_point, const bool save_external_data,
                           const char* data_file_name);

#ifdef __cplusplus
}
#endif

#endif