#include "DeepBrainInference.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

DeepBrainInference::DeepBrainInference() {}

bool DeepBrainInference::begin() {
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Load Model
    model = tflite::GetModel(g_bee_brain_model_data);

    // Resolver
    static tflite::AllOpsResolver resolver;

    // Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate Tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    return true;
}

int DeepBrainInference::predict(float* mel_data) {
    if (!input || !output) return -1;

    // Copy mel data to input tensor (Quantized or Float depending on model)
    // If INT8 quantized, we'd need to cast and rescale.
    // Assuming the user handles rescaling during spectrogram generation.
    for (int i = 0; i < 128 * 87; ++i) {
        input->data.f[i] = mel_data[i];
    }

    // Run Inference
    if (interpreter->Invoke() != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Inference failed");
        return -1;
    }

    // Read result (Binary Classification: Healthy vs ALERT)
    float alert_prob = output->data.f[1]; 
    return (alert_prob > 0.5) ? 1 : 0;
}
