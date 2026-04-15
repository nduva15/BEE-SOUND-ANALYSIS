#ifndef DEEP_BRAIN_INFERENCE_H
#define DEEP_BRAIN_INFERENCE_H

#include <Arduino.h>
#include <stdint.h>
#include <string.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Forward declaration of the model data (exported from xxd or similar)
extern const unsigned char g_bee_brain_model_data[];
extern const int g_bee_brain_model_data_len;

class DeepBrainInference {
public:
    DeepBrainInference();
    bool begin();
    
    /**
     * @brief Run inference on a 128x87 mel-spectrogram
     * @param mel_data Pointer to the flattened spectrogram [128 * 87]
     * @return 0 for Healthy, 1 for ALERT (Queenless/Swarm)
     */
    int predict(float* mel_data);

private:
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    // Arena size - Optimized for Int8 Quantized ResNet/MobileNet
    // Previously 700KB, reduced to 250KB to leave room for WiFi/Bluetooth stacks on ESP32.
    static constexpr int kArenaSize = 1024 * 250; 
    uint8_t tensor_arena[kArenaSize];
};

#endif
