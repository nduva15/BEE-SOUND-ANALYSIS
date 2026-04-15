#include <Arduino.h>
#include <stdint.h>
#include <string.h>
#include <driver/i2s.h>
#include "DeepBrainInference.h"
#include "params.h"

// --- I2S CONFIGURATION (INMP441) ---
const i2s_port_t I2S_PORT = I2S_NUM_0;
const int BLOCK_SIZE = 1024; // Samples per I2S read

void setup_i2s() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = (uint32_t)samplingRate,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = BLOCK_SIZE,
        .use_apll = false
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = 27,   // SCK
        .ws_io_num = 25,    // WS
        .data_out_num = -1,
        .data_in_num = 26   // SD
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
}

// --- GLOBAL OBJECTS ---
DeepBrainInference* brain = nullptr;
float* mel_buffer = nullptr;
int sample_idx = 0;

void setup() {
    Serial.begin(115200);
    Serial.println("🐝 BeeSound Analysis - Edge Firmware v3.1");

    setup_i2s();
    
    brain = new DeepBrainInference();
    if (brain->begin()) {
        Serial.println("✅ DeepBrain v3.1 Inference Engine Ready");
    } else {
        Serial.println("❌ Inference Engine Initialization Failed!");
    }

    // Allocate buffer for 2s window (128 bins x 87 steps)
    mel_buffer = new float[128 * 87];
    memset(mel_buffer, 0, 128 * 87 * sizeof(float));
}

void loop() {
    // --- REAL-TIME CAPTURE & INFERENCE ---
    // Note: In a production system, Feature Extraction (Mel-Spec)
    // would happen here on every hop_length (512 samples).
    
    int32_t i2s_samples[BLOCK_SIZE];
    size_t bytes_read;
    i2s_read(I2S_PORT, &i2s_samples, sizeof(i2s_samples), &bytes_read, portMAX_DELAY);

    // TODO: Implement FFT + Mel-Filterbank to populate mel_buffer
    // For now, we simulate completion of a 2s window
    bool window_complete = false; 

    if (window_complete && brain != nullptr) {
        int state = brain->predict(mel_buffer);
        
        if (state == 1) {
            Serial.println("🚨 ALERT: Anomalous Hive State (Queenless/Swarm) Detected!");
        } else {
            Serial.println("Healthy Baseline Recorded.");
        }
        
        // Prepare for next window
        memset(mel_buffer, 0, 128 * 87 * sizeof(float));
    }
}

