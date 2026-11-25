#include "embed.h"
#include <vitis/ai/facefeature.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <cmath>
#include <cstdio> // File operations and C-style input and output
#include <cstring> // memcpy, used to copy the face region out of the frame
#include <memory>

static std::unique_ptr<vitis::ai::FaceFeature> face_feature_net;

extern "C" int face_embedder_init(void) {
    try {
        // This can be found in vitis::ai:FaceFeature API Reference for 2.5
        face_feature_net = vitis::ai::FaceFeature::create("facerec_resnet20", true);
    }
    catch (const std::exception &e) {
        fprintf(stderr, "face_embedder_init: %s\n", e.what());
        return -1;
    }
    return face_feature_net ? 0 : -1;
}

/*
 * The frames in this pipeline are NV12 (VVAS_VFMT_Y_UV8_420), not BGR:
 *   - y_plane:  one byte per pixel (brightness), stride_bytes bytes per row
 *   - uv_plane: interleaved U and V, one pair per 2x2 pixels, half as many rows
 * The face model wants a BGR image, so we copy out only the face region of
 * both planes and convert that small piece to BGR.
 */
extern "C" int face_embedder_compute(const unsigned char *y_plane,
        const unsigned char *uv_plane,
        int frame_width, int frame_height,
        int stride_bytes, int box_x, int box_y, int box_width, int box_height,
        float *embedding_out) {
    if(!face_feature_net || !y_plane || !uv_plane || !embedding_out) return -1;

    // Keep the box inside the frame
    box_x = std::max(0, box_x);
    box_y = std::max(0, box_y);
    box_width = std::min(box_width, frame_width - box_x);
    box_height = std::min(box_height, frame_height - box_y);

    // NV12 shares one color sample per 2x2 pixels, so use even values
    box_x &= ~1;
    box_y &= ~1;
    box_width &= ~1;
    box_height &= ~1;
    if(box_width < 20 || box_height < 20) return -1; // if box is too small

    // Copy just the face region of both planes into one contiguous NV12 image:
    // box_height rows of Y followed by box_height / 2 rows of UV
    cv::Mat nv12_crop(box_height * 3 / 2, box_width, CV_8UC1);
    for(int row = 0; row < box_height; ++row) {
        std::memcpy(nv12_crop.ptr(row),
                    y_plane + (size_t)(box_y + row) * stride_bytes + box_x,
                    box_width);
    }
    for(int row = 0; row < box_height / 2; ++row) {
        std::memcpy(nv12_crop.ptr(box_height + row),
                    uv_plane + (size_t)(box_y / 2 + row) * stride_bytes + box_x,
                    box_width);
    }

    // Convert the NV12 face crop to the BGR image that the model expects
    cv::Mat face_crop;
    cv::cvtColor(nv12_crop, face_crop, cv::COLOR_YUV2BGR_NV12);

    try {
        //Returns FaceFeatureFloatResult, where run(...) is getting the running result of the neural network
        auto feature_result = face_feature_net->run(face_crop);
        const float *raw_feature = feature_result.feature->data();

        // Scale the vector to length 1 so two embeddings can be compared with a dot product
        float sum_of_square = 0.f;
        for(int i = 0; i < FACE_EMBEDDING_DIM; i++) {
            sum_of_square += raw_feature[i] * raw_feature[i];
        }
        float inverse_length = 1.f / std::sqrt(sum_of_square + 1e-12f);

        for(int i = 0; i < FACE_EMBEDDING_DIM; i++) {
            embedding_out[i] = raw_feature[i] * inverse_length;
        }
    }catch (const std::exception &e) {
        fprintf(stderr, "face_embedder_compute: %s\n", e.what());
        return -1;
    }
    return 0;
}

extern "C" void face_embedder_deinit(void) {face_feature_net.reset();}