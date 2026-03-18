#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

#include "nvdsinfer.h"

// Define types from nvdsinfer_custom_impl.h to avoid NvCaffeParser.h dependency
typedef struct {
    unsigned int numClassesConfigured;
    std::vector<float> perClassPreclusterThreshold;
    std::vector<float> perClassPostclusterThreshold;
    std::vector<float> &perClassThreshold = perClassPreclusterThreshold;
} NvDsInferParseDetectionParams;

typedef bool (* NvDsInferParseCustomFunc) (
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(customParseFunc) \
    static void checkFunc_ ## customParseFunc (NvDsInferParseCustomFunc func = customParseFunc) \
        { checkFunc_ ## customParseFunc (); }; \
    extern "C" bool customParseFunc (std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
           NvDsInferNetworkInfo const &networkInfo, \
           NvDsInferParseDetectionParams const &detectionParams, \
           std::vector<NvDsInferObjectDetectionInfo> &objectList);

static constexpr int NUM_CLASSES = 80;
static constexpr float CONF_THRESH = 0.25f;
static constexpr float NMS_THRESH = 0.45f;

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

static float iou(const Detection& a, const Detection& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

static std::vector<Detection> nms(std::vector<Detection>& dets, float threshold) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    std::vector<Detection> result;
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (!suppressed[j] && iou(dets[i], dets[j]) > threshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

extern "C" bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: No output layers" << std::endl;
        return false;
    }

    const float* output = (const float*)outputLayersInfo[0].buffer;
    if (!output) {
        std::cerr << "ERROR: Output buffer is null" << std::endl;
        return false;
    }

    const int num_proposals = 8400;
    std::vector<Detection> detections;

    for (int i = 0; i < num_proposals; i++) {
        float cx = output[0 * num_proposals + i];
        float cy = output[1 * num_proposals + i];
        float w  = output[2 * num_proposals + i];
        float h  = output[3 * num_proposals + i];

        float max_score = 0.0f;
        int best_class = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float score = output[(4 + c) * num_proposals + i];
            if (score > max_score) {
                max_score = score;
                best_class = c;
            }
        }

        if (max_score < CONF_THRESH) continue;

        Detection det;
        det.x1 = cx - w / 2.0f;
        det.y1 = cy - h / 2.0f;
        det.x2 = cx + w / 2.0f;
        det.y2 = cy + h / 2.0f;
        det.confidence = max_score;
        det.class_id = best_class;
        detections.push_back(det);
    }

    auto kept = nms(detections, NMS_THRESH);

    for (auto& d : kept) {
        NvDsInferObjectDetectionInfo obj;
        obj.classId = d.class_id;
        obj.detectionConfidence = d.confidence;
        obj.left   = d.x1 * networkInfo.width / 640.0f;
        obj.top    = d.y1 * networkInfo.height / 640.0f;
        obj.width  = (d.x2 - d.x1) * networkInfo.width / 640.0f;
        obj.height = (d.y2 - d.y1) * networkInfo.height / 640.0f;

        obj.left   = std::max(0.0f, obj.left);
        obj.top    = std::max(0.0f, obj.top);
        obj.width  = std::min((float)networkInfo.width - obj.left, obj.width);
        obj.height = std::min((float)networkInfo.height - obj.top, obj.height);

        objectList.push_back(obj);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8);
