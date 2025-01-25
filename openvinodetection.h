#ifndef OPENVINODETECTION_H
#define OPENVINODETECTION_H


#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include<opencv2/imgproc.hpp>
#include <iostream>

/*
    "{ backend     | 0 | Choose one of computation backends: "
                        "0: automatically (by default), "
                        "1: Halide language (http://halide-lang.org/), "
                        "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "3: OpenCV implementation }"
    "{ target      | 0 | Choose one of target computation devices: "
                        "0: CPU target (by default), "
                        "1: OpenCL, "
                        "2: OpenCL fp16 (half-float precision), "
                        "3: VPU }";
*/

class OpenvinoDetection
{
public:
    OpenvinoDetection(std::string model_name, std::string config_name, int backend = cv::dnn::DNN_BACKEND_INFERENCE_ENGINE , int target = cv::dnn::DNN_TARGET_CPU);
    ~OpenvinoDetection();
    cv::Mat forward(const cv::Mat &img);

private:
    cv::dnn::Net m_net;
    cv::Mat temp_frame;
    cv::Mat blob;

    int height = 288;
    int width = 384;
    int resize_height = 288;
    int resize_width = 512;
};

OpenvinoDetection::OpenvinoDetection(std::string model_name, std::string config_name, int backend, int target)
{
    std::cout << "Init model.................." << std::endl;
    m_net = cv::dnn::readNetFromModelOptimizer(config_name, model_name);
    m_net.setPreferableBackend(backend);
    m_net.setPreferableTarget(target);
    std::cout << "Model loading successful............." << std::endl;
}

OpenvinoDetection::~OpenvinoDetection()
{
}
cv::Mat OpenvinoDetection::forward(const cv::Mat &img)
{
    const int org_height = img.size[0];
    const int org_width = img.size[1];

    cv::resize(img, temp_frame, cv::Size(resize_width, resize_height)); //
    temp_frame = temp_frame(cv::Range::all(), cv::Range(64, 448)); //center-crop -> 288, 384
    cv::cvtColor(temp_frame, temp_frame, cv::COLOR_BGR2RGB); //

    blob = cv::dnn::blobFromImage(temp_frame, 1.0, cv::Size(384, 288), cv::Scalar(), false, false);
    m_net.setInput(blob);

    cv::Mat detection = m_net.forward(); // 1 x 1 x h x w
    const int rows = detection.size[2];
    const int cols = detection.size[3];
    cv::Mat result(rows, cols, CV_32F, detection.ptr<float>()); //h x w, logits map
    cv::Mat maxCl = cv::Mat::zeros(rows, cols, CV_8UC1);
    for (int row = 0; row < rows; row++)
    {
        const float* ptrScore = result.ptr<float>(row);
        uint8_t* ptrMaxCl = maxCl.ptr<uint8_t>(row);
        for (int col = 0; col  < cols; col ++)
        {
            if (ptrScore[col] > 0.)
            {
                ptrMaxCl[col] = (uchar)1;
            }
        }
    }
    cv::copyMakeBorder(maxCl, maxCl, 0, 0, 64, 64, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::resize(maxCl, maxCl, cv::Size(org_width, org_height),cv::INTER_NEAREST);
    return maxCl;
}


#endif // OPENVINODETECTION_H
