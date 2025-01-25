#pragma once
#ifndef OPENCVDETECTION_H_
#define OPENCVDETECTION_H_

#include<string>
#include<iostream>
#include<opencv2/dnn.hpp>

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

class Detection
{
public:
	Detection(std::string model_name, std::string config_name="", int backend=0, int target=0);
	~Detection();
    cv::Mat forward(cv::Mat img);
    

private:
	cv::dnn::Net net;
};

Detection::Detection(std::string model_name, std::string config_name, int backend, int target)
{   
    // read net
    if (config_name == "")
    {
        net = cv::dnn::readNetFromONNX(model_name);
    }
    else {
        //net = cv::dnn::readNet(model_name, config_name);
        net = cv::dnn::readNetFromModelOptimizer(config_name, model_name);
    }
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    std::cout << "Init model successfull....." << std::endl;
}

Detection::~Detection()
{
}
cv::Mat Detection::forward(cv::Mat img) {
    cv::Mat blob = cv::dnn::blobFromImage(img);

    net.setInput(blob);
    cv::Mat logits = net.forward();
    return logits;
}




#endif // !OPENCVDETECTION_H_
