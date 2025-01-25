#pragma once



#include<iostream>
#include<string>
#include<opencv.hpp>
#include<inference_engine.hpp>
#include<map>

#include<math.h>

using namespace InferenceEngine;

//using namespace cv;
//using namespace cv::dnn;

class ClsLens
{
public:
    ClsLens(std::string model_name, std::string config_name, int batchsize, int fps);
    double forward(const cv::Mat& img);

    int fps_break;
    int count;
    bool flagKai;
private:
    CNNNetwork m_network;
    InputInfo::Ptr m_input_info;
    DataPtr m_output_info;
    InferRequest m_infer_request;
    Blob::Ptr m_input;
    Blob::Ptr m_output;
    int m_batch_size;

    unsigned char* m_input_buffer;
    size_t m_input_height;
    size_t m_input_width;

    std::vector<cv::Mat> m_input_buffer_wrapper;

    int height = 288;
    int width = 384;
    int resize_height = 288;
    int resize_width = 512;
    cv::Mat temp_frame;


};


ClsLens::ClsLens(std::string model_name, std::string config_name, int batchsize, int fps)
{
    std::cout << "Init Model........." << std::endl;
    Core ie;
    CNNNetReader network_reader;
    network_reader.ReadNetwork(config_name);
    network_reader.ReadWeights(model_name);
    m_network = network_reader.getNetwork();
    m_input_info = m_network.getInputsInfo().begin()->second;
    std::string input_name = m_network.getInputsInfo().begin()->first;
    m_input_info->setLayout(Layout::NCHW);
    m_input_info->setPrecision(Precision::U8);

    m_output_info = m_network.getOutputsInfo().begin()->second;
    std::string output_name = m_network.getOutputsInfo().begin()->first;
    m_output_info->setPrecision(Precision::FP32);

    std::map<std::string, std::string> config = { {PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::NO}};

    m_network.setBatchSize(batchsize);
    m_batch_size = batchsize;

    ExecutableNetwork executable_network = ie.LoadNetwork(m_network, "CPU", config);
    m_infer_request = executable_network.CreateInferRequest();
    m_input = m_infer_request.GetBlob(input_name);
    m_output = m_infer_request.GetBlob(output_name);

    m_input_width = m_input->getTensorDesc().getDims()[3];
    m_input_height = m_input->getTensorDesc().getDims()[2];

    m_input_buffer = static_cast<unsigned char*>(m_input->buffer());
    m_input_buffer_wrapper.emplace_back(m_input_height, m_input_width, CV_8UC1, m_input_buffer);
    m_input_buffer_wrapper.emplace_back(m_input_height, m_input_width, CV_8UC1, m_input_buffer + m_input_height * m_input_width);
    m_input_buffer_wrapper.emplace_back(m_input_height, m_input_width, CV_8UC1, m_input_buffer + 2 * m_input_height * m_input_width);

    std::cout << "Model loading SUCCESSFUL!!!!!" << std::endl;

    fps_break = fps * 8;
    count = 0;
    flagKai = false;
}


double ClsLens::forward(const cv::Mat& img)
{
    const int org_height = img.size[0];
    const int org_width = img.size[1];

    cv::resize(img, temp_frame, cv::Size(resize_width, resize_height));
    temp_frame = temp_frame(cv::Range::all(), cv::Range(64, 448)); //center-crop -> 288, 384
    cv::cvtColor(temp_frame, temp_frame, cv::COLOR_BGR2RGB); // 转换为rgb
    cv::split(temp_frame, m_input_buffer_wrapper);

    m_infer_request.Infer();

    auto output_data = m_output->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    double logit{*output_data};
    logit = 1. / (1+ std::exp(-logit));

    if(!flagKai)
    {
        if(logit >= 0.75)
        {
            count++;
        }else{
            count = 0;
        }
        if(count >= fps_break)
        {
            flagKai = true;
        }

    }

    return logit;
}

