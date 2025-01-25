#ifndef OPENVINOIE_H
#define OPENVINOIE_H

#include<iostream>
#include<string>
#include<opencv.hpp>
#include<inference_engine.hpp>
#include<map>

using namespace InferenceEngine;
using namespace cv;
class detectionIE
{
public:
    detectionIE(std::string model_name, std::string config_name, int batchsize = 1);
    ~detectionIE();
    cv::Mat forward(const cv::Mat& img);
    bool check_valid(Mat m);

    cv::Point2f center;
    float center_size;

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

detectionIE::detectionIE(std::string model_name, std::string config_name, int batchsize)
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
}

detectionIE::~detectionIE()
{
}


bool detectionIE::check_valid(Mat m)
{
    int count = 0;
    for(int i=0;i<m.rows;i++)
    {
        for(int j=0;j<m.cols;j++)
        {
            if(m.at<uint8_t>(i, j) > 0)
                count++;
        }
    }
    return count < 10?true:false;
}

cv::Mat detectionIE::forward(const cv::Mat& img)
{
    const int org_height = img.size[0];
    const int org_width = img.size[1];

    cv::resize(img, temp_frame, cv::Size(resize_width, resize_height));
    temp_frame = temp_frame(cv::Range::all(), cv::Range(64, 448)); //center-crop -> 288, 384
    cv::cvtColor(temp_frame, temp_frame, cv::COLOR_BGR2RGB);
    cv::split(temp_frame, m_input_buffer_wrapper);

    m_infer_request.Infer();

    auto output_data = m_output->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

    cv::Mat maxCl = cv::Mat::zeros(height, width, CV_8UC1);
    size_t W = 384;
    size_t H = 288;
    size_t C = 3;
    size_t image_stride = W * H * C;
    for (int rowId = 0; rowId < H; ++rowId) {
        for (int colId = 0; colId < W; ++colId) {
            std::size_t classId = 0;
            if (0) {  // assume the output is already ArgMax'ed

            } else {
                float maxProb = -10.0f;
                for (int chId = 0; chId < C; ++chId) {
                    float prob = output_data[chId * H * W + rowId * W + colId];
                    if (prob > maxProb) {
                        classId = chId;
                        maxProb = prob;
                    }
                }
                maxCl.at<uint8_t>(rowId, colId) = (uint8_t)classId;
            }
        }
    }
    // get cirle center point and length
    cv::Mat cc;
    cv::Mat pointsf;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cc = (maxCl - 1)*(5);
    cv::threshold(cc, cc, 1, 255, cv::THRESH_BINARY);
    if(check_valid(cc))
    {
        center.x = -1;
        center.y = -1;
    }else{
        cv::Canny(cc, cc, 50, 100);
        cv::findContours(cc, contours, hierarchy, 0, 1);
        if(contours[0].size() <= 5)
        {
            center.x = -1;
            center.y = -1;
        }else{
            cv::Mat(contours[0]).convertTo(pointsf, CV_32F);
            cv::RotatedRect box = cv::fitEllipse(pointsf);
            center = box.center;
            center.x += 64;
            center.x *= float(org_width) / (W + 128);
            center.y *= float(org_height) / H;
        }
    }

    cc = maxCl * 10;
    cv::threshold(cc, cc, 1, 255, cv::THRESH_BINARY);
    if(check_valid(cc))
    {
        center_size = -1;
    }else{
        cv::Canny(cc, cc, 50, 100);
        cv::findContours(cc, contours, hierarchy, 0, 1);
        if(contours[0].size() <= 5)
        {
            center_size = -1;
        }else{
            cv::Mat(contours[0]).convertTo(pointsf, CV_32F);
            cv::RotatedRect box2 = cv::fitEllipse(pointsf);
            center_size = box2.size.height > box2.size.width ? box2.size.height : box2.size.width;
            center_size *= box2.size.height > box2.size.width ? float(org_height) / H : float(org_width) / (W + 128);
        }
    }

    cv::copyMakeBorder(maxCl, maxCl, 0, 0, 64, 64, cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::resize(maxCl, maxCl, cv::Size(org_width, org_height), cv::INTER_NEAREST);
    return maxCl;


}


#endif // OPENVINOIE_H
