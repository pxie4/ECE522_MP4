#pragma once
#include "Utils.cuh"
#include <vector>
#include <cudnn.h>

using std::vector;

class Conv2d_Backward_Input : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        cudnnTensorDescriptor_t input_descriptor;
        cudnnFilterDescriptor_t filter_descriptor;
        cudnnTensorDescriptor_t d_output_descriptor;
        cudnnTensorDescriptor_t d_input_descriptor;
        cudnnConvolutionDescriptor_t convolution_descriptor;

        cudnnConvolutionBwdDataAlgo_t algorithm;
        
        size_t workspace_size;
        float *workspace_data;

        long input_n, input_c, input_h, input_w;
        long filter_n, filter_c, filter_h, filter_w;
        int output_n, output_c, output_h, output_w;
        double input_ratio, output_ratio;

        long padding_h, padding_w;
        long stride_h, stride_w;
        long dilation_h, dilation_w;

        const float alpha = 1.f;
        const float beta = 0.f;

        float *d_input_data;
        float *filter_data;
        float *d_output_data;

        Conv2d_Backward_Input(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Conv2d_Backward_Input();
        float Run();
        size_t getWorkspaceSize();
};
