#include <assert.h>
#include <vector>
#include "ReLU_Backward.h"

ReLU_Backward::ReLU_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. input_height    3. input_width
    input_n    = args[0];   input_c    = args[1];   input_h    = args[2];   input_w  = args[3];
    input_ratio = args[4]; output_ratio = args[5];

    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&d_input_descriptor));
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_descriptor));
    CUDNN_CALL(cudnnSetActivationDescriptor(activation_descriptor,
                                            CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN,
                                            0 /* RELU_coef */));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&d_output_descriptor));

    // SetInputDescriptor
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            d_input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    // SetOutputDescriptor
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            output_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            d_output_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&output_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        GPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
        GPUFillRand(output_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
        GPUFillRand(d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
    }
    cudaDeviceSynchronize();
}

ReLU_Backward::~ReLU_Backward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(d_input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(d_output_descriptor));
    if (!is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(d_input_data));
        CUDA_CALL(cudaFree(output_data));
        CUDA_CALL(cudaFree(d_output_data));
    }
}

float ReLU_Backward::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&output_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
        CPUFillRand(output_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
        CPUFillRand(d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float));

        GPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * input_ratio);
        GPUFillRand(d_input_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * output_ratio);
        GPUFillRand(d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float one = 1;
    float zero = 0;
    
    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    CUDNN_CALL(cudnnActivationBackward(
            handle,
            activation_descriptor,
            &one,
            output_descriptor,
            output_data,
            d_output_descriptor,
            d_output_data,
            input_descriptor,
            input_data,
            &zero,
            d_input_descriptor,
            d_input_data));
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(d_input_data));
        CUDA_CALL(cudaFree(output_data));
        CUDA_CALL(cudaFree(d_output_data));
    }
    return milliseconds;
}







Softmax_Backward::Softmax_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. input_height    3. input_width
    input_n    = args[0];   input_c    = args[1];   input_h    = args[2];   input_w  = args[3];
    input_ratio = args[4]; output_ratio = args[5];

    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&d_input_descriptor));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&d_output_descriptor));

    // SetInputDescriptor
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            d_input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    // SetOutputDescriptor

    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            d_output_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        GPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
        GPUFillRand(d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
    }
    cudaDeviceSynchronize();
}

Softmax_Backward::~Softmax_Backward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(d_input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(d_output_descriptor));
    if (!is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(d_input_data));
        CUDA_CALL(cudaFree(d_output_data));
    }
}

float Softmax_Backward::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));

        CUDA_CALL(cudaMallocManaged(&d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float));

        CPUFillRand(d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float));

        GPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * input_ratio);
        GPUFillRand(d_input_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * output_ratio);
        GPUFillRand(d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float one = 1;
    float zero = 0;
    
    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    CUDNN_CALL(cudnnSoftmaxBackward(
            handle,
            algorithm,
            mode,
            &one,
            input_descriptor,
            input_data,
            d_output_descriptor,
            d_output_data,
            &zero,
            d_input_descriptor,
            d_input_data));
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(d_input_data));
        CUDA_CALL(cudaFree(d_output_data));
    }
    return milliseconds;
}
