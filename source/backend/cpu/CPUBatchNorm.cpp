//
//  CPUBatchNorm.cpp
//  MNN
//
//  Created by MNN on 2019/11/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUBatchNorm.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include <MNN/MNNDefine.h>
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include <vector>

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

CPUBatchNorm::CPUBatchNorm(Backend* backend, const MNN::Op* op) : Execution(backend) {
    
}

ErrorCode CPUBatchNorm::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // input, scale, bias, running_mean, running_variance, epsilon, momentum, is_training
    // scale and bias are learnable
    // Note: running_mean and running_variance initialization should not be handled here
    MNN_ASSERT(inputs.size() == 8);
    // output_feature, new_running_mean, new_running_variance, normalized_data, 1 / sqrt(sample_variance + epsilon), later two for compute gradients convinient
    MNN_ASSERT(outputs.size() == 5);
    auto input = inputs[0];
    auto inputData = input->host<float>();
    MNN_ASSERT(input->dimensions() == 4);
    MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NCHW);
    auto scale = inputs[1]->host<float>();
    auto bias = inputs[2]->host<float>();
    auto runningMean = inputs[3]->host<float>();
    auto runningVar = inputs[4]->host<float>();
    auto epsilon = inputs[5]->host<float>()[0];
    auto momentum = inputs[6]->host<float>()[0];
    auto isTraining = inputs[7]->host<bool>()[0];
    auto output = outputs[0];
    auto outputData = output->host<float>();
    auto newRunningMean = outputs[1]->host<float>();
    auto newRunningVar = outputs[2]->host<float>();
    auto normalizedData = outputs[3]->host<float>(); // (input - sample_mean) / sqrt(sample_variance + epsilon)
    auto rSampleStd = outputs[4]->host<float>(); // 1 / sqrt(sample_variance + epsilon)
    MNN_ASSERT(inputs[1]->elementSize() == input->channel());
    MNN_ASSERT(inputs[3]->elementSize() == input->channel());

    const int batchDataCount = input->buffer().dim[0].stride; // C * H * W
    const int spacialDataCount = input->buffer().dim[1].stride; // H * W
    const int inputCount = input->elementSize();

    if (isTraining) { // train
        std::vector<float> sampleMean(input->channel(), 0);
        std::vector<float> sampleVar(input->channel(), 0);

        for (int i = 0; i < inputCount; i++) {
            int index = (i % batchDataCount) / spacialDataCount;
            sampleMean[index] += (inputData[i] / spacialDataCount/ input->batch());
        }
        for (int i = 0; i < inputCount; i++) {
            int index = (i % batchDataCount) / spacialDataCount;
            sampleVar[index] += (
                ((inputData[i] - sampleMean[index]) * (inputData[i] - sampleMean[index]))
                 / spacialDataCount / input->batch()
                );
        }
        for (int i = 0; i < input->channel(); i++) {
            rSampleStd[i] = 1 / sqrt(sampleVar[i] + epsilon);
        }
        for (int i = 0; i < inputCount; i++) {
            int index = (i % batchDataCount) / spacialDataCount;
            normalizedData[i] = (inputData[i] - sampleMean[index]) * rSampleStd[index];
            outputData[i] = normalizedData[i] * scale[index] + bias[index];
        }
        for (int i = 0; i < input->channel(); i++) {
            newRunningMean[i] = momentum * runningMean[i] + (1 - momentum) * sampleMean[i];
            newRunningVar[i] = momentum * runningVar[i] + (1 - momentum) * sampleVar[i];
        }
    }
    else { // test
        for (int i = 0; i < inputCount; i++) {
            int index = (i % batchDataCount) / spacialDataCount;
            outputData[i] = (inputData[i] - runningMean[index]) / sqrt(runningVar[index] + epsilon);
            outputData[i] = outputData[i] * scale[index] + bias[index];
        }
    }

    return NO_ERROR;
}

class CPUBatchNormCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUBatchNorm(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUBatchNormCreator, OpType_BatchNorm);

} // namespace MNN
