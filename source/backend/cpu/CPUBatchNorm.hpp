//
//  CPUBatchNorm.hpp
//  MNN
//
//  Created by MNN on 2019/11/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBatchNorm_hpp
#define CPUBatchNorm_hpp

#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {
class CPUBatchNorm : public Execution {
public:
    CPUBatchNorm(Backend *backend, const MNN::Op *op);
    virtual ~CPUBatchNorm() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
} // namespace MNN

#endif /* CPUBatchNorm_hpp */
