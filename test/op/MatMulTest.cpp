//
//  MatMulTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"
#include "TestUtils.h"

using namespace MNN::Express;
class BatchMatMulTest : public MNNTestCase {
public:
    virtual ~BatchMatMulTest() = default;
    virtual bool run() {
        auto input_x = _Input({4,2}, NCHW);
        auto input_y = _Input({4,2}, NCHW);
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const float data_x[] = {-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0};
        const float data_y[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
        auto ptr_x          = input_x->writeMap<float>();
        auto ptr_y          = input_y->writeMap<float>();
        memcpy(ptr_x, data_x, 8 * sizeof(float));
        memcpy(ptr_y, data_y, 8 * sizeof(float));
        input_x->unMap();
        input_y->unMap();
        auto output = _MatMul(input_x, input_y, false, true);
        const std::vector<float> expectedOutput = {-3.0, -3.0, -3.0, -3.0,
                                                   -7.0, -7.0, -7.0, -7.0,
						   11.0, 11.0, 11.0, 11.0,
						   15.0, 15.0, 15.0, 15.0};
        auto gotOutput = output->readMap<float>();
        for(int i=0; i<16; i++) {
	    MNN_ERROR("%f!\n",gotOutput[i]);
        }
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.01)) {
            MNN_ERROR("BatchMatMulTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(BatchMatMulTest, "op/batchmatmul");
