//
//  MergeOptimizer.cpp
//  MNN
//
//  Created by MNN on 2019/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN/expr/optimizer/MergeOptimizer.hpp"
#include <map>
#include "BasicOptimizer_generated.h"
namespace MNN {
namespace Express {

MergeOptimizer::MergeOptimizer(MNNForwardType type, int numberThread, BackendConfig* config) {
    if (nullptr != config) {
        mConfig = *config;
    }
    mType         = type;
    mNumberThread = numberThread;
}

Optimizer::Cost MergeOptimizer::onMeasure(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters) {
    Cost cost;
    cost.compute = 0.0f;
    cost.memory  = 0.0f;
    return cost;
}
bool MergeOptimizer::onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters) {
    auto sequence = Variable::getExecuteOrder(outputs);
    if (1 == sequence.size()) {
        return true;
    }
    std::map<EXPRP, int> varIndexOffset;
    std::vector<EXPRP> queue;
    std::vector<VARP> inputs;
    std::unique_ptr<MNN::Optimizer::MergeT> merge(new MNN::Optimizer::MergeT);
    queue.reserve(sequence.size());
    merge->tensorNumber = sequence.size();
    merge->backend.reset(new MNN::Optimizer::BackendConfigT);
    merge->backend->numberThread = mNumberThread;
    merge->backend->type         = (MNN::ForwardType)mType;
    merge->backend->power        = (int)mConfig.power;
    merge->backend->precision    = (int)mConfig.precision;
    merge->backend->memroy       = (int)mConfig.memory;

    int tensorOffset = 0;
    for (int i = 0; i < sequence.size(); ++i) {
        auto expr      = sequence[i];
        varIndexOffset[expr] = tensorOffset;
        tensorOffset += expr->outputSize();
        if (expr->get()->type() == OpType_Input) {
            auto outputs = expr->outputs();
            for (auto iter : outputs) {
                auto var = iter.lock();
                if (nullptr != var) {
                    inputs.emplace_back(var);
                    break;
                }
            }
            merge->inputIndexes.emplace_back(i);
            continue;
        }
        queue.emplace_back(expr);
    }
    for (auto expr : queue) {
        std::unique_ptr<OpT> op(expr->get()->UnPack());
        auto outputIndexStart = varIndexOffset[expr];
        op->outputIndexes.resize(expr->outputSize());
        for (int i=0; i<expr->outputSize(); ++i) {
            op->outputIndexes[i] = outputIndexStart + i;
        }
        auto exprinputs       = expr->inputs();
        op->inputIndexes.resize(exprinputs.size());
        for (int i = 0; i < exprinputs.size(); ++i) {
            auto inputExpr = exprinputs[i]->expr();
            op->inputIndexes[i] = varIndexOffset[inputExpr.first] + inputExpr.second;
        }
        merge->oplists.emplace_back(std::move(op));
    }
    for (auto var : outputs) {
        auto expr = var->expr();
        merge->outputIndexes.emplace_back(varIndexOffset[expr.first] + expr.second);
    }

    std::unique_ptr<OpT> mergeOp(new OpT);
    mergeOp->type       = OpType_Extra;
    mergeOp->name       = outputs[0]->name();
    mergeOp->main.type  = OpParameter_Extra;
    mergeOp->main.value = new ExtraT;
    auto plugin         = mergeOp->main.AsExtra();
    plugin->type        = "Session";
    plugin->engine      = "MNN";

    flatbuffers::FlatBufferBuilder builder;
    auto offset = MNN::Optimizer::Merge::Pack(builder, merge.get());
    builder.Finish(offset);
    plugin->info.resize(builder.GetSize());
    ::memcpy(plugin->info.data(), builder.GetBufferPointer(), builder.GetSize());

    auto mergeExpr = Expr::create(mergeOp.get(), inputs, (int)outputs.size());
    mergeExpr->setName(outputs[0]->name());
    for (int i = 0; i < outputs.size(); ++i) {
        Variable::setExpr(outputs[i], mergeExpr, i);
    }
    return true;
}
} // namespace Express
} // namespace MNN
