//
//  transformer.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <MNN/Interpreter.hpp>
#include "OpConverter.hpp"
#include "core/Macro.h"
#include "OpGrad.hpp"
#include <MNN/expr/ExprCreator.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "rapidjson/document.h"

using namespace MNN;
using namespace MNN::Express;
using namespace std;

static void constToTrainableParam(EXPRP expr) {
    auto constOpT = expr->get()->UnPack();
    auto constBlobT = constOpT->main.AsBlob();

    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_TrainableParam;
    op->main.type  = OpParameter_Blob;
    op->main.value = new BlobT(*constBlobT);

    expr->set(op.get());
}

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./transformer.out temp.bin dst.bin config.json\n");
        return 0;
    }
    rapidjson::Document document;
    {
        std::ifstream fileNames(argv[3]);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        FUNC_PRINT(document.HasParseError());
        FUNC_PRINT(document.IsArray());
        FUNC_PRINT(document.IsObject());
    }
    auto configObject = document.GetObject();
    std::vector<std::string> variableLimits;
    if (configObject.HasMember("Optimizor")) {
        auto optimizor = configObject["Optimizor"].GetObject();
        if (optimizor.HasMember("Variables")) {
            auto limitArray = optimizor["Variables"].GetArray();
            for (auto vIter = limitArray.begin(); vIter != limitArray.end(); vIter++) {
                variableLimits.emplace_back(vIter->GetString());
                MNN_PRINT("Variabale contain : %s \n", vIter->GetString());
            }
        }
    }
    const char* inputModeFileName = argv[1];
    FUNC_PRINT_ALL(inputModeFileName, s);
    auto inputsOutputs = Variable::getInputAndOutput(Variable::loadMap(argv[1]));
    auto exprs = Variable::getExecuteOrder(Variable::mapToSequence(inputsOutputs.second));
    if (configObject.HasMember("Shape")) {
        auto shapeArray = configObject["Shape"].GetObject();
        for (auto shapeIter = shapeArray.begin(); shapeIter != shapeArray.end(); shapeIter++) {
            auto dimArray = shapeIter->value.GetArray();
            std::vector<int> dims;
            for (auto dimIter = dimArray.begin(); dimIter != dimArray.end(); dimIter++) {
                dims.emplace_back(dimIter->GetInt());
            }
            FUNC_PRINT_ALL(shapeIter->name.GetString(), s);
            std::string key = shapeIter->name.GetString();
            for (auto& var : exprs) {
                if (var->name() == key) {
                    auto tempVar = Variable::create(var);
                    tempVar->resize(dims);
                    break;
                }
            }
        }
    }
    {
        AUTOTIME;
        // Turn convolution be trainable convolution
        for (auto expr : exprs) {
            FUNC_PRINT_ALL(expr->name().c_str(), s);
            auto newExpr = OpConverter::convert(expr);
            if (newExpr.get() != expr.get()) {
                auto outputs = expr->outputs();
                for (auto o : outputs) {
                    auto var = o.lock();
                    if (nullptr != var) {
                        Variable::setExpr(var, newExpr, var->expr().second);
                    }
                }
            }
        }
    }
    exprs = Variable::getExecuteOrder(Variable::mapToSequence(inputsOutputs.second));

    // Collect trainable param (note: trainable param must be leaf node)
    std::set<EXPRP> updateExprs;
    for (auto expr : exprs) {
        if (expr->get()->type() == OpType_Const) {
            auto name = expr->name();
            bool match = variableLimits.empty();
            for (auto limit : variableLimits) {
                if (name.find(limit) != std::string::npos) {
                    match = true;
                    break;
                }
            }
            if (match) {
                MNN_PRINT("Add Variable: %s\n", name.c_str());
                constToTrainableParam(expr);
                updateExprs.insert(expr);
            }
        }
    }

    VARP loss;
    bool hasLoss      = configObject.HasMember("Loss");
    if (!hasLoss) {
        auto output = inputsOutputs.second.begin()->second;
        auto outputShape = output->getInfo();
        if (outputShape->order == NC4HW4) {
            auto outputName = output->name();
            output->setName(outputName + "Origin");
            output = _Convert(output, NHWC);
            outputShape = output->getInfo();
            output->setName(outputName);
        }
        auto outputReal = _Input(outputShape->dim, outputShape->order);
        outputReal->setName(output->name() + "_Compare");
#ifdef USE_ELU
        auto sub = _Subtract(output, outputReal);
        sub->setName(output->name() + "_Sub");
        loss = (_ReduceSum(_Multiply(sub, sub), {}));
#else
        auto mul = _Multiply(_Log(output), outputReal);
        mul->setName(output->name() + "_Mul");
        loss = _Negative(_ReduceSum(mul, {}));
#endif
        auto l2 = _Const(0.0f);
        for (auto expr : updateExprs) {
            auto var = expr->outputs().begin()->lock();
            MNN_ASSERT(nullptr != var);
            l2 = _Add(l2, _ReduceSum(_Multiply(var, var), {}));
        }
        loss = _Add(loss, _Multiply(l2, _Const(0.0005f)));
        loss->setName("Loss");
        inputsOutputs.second.insert(std::make_pair("Loss", loss));
        exprs = Variable::getExecuteOrder(Variable::mapToSequence( inputsOutputs.second));
    } else {
        for (auto expr : exprs) {
            std::unique_ptr<OpT> opT(expr->get()->UnPack());
            if (opT->name == configObject["Loss"].GetObject()["op"].GetString()) {
                loss = Variable::create(expr);
                break;
            }
        }
    }
    MNN_ASSERT(nullptr != loss);
    std::map<EXPRP, std::vector<VARP>> backwardMap;
    {
        auto shape = loss->getInfo();
        MNN_ASSERT(shape->size == 1);
        auto init = _Const(1.0f, shape->dim, shape->order);
        backwardMap[loss->expr().first] = std::vector<VARP>{init};
    }
    {
        AUTOTIME;
        std::map<EXPRP, int> exprRef;
        std::stack<EXPRP> exprExecuteOrder;
        std::map<EXPRP, std::vector<bool>> exprExecuted;
        for (auto expr : exprs) {
            exprExecuteOrder.push(expr);
        }
        while (!exprExecuteOrder.empty()) {
            auto expr = exprExecuteOrder.top();
            exprExecuteOrder.pop();
            auto& inputs = expr->inputs();
            if (backwardMap.find(expr) == backwardMap.end()) {
                continue;
            }
            auto grad = OpGrad::get(expr->get()->type());
            if (nullptr == grad) {
                continue;
            }
            std::vector<VARP> outputs(expr->outputSize());
            for (auto v : expr->outputs()) {
                auto vp = v.lock();
                if (nullptr == vp) {
                    continue;
                }
                outputs[vp->expr().second] = vp;
            }
            auto inputGrad = grad->onGrad(expr, outputs, backwardMap[expr]);
            if (inputGrad.empty()) {
                continue;
            }
            MNN_ASSERT(inputGrad.size() == inputs.size());
            for (int i=0; i<inputs.size(); ++i) {
                auto inputExpr = inputs[i]->expr().first;
                auto index = inputs[i]->expr().second;
                auto backward = inputGrad[i];
                if (nullptr == backward) {
                    continue;
                }
                if (backwardMap.find(inputExpr) == backwardMap.end()) {
                    backwardMap.insert(std::make_pair(inputExpr, std::vector<VARP>(inputExpr->outputSize())));
                }
                auto& inputVarMap = backwardMap[inputExpr];
                if (nullptr == inputVarMap[index]) {
                    inputVarMap[index] = backward;
                } else {
                    inputVarMap[index] = _Add(inputVarMap[index], backward);
                }
            }
        }
    }
    //Make Update
    std::map<VARP, VARP> varUpdateMap;
    auto learningRate = _Input();
    learningRate->setName("LearningRate");
    for (auto expr : updateExprs) {
        auto iter = backwardMap.find(expr);
        if (iter == backwardMap.end()) {
            continue;
        }
        auto& vars = iter->second;
        MNN_ASSERT(vars.size() == 1);
        auto originVar = expr->outputs();
        auto var = originVar.begin()->lock();
        MNN_ASSERT(nullptr != var);
        vars[0] = _Subtract(var, _Multiply(vars[0], learningRate));
        vars[0]->setName("update_" + var->name());
        varUpdateMap[var] = vars[0];
    }
    std::unique_ptr<MNN::NetT> netStruct(new MNN::NetT);
    netStruct->usage = Usage_TRAIN;
    std::vector<VARP> resultOutputs{loss};
    for (auto output : inputsOutputs.second) {
        resultOutputs.emplace_back(output.second);
    }
    for (auto iter : varUpdateMap) {
        resultOutputs.emplace_back(iter.second);
    }
    Variable::save(resultOutputs, netStruct.get());
    for (int i=0; i<netStruct->oplists.size(); ++i) {
        auto& op = netStruct->oplists[i];
        for (auto iter : varUpdateMap) {
            if (iter.second->name() == op->name) {
                for (int j=0; j<netStruct->oplists.size(); ++j) {
                    auto& opSub = netStruct->oplists[j];
                    if (opSub->name == iter.first->name()) {
                        op->outputIndexes = opSub->outputIndexes;
                    }
                }
            }
        }
    }
    {
        flatbuffers::FlatBufferBuilder builder(1024);
        auto offset = Net::Pack(builder, netStruct.get());
        builder.Finish(offset);
        // TODO, use FileWriter instead
        FILE* f = fopen(argv[2], "wb");
        fwrite(builder.GetBufferPointer(), 1, builder.GetSize(), f);
        fclose(f);
    }

    return 0;
}
