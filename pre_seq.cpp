#include <iostream>                 // 标准输入输出流头文件
#include <fstream>                  // 文件流操作头文件
#include <vector>                   // 向量容器头文件
#include <string>                   // 字符串类头文件
#include <sstream>                  // 字符串流头文件
#include "onnxruntime_cxx_api.h"    // ONNX Runtime C++ API 头文件
using namespace std;                // 使用标准命名空间

// 定义结构体用于保存 LSTM 输出结果和状态
struct LSTMOutput {
    vector<float> output;           // LSTM 的输出结果
    vector<int64_t> output_shape;   // 输出结果的形状
    vector<float> hn;               // 最终隐藏状态
    vector<int64_t> hn_shape;       // 隐藏状态形状
    vector<float> cn;               // 最终细胞状态
    vector<int64_t> cn_shape;       // 细胞状态形状
};

// 从 CSV 文件读取数据，存入 data，并返回 batch_size 和 seq_len
bool load_csv(const string& filename, vector<float>& data, int64_t& batch_size, int64_t& seq_len) {
    ifstream ifs(filename);                 // 打开 CSV 文件
    if (!ifs.is_open()) return false;      // 打开失败则返回 false
    string line;
    vector<vector<float>> temp;            // 临时二维数组保存每行数据
    while (getline(ifs, line)) {           // 按行读取 CSV
        stringstream ss(line);
        string cell;
        vector<float> row;
        while (getline(ss, cell, ',')) {   // 按逗号分割每行
            row.push_back(stof(cell));     // 转换为 float 存入 row
        }
        if (!row.empty()) temp.push_back(move(row));  // 移动 row 到 temp
    }
    ifs.close();                           // 关闭文件
    batch_size = temp.size();              // 行数为 batch_size
    seq_len = temp.empty() ? 0 : temp[0].size();  // 列数为 seq_len
    data.reserve(batch_size * seq_len);    // 预分配内存
    for (auto& r : temp) {
        if (r.size() != seq_len) return false;  // 检查每行是否等长
        data.insert(data.end(), r.begin(), r.end()); // 插入到 data
    }
    return true;
}

// LSTM 推理类
class LSTMInferencer {
public:
    // 构造函数，加载 ONNX 模型
    LSTMInferencer(const wstring& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "lstm_infer"),
        session_options(),
        session(env, model_path.c_str(), session_options) {
        session_options.SetIntraOpNumThreads(1);  // 设置单线程
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // 启用所有优化
    }

    // 执行单步 LSTM 推理，输入当前 x、h_t、c_t，返回输出和新状态
    LSTMOutput infer_step(const vector<float>& x_t,
        const vector<int64_t>& in_shape,
        vector<float>& h_t, vector<float>& c_t,
        const vector<int64_t>& state_shape) {

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); // 创建 CPU 内存信息

        // 创建输入张量
        Ort::Value x_tensor = Ort::Value::CreateTensor<float>(
            mem_info, const_cast<float*>(x_t.data()), x_t.size(), in_shape.data(), in_shape.size());
        size_t state_elems = h_t.size();  // 状态元素个数
        Ort::Value h_tensor = Ort::Value::CreateTensor<float>(
            mem_info, h_t.data(), state_elems, state_shape.data(), state_shape.size());
        Ort::Value c_tensor = Ort::Value::CreateTensor<float>(
            mem_info, c_t.data(), state_elems, state_shape.data(), state_shape.size());

        // 定义输入输出名称
        const char* input_names[] = { "input", "h0", "c0" };
        const char* output_names[] = { "output", "hn", "cn" };

        // 执行推理
        array<Ort::Value, 3> inputs = { move(x_tensor), move(h_tensor), move(c_tensor) };
        auto outputs = session.Run(Ort::RunOptions{ nullptr },
            input_names, inputs.data(), inputs.size(),
            output_names, 3);

        LSTMOutput result;
        // 获取 output
        {
            auto& out0 = outputs[0];
            result.output_shape = out0.GetTensorTypeAndShapeInfo().GetShape();
            result.output.assign(out0.GetTensorMutableData<float>(),
                out0.GetTensorMutableData<float>() + out0.GetTensorTypeAndShapeInfo().GetElementCount());
        }
        // 获取 hn
        {
            auto& out1 = outputs[1];
            result.hn_shape = out1.GetTensorTypeAndShapeInfo().GetShape();
            result.hn.assign(out1.GetTensorMutableData<float>(),
                out1.GetTensorMutableData<float>() + out1.GetTensorTypeAndShapeInfo().GetElementCount());
        }
        // 获取 cn
        {
            auto& out2 = outputs[2];
            result.cn_shape = out2.GetTensorTypeAndShapeInfo().GetShape();
            result.cn.assign(out2.GetTensorMutableData<float>(),
                out2.GetTensorMutableData<float>() + out2.GetTensorTypeAndShapeInfo().GetElementCount());
        }

        // 更新传入的 h_t 和 c_t
        h_t = result.hn;
        c_t = result.cn;
        return result;  // 返回结果
    }

private:
    Ort::Env env;                   // ONNX 运行环境
    Ort::SessionOptions session_options; // 会话选项
    Ort::Session session;           // 推理会话
};

int main() {
    // 模型文件和输入数据文件路径
    wstring model_path = L"lstm_day_diff.onnx";
    string csv_path = "input.csv";

    // 读取 CSV 数据到 data 中
    vector<float> data;
    int64_t batch_size, total_seq_len;
    if (!load_csv(csv_path, data, batch_size, total_seq_len)) {
        cerr << "Failed to load CSV data.\n";
        return -1;
    }

    // LSTM 参数设定
    int64_t num_layers = 2;
    int64_t input_dim = 1;          // 每个时间步特征维度为1
    int64_t hidden_size = 50;       // 隐藏层维度为50

    // 提取每个样本的第一个值作为基线值
    vector<float> baseline(batch_size);
    for (int64_t b = 0; b < batch_size; ++b) {
        baseline[b] = data[b * total_seq_len + 0];
    }

    // 初始化 h_t 和 c_t 状态
    vector<int64_t> state_shape = { num_layers, batch_size, hidden_size };
    size_t state_elems = num_layers * batch_size * hidden_size;
    vector<float> h_t(state_elems, 0.0f), c_t(state_elems, 0.0f);

    // 创建 LSTM 推理器实例
    LSTMInferencer inferencer(model_path);

    // 存储所有时间步的预测结果
    vector<vector<float>> preds_all;
    preds_all.reserve(total_seq_len);

    // 对每个时间步进行逐步推理（输入长度随时间增长）
    int64_t offset = 1439;
    for (int64_t t = 0; t < offset; ++t) {
        int64_t curr_len = total_seq_len - offset + t;
        if (curr_len <= 0) continue; // 跳过长度小于等于0的情况

        // 构造输入 x_t: 差分值 [batch_size, curr_len, 1]
        vector<int64_t> in_shape = { batch_size, curr_len, input_dim };
        vector<float> x_t(batch_size * curr_len * input_dim);
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < curr_len; ++s) {
                float raw = data[b * total_seq_len + s];
                x_t[b * curr_len + s] = raw - baseline[b]; // 减去基线值
            }
        }

        // 推理当前时间步
        LSTMOutput out = inferencer.infer_step(x_t, in_shape, h_t, c_t, state_shape);
        preds_all.push_back(move(out.output)); // 存储输出
    }

    // 将预测结果保存到 CSV 文件
    ofstream ofs("output.csv");
    for (int64_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < preds_all.size(); ++t) {
            ofs << preds_all[t][b] << (t + 1 < preds_all.size() ? ',' : '\n'); // 写入结果
        }
    }
    ofs.close();

    cout << "Iterative differencing inference done, results saved to output.csv\n"; // 打印完成信息
    return 0;
}
