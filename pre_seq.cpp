#include <iostream>                 // ��׼���������ͷ�ļ�
#include <fstream>                  // �ļ�������ͷ�ļ�
#include <vector>                   // ��������ͷ�ļ�
#include <string>                   // �ַ�����ͷ�ļ�
#include <sstream>                  // �ַ�����ͷ�ļ�
#include "onnxruntime_cxx_api.h"    // ONNX Runtime C++ API ͷ�ļ�
using namespace std;                // ʹ�ñ�׼�����ռ�

// ����ṹ�����ڱ��� LSTM ��������״̬
struct LSTMOutput {
    vector<float> output;           // LSTM ��������
    vector<int64_t> output_shape;   // ����������״
    vector<float> hn;               // ��������״̬
    vector<int64_t> hn_shape;       // ����״̬��״
    vector<float> cn;               // ����ϸ��״̬
    vector<int64_t> cn_shape;       // ϸ��״̬��״
};

// �� CSV �ļ���ȡ���ݣ����� data�������� batch_size �� seq_len
bool load_csv(const string& filename, vector<float>& data, int64_t& batch_size, int64_t& seq_len) {
    ifstream ifs(filename);                 // �� CSV �ļ�
    if (!ifs.is_open()) return false;      // ��ʧ���򷵻� false
    string line;
    vector<vector<float>> temp;            // ��ʱ��ά���鱣��ÿ������
    while (getline(ifs, line)) {           // ���ж�ȡ CSV
        stringstream ss(line);
        string cell;
        vector<float> row;
        while (getline(ss, cell, ',')) {   // �����ŷָ�ÿ��
            row.push_back(stof(cell));     // ת��Ϊ float ���� row
        }
        if (!row.empty()) temp.push_back(move(row));  // �ƶ� row �� temp
    }
    ifs.close();                           // �ر��ļ�
    batch_size = temp.size();              // ����Ϊ batch_size
    seq_len = temp.empty() ? 0 : temp[0].size();  // ����Ϊ seq_len
    data.reserve(batch_size * seq_len);    // Ԥ�����ڴ�
    for (auto& r : temp) {
        if (r.size() != seq_len) return false;  // ���ÿ���Ƿ�ȳ�
        data.insert(data.end(), r.begin(), r.end()); // ���뵽 data
    }
    return true;
}

// LSTM ������
class LSTMInferencer {
public:
    // ���캯�������� ONNX ģ��
    LSTMInferencer(const wstring& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "lstm_infer"),
        session_options(),
        session(env, model_path.c_str(), session_options) {
        session_options.SetIntraOpNumThreads(1);  // ���õ��߳�
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // ���������Ż�
    }

    // ִ�е��� LSTM �������뵱ǰ x��h_t��c_t�������������״̬
    LSTMOutput infer_step(const vector<float>& x_t,
        const vector<int64_t>& in_shape,
        vector<float>& h_t, vector<float>& c_t,
        const vector<int64_t>& state_shape) {

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); // ���� CPU �ڴ���Ϣ

        // ������������
        Ort::Value x_tensor = Ort::Value::CreateTensor<float>(
            mem_info, const_cast<float*>(x_t.data()), x_t.size(), in_shape.data(), in_shape.size());
        size_t state_elems = h_t.size();  // ״̬Ԫ�ظ���
        Ort::Value h_tensor = Ort::Value::CreateTensor<float>(
            mem_info, h_t.data(), state_elems, state_shape.data(), state_shape.size());
        Ort::Value c_tensor = Ort::Value::CreateTensor<float>(
            mem_info, c_t.data(), state_elems, state_shape.data(), state_shape.size());

        // ���������������
        const char* input_names[] = { "input", "h0", "c0" };
        const char* output_names[] = { "output", "hn", "cn" };

        // ִ������
        array<Ort::Value, 3> inputs = { move(x_tensor), move(h_tensor), move(c_tensor) };
        auto outputs = session.Run(Ort::RunOptions{ nullptr },
            input_names, inputs.data(), inputs.size(),
            output_names, 3);

        LSTMOutput result;
        // ��ȡ output
        {
            auto& out0 = outputs[0];
            result.output_shape = out0.GetTensorTypeAndShapeInfo().GetShape();
            result.output.assign(out0.GetTensorMutableData<float>(),
                out0.GetTensorMutableData<float>() + out0.GetTensorTypeAndShapeInfo().GetElementCount());
        }
        // ��ȡ hn
        {
            auto& out1 = outputs[1];
            result.hn_shape = out1.GetTensorTypeAndShapeInfo().GetShape();
            result.hn.assign(out1.GetTensorMutableData<float>(),
                out1.GetTensorMutableData<float>() + out1.GetTensorTypeAndShapeInfo().GetElementCount());
        }
        // ��ȡ cn
        {
            auto& out2 = outputs[2];
            result.cn_shape = out2.GetTensorTypeAndShapeInfo().GetShape();
            result.cn.assign(out2.GetTensorMutableData<float>(),
                out2.GetTensorMutableData<float>() + out2.GetTensorTypeAndShapeInfo().GetElementCount());
        }

        // ���´���� h_t �� c_t
        h_t = result.hn;
        c_t = result.cn;
        return result;  // ���ؽ��
    }

private:
    Ort::Env env;                   // ONNX ���л���
    Ort::SessionOptions session_options; // �Ựѡ��
    Ort::Session session;           // ����Ự
};

int main() {
    // ģ���ļ������������ļ�·��
    wstring model_path = L"lstm_day_diff.onnx";
    string csv_path = "input.csv";

    // ��ȡ CSV ���ݵ� data ��
    vector<float> data;
    int64_t batch_size, total_seq_len;
    if (!load_csv(csv_path, data, batch_size, total_seq_len)) {
        cerr << "Failed to load CSV data.\n";
        return -1;
    }

    // LSTM �����趨
    int64_t num_layers = 2;
    int64_t input_dim = 1;          // ÿ��ʱ�䲽����ά��Ϊ1
    int64_t hidden_size = 50;       // ���ز�ά��Ϊ50

    // ��ȡÿ�������ĵ�һ��ֵ��Ϊ����ֵ
    vector<float> baseline(batch_size);
    for (int64_t b = 0; b < batch_size; ++b) {
        baseline[b] = data[b * total_seq_len + 0];
    }

    // ��ʼ�� h_t �� c_t ״̬
    vector<int64_t> state_shape = { num_layers, batch_size, hidden_size };
    size_t state_elems = num_layers * batch_size * hidden_size;
    vector<float> h_t(state_elems, 0.0f), c_t(state_elems, 0.0f);

    // ���� LSTM ������ʵ��
    LSTMInferencer inferencer(model_path);

    // �洢����ʱ�䲽��Ԥ����
    vector<vector<float>> preds_all;
    preds_all.reserve(total_seq_len);

    // ��ÿ��ʱ�䲽�������������볤����ʱ��������
    int64_t offset = 1439;
    for (int64_t t = 0; t < offset; ++t) {
        int64_t curr_len = total_seq_len - offset + t;
        if (curr_len <= 0) continue; // ��������С�ڵ���0�����

        // �������� x_t: ���ֵ [batch_size, curr_len, 1]
        vector<int64_t> in_shape = { batch_size, curr_len, input_dim };
        vector<float> x_t(batch_size * curr_len * input_dim);
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < curr_len; ++s) {
                float raw = data[b * total_seq_len + s];
                x_t[b * curr_len + s] = raw - baseline[b]; // ��ȥ����ֵ
            }
        }

        // ����ǰʱ�䲽
        LSTMOutput out = inferencer.infer_step(x_t, in_shape, h_t, c_t, state_shape);
        preds_all.push_back(move(out.output)); // �洢���
    }

    // ��Ԥ�������浽 CSV �ļ�
    ofstream ofs("output.csv");
    for (int64_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < preds_all.size(); ++t) {
            ofs << preds_all[t][b] << (t + 1 < preds_all.size() ? ',' : '\n'); // д����
        }
    }
    ofs.close();

    cout << "Iterative differencing inference done, results saved to output.csv\n"; // ��ӡ�����Ϣ
    return 0;
}
