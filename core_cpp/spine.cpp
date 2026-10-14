#include <mujoco/mujoco.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Policy {
    int layers = 0;
    int observation_size = 0;
    int action_size = 0;
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> biases;

    static bool read_spec_value(const std::string& contents,
                                const std::string& key, int& value) {
        const std::string quoted_key = "\"" + key + "\"";
        const std::size_t key_position = contents.find(quoted_key);
        if (key_position == std::string::npos) {
            return false;
        }
        const std::size_t colon = contents.find(':', key_position + quoted_key.size());
        if (colon == std::string::npos) {
            return false;
        }
        const char* value_start = contents.c_str() + colon + 1;
        char* value_end = nullptr;
        const long parsed = std::strtol(value_start, &value_end, 10);
        if (value_start == value_end || parsed <= 0 ||
            parsed > std::numeric_limits<int>::max()) {
            return false;
        }
        value = static_cast<int>(parsed);
        return true;
    }

    static bool read_floats(const std::string& path,
                            std::vector<float>& values) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) {
            std::fprintf(stderr, "[spine] cannot open policy file: %s\n",
                         path.c_str());
            return false;
        }
        const std::streamoff size = input.tellg();
        if (size <= 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
            std::fprintf(stderr, "[spine] invalid float file size: %s\n",
                         path.c_str());
            return false;
        }
        values.resize(static_cast<std::size_t>(size) / sizeof(float));
        input.seekg(0, std::ios::beg);
        input.read(reinterpret_cast<char*>(values.data()), size);
        if (!input || !std::all_of(values.begin(), values.end(),
                                   [](float value) { return std::isfinite(value); })) {
            std::fprintf(stderr, "[spine] invalid policy values: %s\n",
                         path.c_str());
            return false;
        }
        return true;
    }

    bool load(const std::string& directory) {
        std::ifstream spec_file(directory + "/policy_spec.json");
        if (!spec_file) {
            std::fprintf(stderr, "[spine] cannot read policy spec in %s\n",
                         directory.c_str());
            return false;
        }
        const std::string spec(
            (std::istreambuf_iterator<char>(spec_file)),
            std::istreambuf_iterator<char>());
        if (!read_spec_value(spec, "layers", layers) ||
            !read_spec_value(spec, "obs", observation_size) ||
            !read_spec_value(spec, "act", action_size) ||
            layers > 32) {
            std::fprintf(stderr, "[spine] invalid policy spec\n");
            return false;
        }

        weights.resize(layers);
        biases.resize(layers);
        int input_size = observation_size;
        for (int layer = 0; layer < layers; ++layer) {
            const std::string prefix = directory + "/";
            if (!read_floats(prefix + "w" + std::to_string(layer) + ".bin",
                             weights[layer]) ||
                !read_floats(prefix + "b" + std::to_string(layer) + ".bin",
                             biases[layer])) {
                return false;
            }
            const int output_size = static_cast<int>(biases[layer].size());
            if (output_size <= 0 ||
                weights[layer].size() !=
                    static_cast<std::size_t>(input_size) * output_size ||
                (layer == layers - 1 && output_size != action_size)) {
                std::fprintf(stderr,
                             "[spine] inconsistent dimensions in layer %d\n",
                             layer);
                return false;
            }
            input_size = output_size;
        }
        return true;
    }

    bool forward(const std::vector<float>& input,
                 std::vector<float>& output) const {
        if (static_cast<int>(input.size()) != observation_size) {
            return false;
        }
        std::vector<float> current = input;
        for (int layer = 0; layer < layers; ++layer) {
            const int output_size = static_cast<int>(biases[layer].size());
            const int input_size = static_cast<int>(current.size());
            std::vector<float> next(output_size);
            for (int output_index = 0; output_index < output_size; ++output_index) {
                float sum = biases[layer][output_index];
                for (int input_index = 0; input_index < input_size; ++input_index) {
                    sum += weights[layer][input_index * output_size + output_index]
                        * current[input_index];
                }
                next[output_index] =
                    layer < layers - 1 ? std::tanh(sum) : sum;
            }
            current.swap(next);
        }
        if (!std::all_of(current.begin(), current.end(),
                         [](float value) { return std::isfinite(value); })) {
            return false;
        }
        output = std::move(current);
        return true;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: spine_bin <model.xml> <policy_directory>\n");
        return 1;
    }

    Policy policy;
    if (!policy.load(argv[2])) {
        return 1;
    }

    char error[1000] = "could not load MuJoCo model";
    mjModel* model = mj_loadXML(argv[1], nullptr, error, sizeof(error));
    if (model == nullptr) {
        std::fprintf(stderr, "[spine] %s\n", error);
        return 1;
    }
    if (model->nq + model->nv != policy.observation_size ||
        model->nu != policy.action_size || model->nq < 7) {
        std::fprintf(stderr,
                     "[spine] model/policy mismatch: model obs=%d act=%d, "
                     "policy obs=%d act=%d\n",
                     static_cast<int>(model->nq + model->nv),
                     static_cast<int>(model->nu),
                     policy.observation_size, policy.action_size);
        mj_deleteModel(model);
        return 1;
    }

    model->opt.timestep = 0.002;
    mjData* data = mj_makeData(model);
    if (data == nullptr) {
        std::fprintf(stderr, "[spine] could not allocate MuJoCo data\n");
        mj_deleteModel(model);
        return 1;
    }
    mj_forward(model, data);

    std::printf("[spine] layers=%d obs=%d act=%d\n",
                policy.layers,
                static_cast<int>(model->nq + model->nv),
                static_cast<int>(model->nu));
    std::vector<float> joint_defaults(model->qpos0 + 7,
                                      model->qpos0 + 7 + model->nu);
    const int torso = mj_name2id(model, mjOBJ_BODY, "torso_link");
    if (torso < 0) {
        std::fprintf(stderr, "[spine] model has no torso_link body\n");
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }

    std::vector<float> observation(policy.observation_size);
    std::vector<float> action;
    constexpr long target_steps = 2500;
    constexpr long period_microseconds = 2000;
    const auto start = std::chrono::steady_clock::now();

    for (long step = 0; step < target_steps; ++step) {
        for (int index = 0; index < model->nq; ++index) {
            observation[index] = static_cast<float>(data->qpos[index]);
        }
        for (int index = 0; index < model->nv; ++index) {
            observation[model->nq + index] =
                static_cast<float>(data->qvel[index]);
        }
        if (!policy.forward(observation, action)) {
            std::fprintf(stderr,
                         "[spine] policy returned non-finite or invalid action\n");
            mj_deleteData(data);
            mj_deleteModel(model);
            return 1;
        }

        for (int index = 0; index < model->nu; ++index) {
            const float bounded_action =
                std::clamp(action[index], -1.0f, 1.0f);
            mjtNum control = joint_defaults[index] + 0.5f * bounded_action;
            if (model->actuator_ctrllimited[index]) {
                control = std::clamp(
                    control,
                    model->actuator_ctrlrange[2 * index],
                    model->actuator_ctrlrange[2 * index + 1]);
            }
            data->ctrl[index] = control;
        }

        mj_step(model, data);
        const mjtNum torso_height = data->xpos[3 * torso + 2];
        if (!std::isfinite(torso_height) ||
            !std::all_of(data->ctrl, data->ctrl + model->nu,
                         [](mjtNum value) { return std::isfinite(value); })) {
            std::fprintf(stderr, "[spine] simulation produced non-finite values\n");
            mj_deleteData(data);
            mj_deleteModel(model);
            return 1;
        }
        if (step % 500 == 0) {
            std::printf("[spine] t=%.1fs torso_z=%.3f ctrl0=%.2f\n",
                        step / 500.0, torso_height, data->ctrl[0]);
            std::fflush(stdout);
        }

        const auto deadline =
            start + std::chrono::microseconds((step + 1) * period_microseconds);
        std::this_thread::sleep_until(deadline);
    }

    std::printf("[spine] 500Hz loop complete\n");
    mj_deleteData(data);
    mj_deleteModel(model);
    return 0;
}
