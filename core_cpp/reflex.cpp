#include <mujoco/mujoco.h>
#include <iostream>
#include <chrono>
#include <thread>

mjModel* m = nullptr;
mjData* d = nullptr;

// This runs every physics step (500Hz)
void controller(const mjModel* m, mjData* d) {
    // Dummy balance reflex: apply upward force to torso to counter gravity
    int torso_id = mj_name2id(m, mjOBJ_BODY, "torso_link");
    if (torso_id < 0) torso_id = mj_name2id(m, mjOBJ_BODY, "torso");
    
    if (torso_id >= 0) {
        // Add 80 Newtons upward (Z-axis) to the torso
        d->xfrc_applied[6 * torso_id + 2] = 80.0; 
    }
}

int main(int argc, const char** argv) {
    if (argc < 2) { std::cerr << "Usage: reflex <model.xml>\n"; return 1; }
    
    char error[1000] = "Could not load model";
    m = mj_loadXML(argv[1], nullptr, error, 1000);
    if (!m) { std::cerr << error << "\n"; return 1; }
    
    d = mj_makeData(m);
    mjcb_control = controller; // Attach our reflex

    auto start = std::chrono::steady_clock::now();
    int steps = 0;
    int target_steps = 2500; // 5 seconds at 500Hz
    
    std::cout << "[C++ Spine] Starting 500Hz physics loop...\n";

    while (steps < target_steps) {
        mj_step(m, d);
        steps++;
        
        // Strict 500Hz timing (2000 microseconds per step)
        auto now = std::chrono::steady_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        auto target_us = steps * 2000; 
        
        if (target_us > elapsed_us) {
            std::this_thread::sleep_for(std::chrono::microseconds(target_us - elapsed_us));
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "[C++ Spine] Done. " << steps << " steps in " << total_ms << "ms.\n";
    std::cout << "[C++ Spine] Real-time factor: " << (5000.0 / total_ms) << "x\n";

    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
