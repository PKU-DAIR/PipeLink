#include "evaluator/harness/parity/paired_reload.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "normal";
    int scored = 0, warmup = 0, reloads = 0;
    bool reversed = false;
    auto time_batch = [&](bool candidate, int count) {
        if (count == 8) warmup += count;
        else scored += count;
        const double base = (mode == "short" ? 0.0001 : 2.0) * (candidate ? 2.0 : 1.0);
        if (mode == "invalid" && reversed && count == 10) return 0.0;
        return count * base * ((candidate != reversed) ? 4.0 : 1.0);
    };
    try {
        const auto pair = pipelink::parity::time_pair(time_batch, [&] {
            ++reloads;
            if (mode == "reload-error") throw std::runtime_error("reload failure");
            reversed = true;
        });
        std::cout << "{\"pair\":";
        pipelink::parity::write_json(std::cout, pair);
        std::cout << ",\"scored\":" << scored << ",\"warmup\":" << warmup
                  << ",\"reloads\":" << reloads << "}\n";
        return 0;
    } catch (const std::exception&) {
        std::cout << "{\"scored\":" << scored << ",\"warmup\":" << warmup
                  << ",\"reloads\":" << reloads << "}\n";
        return 3;
    }
}
