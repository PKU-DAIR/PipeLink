#pragma once
#include <chrono>
#include <iostream>
#include <iterator>
#include "json.cuh"

namespace pipelink::attention::slot {
using Clock = std::chrono::system_clock;
struct Request {
  std::string action;
  std::string case_id;
  std::string image_id;
  std::string image_role;
  std::string protocol_version;
  int warmups = -1;
  double slot_seconds = -1.0;
  double discard_seconds = -1.0;
  double steady_seconds = -1.0;
  double batch_quantum_ms = -1.0;
};

Request read_request() {
  Request r;
  std::string raw{std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>()};
  Json request = JsonParser(raw).parse();
  if (request.at("schema_version").string() != "m3-parity-slot-request-v1") {
    throw std::runtime_error("unsupported request schema");
  }
  r.action = request.at("action").string();
  r.case_id = request.at("case_id").string();
  r.image_id = request.at("image").at("image_id").string();
  r.image_role = request.at("image_role").string();
  Json const& protocol = request.at("protocol");
  r.warmups = protocol.at("warmups_per_image").integer();
  r.slot_seconds = protocol.at("slot_seconds").number();
  r.discard_seconds = protocol.at("leading_discard_seconds").number();
  r.steady_seconds = protocol.at("steady_seconds").number();
  r.batch_quantum_ms = protocol.at("configured_batch_quantum_ms").number();
  r.protocol_version = protocol.at("protocol_variant").string();
  if (request.at("cuda_visible_device").integer() != 0) throw std::runtime_error("visible CUDA device must be 0");
  if (r.action != "warmup" && r.action != "slot" && r.action != "sanitize") {
    throw std::runtime_error("unsupported action");
  }
  if (r.warmups < 0 || r.slot_seconds <= 0.0 || r.discard_seconds < 0.0 ||
      r.steady_seconds <= 0.0 || r.batch_quantum_ms <= 0.0 ||
      std::abs(r.discard_seconds + r.steady_seconds - r.slot_seconds) > 1e-9) {
    throw std::runtime_error("protocol timing fields are required and inconsistent");
  }
  return r;
}

uint64_t epoch_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

} // namespace pipelink::attention::slot
