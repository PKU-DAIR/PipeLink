#include "consumer.cuh"

#include <cuda.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void check(CUresult result, char const* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  char const* name = nullptr;
  char const* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  throw std::runtime_error(
      std::string(operation) + ": " + (name == nullptr ? "unknown" : name) +
      " (" + (description == nullptr ? "no description" : description) + ")");
}

unsigned parse_unsigned(char const* text, char const* name) {
  char* end = nullptr;
  unsigned long const value = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0' || value > 0xfffffffful) {
    throw std::runtime_error(std::string("invalid unsigned ") + name);
  }
  return static_cast<unsigned>(value);
}

void wait_with_watchdog(CUevent event) {
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (true) {
    CUresult const result = cuEventQuery(event);
    if (result == CUDA_SUCCESS) {
      return;
    }
    if (result != CUDA_ERROR_NOT_READY) {
      check(result, "cuEventQuery");
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      std::fprintf(stderr, "watchdog timeout: consumer kernel exceeded 5 seconds\n");
      std::fflush(stderr);
      std::_Exit(124);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

uint64_t output_digest(std::vector<unsigned> const& output) {
  uint64_t hash = 1469598103934665603ull;
  for (unsigned value : output) {
    hash ^= value;
    hash *= 1099511628211ull;
  }
  return hash;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 10) {
    std::fprintf(
        stderr,
        "usage: %s CUBIN KERNEL TOTAL_BYTES INTERPRETER_OFFSET "
        "INSTRUCTION1_OFFSET INSTRUCTION2_OFFSET DYNAMIC_SMEM OUTPUT INPUT\n",
        argv[0]);
    return 2;
  }

  try {
    using namespace pipelink_profile_consumer;
    unsigned const total_bytes = parse_unsigned(argv[3], "total bytes");
    unsigned const interpreter_offset = parse_unsigned(argv[4], "interpreter offset");
    unsigned const instruction1_offset = parse_unsigned(argv[5], "instruction1 offset");
    unsigned const instruction2_offset = parse_unsigned(argv[6], "instruction2 offset");
    unsigned const dynamic_smem = parse_unsigned(argv[7], "dynamic smem");
    if (interpreter_offset + sizeof(InterpreterParams) > total_bytes ||
        instruction1_offset + sizeof(InstructionParams) > total_bytes ||
        instruction2_offset + sizeof(InstructionParams) > total_bytes) {
      throw std::runtime_error("params layout offsets exceed aggregate size");
    }

    check(cuInit(0), "cuInit");
    CUdevice device;
    check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context;
    CUctxCreateParams context_params{};
    check(cuCtxCreate(&context, &context_params, 0, device), "cuCtxCreate");

    CUmodule module;
    check(cuModuleLoad(&module, argv[1]), "cuModuleLoad");
    CUfunction kernel;
    check(cuModuleGetFunction(&kernel, module, argv[2]), "cuModuleGetFunction");
    check(cuFuncSetAttribute(
              kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
              static_cast<int>(dynamic_smem)),
          "cuFuncSetAttribute");

    CUdeviceptr descriptor_device;
    CUdeviceptr input_device;
    CUdeviceptr intermediate_device;
    CUdeviceptr output_device;
    CUdeviceptr counters_device;
    CUdeviceptr errors_device;
    check(cuMemAlloc(&descriptor_device, sizeof(InstructionDescriptor)),
          "cuMemAlloc(descriptor)");
    check(cuMemAlloc(&input_device, kElements * sizeof(unsigned)),
          "cuMemAlloc(input)");
    check(cuMemAlloc(&intermediate_device, kElements * sizeof(unsigned)),
          "cuMemAlloc(intermediate)");
    check(cuMemAlloc(&output_device, kElements * sizeof(unsigned)),
          "cuMemAlloc(output)");
    check(cuMemAlloc(&counters_device, kCounterCapacity * sizeof(unsigned)),
          "cuMemAlloc(counters)");
    check(cuMemAlloc(&errors_device, sizeof(unsigned)), "cuMemAlloc(errors)");

    InstructionDescriptor const descriptor{
        kElements, 3u, 17u, 0x9e3779b9u, 23u, kPageBytes,
        kStage1Opcode, kStage2Opcode};
    std::vector<unsigned> input(kElements);
    std::vector<unsigned> expected(kElements);
    for (unsigned index = 0; index < kElements; ++index) {
      input[index] = input_value(index);
      expected[index] = stage2_value(stage1_value(input[index], descriptor), descriptor);
    }
    check(cuMemcpyHtoD(descriptor_device, &descriptor, sizeof(descriptor)),
          "cuMemcpyHtoD(descriptor)");
    check(cuMemcpyHtoD(input_device, input.data(), input.size() * sizeof(unsigned)),
          "cuMemcpyHtoD(input)");
    check(cuMemsetD32(intermediate_device, 0xdeadbeefu, kElements),
          "cuMemsetD32(intermediate)");
    check(cuMemsetD32(output_device, 0xdeadbeefu, kElements),
          "cuMemsetD32(output)");
    check(cuMemsetD32(counters_device, 0, kCounterCapacity),
          "cuMemsetD32(counters)");
    check(cuMemsetD32(errors_device, 0, 1), "cuMemsetD32(errors)");

    InterpreterParams const interpreter{
        reinterpret_cast<InstructionDescriptor const*>(descriptor_device),
        reinterpret_cast<unsigned const*>(input_device),
        reinterpret_cast<unsigned*>(intermediate_device),
        reinterpret_cast<unsigned*>(output_device),
        reinterpret_cast<unsigned*>(counters_device),
        reinterpret_cast<unsigned*>(errors_device)};
    InstructionParams const instruction1{kStage1Opcode, {0, 0, 0}};
    InstructionParams const instruction2{kStage2Opcode, {0, 0, 0}};
    std::vector<unsigned char> packed(total_bytes, 0);
    std::memcpy(packed.data() + interpreter_offset, &interpreter, sizeof(interpreter));
    std::memcpy(packed.data() + instruction1_offset, &instruction1, sizeof(instruction1));
    std::memcpy(packed.data() + instruction2_offset, &instruction2, sizeof(instruction2));

    CUstream stream;
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    CUevent done;
    check(cuEventCreate(&done, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
    void* arguments[] = {packed.data()};
    check(cuLaunchKernel(
              kernel, kBlocks, 1, 1, kThreads, 1, 1, dynamic_smem, stream,
              arguments, nullptr),
          "cuLaunchKernel");
    check(cuEventRecord(done, stream), "cuEventRecord");
    wait_with_watchdog(done);

    std::vector<unsigned> output(kElements);
    unsigned counter = 0;
    unsigned errors = 0;
    check(cuMemcpyDtoH(output.data(), output_device, output.size() * sizeof(unsigned)),
          "cuMemcpyDtoH(output)");
    check(cuMemcpyDtoH(
              &counter, counters_device + kCounterSlot * sizeof(unsigned),
              sizeof(counter)),
          "cuMemcpyDtoH(counter)");
    check(cuMemcpyDtoH(&errors, errors_device, sizeof(errors)),
          "cuMemcpyDtoH(errors)");

    unsigned mismatches = 0;
    for (unsigned index = 0; index < kElements; ++index) {
      mismatches += output[index] == expected[index] ? 0u : 1u;
    }
    std::ofstream output_file(argv[8], std::ios::binary | std::ios::trunc);
    output_file.write(
        reinterpret_cast<char const*>(output.data()),
        static_cast<std::streamsize>(output.size() * sizeof(unsigned)));
    if (!output_file) {
      throw std::runtime_error("cannot write output file");
    }
    std::ofstream input_file(argv[9], std::ios::binary | std::ios::trunc);
    input_file.write(
        reinterpret_cast<char const*>(input.data()),
        static_cast<std::streamsize>(input.size() * sizeof(unsigned)));
    if (!input_file) {
      throw std::runtime_error("cannot write input file");
    }


    bool const passed = counter == 1 && errors == 0 && mismatches == 0;
    std::printf(
        "count=%u counter=%u errors=%u mismatches=%u digest=%016llx status=%s\n",
        kElements, counter, errors, mismatches,
        static_cast<unsigned long long>(output_digest(output)),
        passed ? "PASS" : "FAIL");

    check(cuEventDestroy(done), "cuEventDestroy");
    check(cuStreamDestroy(stream), "cuStreamDestroy");
    check(cuMemFree(errors_device), "cuMemFree(errors)");
    check(cuMemFree(counters_device), "cuMemFree(counters)");
    check(cuMemFree(output_device), "cuMemFree(output)");
    check(cuMemFree(intermediate_device), "cuMemFree(intermediate)");
    check(cuMemFree(input_device), "cuMemFree(input)");
    check(cuMemFree(descriptor_device), "cuMemFree(descriptor)");
    check(cuModuleUnload(module), "cuModuleUnload");
    check(cuCtxDestroy(context), "cuCtxDestroy");
    return passed ? 0 : 1;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "runner error: %s\n", error.what());
    return 1;
  }
}
