#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr const char* kSchemaVersion = "device-dispatch-v1";
constexpr int kDefaultBlockSize = 256;

struct alignas(16) AgentState {
    std::uint32_t pc;
    std::uint32_t budget;
    std::uint32_t risk;
    std::uint32_t route;
};

static_assert(sizeof(AgentState) == 16, "AgentState layout must be stable");

__host__ __device__ inline std::uint32_t rotate_left(std::uint32_t value, unsigned shift) {
    shift &= 31U;
    return (value << shift) | (value >> ((32U - shift) & 31U));
}

// A deterministic integer-only transition standing in for route selection,
// budget accounting, risk update, and program-counter advancement.
__host__ __device__ inline AgentState transition(AgentState state, std::uint32_t step) {
    std::uint32_t mixed = state.risk;
    mixed ^= state.pc * 0x9e3779b9U;
    mixed ^= state.route * 0x85ebca6bU;
    mixed ^= (step + 1U) * 0xc2b2ae35U;
    mixed ^= mixed >> 16U;
    mixed *= 0x7feb352dU;
    mixed ^= mixed >> 15U;

    const std::uint32_t cost = 1U + (mixed & 7U);
    const std::uint32_t next_pc = (state.pc + 1U + ((mixed >> 3U) & 7U)) % 29U;
    const std::uint32_t next_route =
        (state.route * 5U + next_pc + ((mixed >> 11U) & 15U)) & 15U;

    state.pc = next_pc;
    state.budget = state.budget > cost ? state.budget - cost : 0U;
    state.route = next_route;
    state.risk = rotate_left(mixed ^ state.budget ^ (next_route * 0x27d4eb2dU),
                             5U + (next_route & 7U));
    return state;
}

// Host-only reference written separately from transition(). Keeping the oracle
// on a distinct code path catches sequencing and implementation mistakes that
// a second call to the timed function would reproduce automatically.
AgentState reference_transition(AgentState input, std::uint32_t step) {
    std::uint32_t value = input.risk ^ (input.pc * 0x9e3779b9U);
    value = value ^ (input.route * 0x85ebca6bU);
    value = value ^ ((step + 1U) * 0xc2b2ae35U);
    value = value ^ (value >> 16U);
    value = value * 0x7feb352dU;
    value = value ^ (value >> 15U);

    const std::uint32_t charge = (value & 7U) + 1U;
    const std::uint32_t program_counter =
        (input.pc + ((value >> 3U) & 7U) + 1U) % 29U;
    const std::uint32_t route =
        ((input.route * 5U) + program_counter + ((value >> 11U) & 15U)) & 15U;
    const std::uint32_t remaining_budget =
        input.budget > charge ? input.budget - charge : 0U;
    const unsigned rotation = 5U + (route & 7U);
    const std::uint32_t risk_input =
        value ^ remaining_budget ^ (route * 0x27d4eb2dU);
    const std::uint32_t risk =
        (risk_input << rotation) | (risk_input >> (32U - rotation));
    return AgentState{program_counter, remaining_budget, risk, route};
}

__global__ void transition_kernel(AgentState* states, std::size_t count, std::uint32_t step) {
    const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        states[index] = transition(states[index], step);
    }
}

__global__ void launch_device_graph_kernel(cudaGraphExec_t child, int* launch_status) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const cudaError_t status = cudaGraphLaunch(child, cudaStreamGraphFireAndForget);
        *launch_status = static_cast<int>(status);
    }
}

struct Config {
    std::string experiment_id = "device-dispatch-pilot";
    fs::path output_dir = "data/raw";
    std::vector<std::size_t> agent_counts{32, 256, 2048, 16384};
    std::vector<std::uint32_t> step_counts{1, 8, 64};
    int warmups = 10;
    int repetitions = 50;
    std::uint64_t seed = 20260811ULL;
    int block_size = kDefaultBlockSize;
    bool smoke = false;
};

struct HardwareInfo {
    bool available = false;
    int device_count = 0;
    int device_index = 0;
    std::string discovery_error;
    std::string name;
    std::string uuid;
    int compute_major = 0;
    int compute_minor = 0;
    std::uint64_t total_global_memory = 0;
    int multiprocessors = 0;
    int clock_rate_khz = 0;
    int memory_clock_rate_khz = 0;
    int memory_bus_width_bits = 0;
    int l2_bytes = 0;
    int unified_addressing = 0;
    int cooperative_launch = 0;
    int pci_domain = 0;
    int pci_bus = 0;
    int pci_device = 0;
    int runtime_version = 0;
    int driver_version = 0;
};

struct Measurement {
    std::string status = "ok";
    std::string failure_stage;
    int error_code = 0;
    std::string error_message;
    std::uint64_t wall_ns = 0;
    std::optional<std::uint64_t> device_ns;
    std::string device_time_scope;
    std::uint64_t observed_checksum = 0;
    bool has_observed_checksum = false;
    bool exact_match = false;
};

struct Row {
    std::string timestamp_utc;
    std::string run_id;
    std::string experiment_id;
    std::string phase;
    std::string mechanism;
    std::size_t agents = 0;
    std::uint32_t steps = 0;
    int repetition = -1;
    int order_index = -1;
    std::string status;
    std::string failure_stage;
    int error_code = 0;
    std::string error_message;
    std::optional<std::uint64_t> wall_ns;
    std::optional<std::uint64_t> device_ns;
    std::string device_time_scope;
    std::uint64_t expected_checksum = 0;
    std::optional<std::uint64_t> observed_checksum;
    bool exact_match = false;
    std::uint64_t seed = 0;
    int block_size = 0;
};

std::string utc_timestamp(bool compact = false) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t raw = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&raw, &tm);
    std::ostringstream output;
    output << std::put_time(&tm, compact ? "%Y%m%dT%H%M%SZ" : "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(ch) << std::dec;
                } else {
                    output << ch;
                }
        }
    }
    return output.str();
}

std::string csv_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

std::string environment_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

template <typename T>
std::string json_array(const std::vector<T>& values) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        output << values[index];
    }
    output << ']';
    return output.str();
}

template <typename T>
std::vector<T> parse_list(const std::string& text) {
    std::vector<T> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) throw std::invalid_argument("empty item in list: " + text);
        const unsigned long long parsed = std::stoull(item);
        if (parsed == 0) throw std::invalid_argument("list values must be positive: " + text);
        values.push_back(static_cast<T>(parsed));
    }
    if (values.empty()) throw std::invalid_argument("list must not be empty");
    return values;
}

void print_help(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --experiment-id ID       Output identifier (default device-dispatch-pilot)\n"
        << "  --output-dir PATH        Append-only artifact directory (default data/raw)\n"
        << "  --agents CSV             Agent counts (default 32,256,2048,16384)\n"
        << "  --steps CSV              Sequential transition counts (default 1,8,64)\n"
        << "  --warmups N              Warmups per mechanism and cell (default 10)\n"
        << "  --repetitions N          Recorded repetitions per mechanism and cell (default 50)\n"
        << "  --seed N                 Deterministic seed (default 20260811)\n"
        << "  --block-size N           CUDA threads per block (default 256)\n"
        << "  --smoke                  Tiny 32/256 x 1/8 run with 1 warmup and 2 repetitions\n"
        << "  --help                   Print this message without touching CUDA\n";
}

Config parse_args(int argc, char** argv) {
    Config config;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&](const std::string& option) -> std::string {
            if (++index >= argc) throw std::invalid_argument("missing value for " + option);
            return argv[index];
        };
        if (argument == "--experiment-id") {
            config.experiment_id = value(argument);
        } else if (argument == "--output-dir") {
            config.output_dir = value(argument);
        } else if (argument == "--agents") {
            config.agent_counts = parse_list<std::size_t>(value(argument));
        } else if (argument == "--steps") {
            config.step_counts = parse_list<std::uint32_t>(value(argument));
        } else if (argument == "--warmups") {
            config.warmups = std::stoi(value(argument));
        } else if (argument == "--repetitions") {
            config.repetitions = std::stoi(value(argument));
        } else if (argument == "--seed") {
            config.seed = std::stoull(value(argument));
        } else if (argument == "--block-size") {
            config.block_size = std::stoi(value(argument));
        } else if (argument == "--smoke") {
            config.smoke = true;
        } else if (argument == "--help") {
            print_help(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (config.smoke) {
        config.agent_counts = {32, 256};
        config.step_counts = {1, 8};
        config.warmups = 1;
        config.repetitions = 2;
    }
    if (config.experiment_id.empty()) throw std::invalid_argument("experiment id must not be empty");
    if (config.warmups < 0) throw std::invalid_argument("warmups must be non-negative");
    if (config.repetitions <= 0) throw std::invalid_argument("repetitions must be positive");
    if (config.block_size <= 0 || config.block_size > 1024) {
        throw std::invalid_argument("block size must be in [1, 1024]");
    }
    for (const auto steps : config.step_counts) {
        if (steps > 4096U) throw std::invalid_argument("steps are bounded at 4096");
    }
    return config;
}

std::vector<AgentState> initial_states(std::size_t count, std::uint64_t seed) {
    std::vector<AgentState> states(count);
    for (std::size_t index = 0; index < count; ++index) {
        std::uint64_t value = seed + 0x9e3779b97f4a7c15ULL * (index + 1ULL);
        value ^= value >> 30U;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27U;
        value *= 0x94d049bb133111ebULL;
        value ^= value >> 31U;
        states[index] = AgentState{
            static_cast<std::uint32_t>(value % 29ULL),
            256U + static_cast<std::uint32_t>((value >> 8U) & 1023ULL),
            static_cast<std::uint32_t>(value),
            static_cast<std::uint32_t>((value >> 32U) & 15ULL),
        };
    }
    return states;
}

void cpu_apply(std::vector<AgentState>& states, std::uint32_t steps) {
    for (std::uint32_t step = 0; step < steps; ++step) {
        for (AgentState& state : states) {
            state = transition(state, step);
        }
    }
}

void reference_apply(std::vector<AgentState>& states, std::uint32_t steps) {
    for (std::uint32_t step = 0; step < steps; ++step) {
        for (AgentState& state : states) {
            state = reference_transition(state, step);
        }
    }
}

std::optional<std::string> first_state_difference(const std::vector<AgentState>& expected,
                                                  const std::vector<AgentState>& observed) {
    if (expected.size() != observed.size()) {
        return "state vector size mismatch: expected=" + std::to_string(expected.size()) +
               ", observed=" + std::to_string(observed.size());
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const AgentState& left = expected[index];
        const AgentState& right = observed[index];
        if (left.pc == right.pc && left.budget == right.budget && left.risk == right.risk &&
            left.route == right.route) {
            continue;
        }
        std::ostringstream message;
        message << "first state mismatch at agent " << index << ": expected=(" << left.pc << ','
                << left.budget << ',' << left.risk << ',' << left.route << "), observed=("
                << right.pc << ',' << right.budget << ',' << right.risk << ',' << right.route
                << ')';
        return message.str();
    }
    return std::nullopt;
}

std::uint64_t checksum(const std::vector<AgentState>& states) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const AgentState& state : states) {
        for (const std::uint32_t value : {state.pc, state.budget, state.risk, state.route}) {
            for (unsigned byte = 0; byte < 4U; ++byte) {
                hash ^= static_cast<std::uint8_t>(value >> (byte * 8U));
                hash *= 1099511628211ULL;
            }
        }
    }
    return hash;
}

Measurement cpu_measure(const std::vector<AgentState>& initial,
                        std::uint32_t steps,
                        std::uint64_t expected_checksum,
                        const std::vector<AgentState>& oracle) {
    std::vector<AgentState> working = initial;
    const auto start = Clock::now();
    cpu_apply(working, steps);
    const auto stop = Clock::now();
    Measurement result;
    result.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    result.observed_checksum = checksum(working);
    result.has_observed_checksum = true;
    const std::optional<std::string> difference = first_state_difference(oracle, working);
    result.exact_match = !difference.has_value();
    if (!result.exact_match) {
        result.status = "correctness_failure";
        result.failure_stage = "state_comparison";
        result.error_message = *difference;
    } else if (result.observed_checksum != expected_checksum) {
        result.status = "correctness_failure";
        result.failure_stage = "checksum";
        result.error_message = "equal C++ states produced unequal checksums";
        result.exact_match = false;
    }
    return result;
}

std::string cuda_message(cudaError_t status) {
    const char* name = cudaGetErrorName(status);
    const char* text = cudaGetErrorString(status);
    return std::string(name == nullptr ? "cudaErrorUnknown" : name) + ": " +
           (text == nullptr ? "unknown CUDA error" : text);
}

HardwareInfo discover_hardware() {
    HardwareInfo info;
    cudaRuntimeGetVersion(&info.runtime_version);
    cudaDriverGetVersion(&info.driver_version);
    cudaError_t status = cudaGetDeviceCount(&info.device_count);
    if (status != cudaSuccess || info.device_count == 0) {
        info.discovery_error = status == cudaSuccess ? "no CUDA devices" : cuda_message(status);
        cudaGetLastError();
        return info;
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, info.device_index);
    if (status != cudaSuccess) {
        info.discovery_error = cuda_message(status);
        cudaGetLastError();
        return info;
    }
    info.available = true;
    info.name = properties.name;
    info.total_global_memory = properties.totalGlobalMem;
    // CUDA 13 removed several formerly deprecated cudaDeviceProp fields,
    // including clockRate and memoryClockRate. Query attributes throughout so
    // the same source also remains forward-compatible with later toolkits.
    const auto attribute = [&](int& destination, cudaDeviceAttr name) {
        if (cudaDeviceGetAttribute(&destination, name, info.device_index) != cudaSuccess) {
            destination = 0;
            cudaGetLastError();
        }
    };
    attribute(info.compute_major, cudaDevAttrComputeCapabilityMajor);
    attribute(info.compute_minor, cudaDevAttrComputeCapabilityMinor);
    attribute(info.multiprocessors, cudaDevAttrMultiProcessorCount);
    attribute(info.clock_rate_khz, cudaDevAttrClockRate);
    attribute(info.memory_clock_rate_khz, cudaDevAttrMemoryClockRate);
    attribute(info.memory_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth);
    attribute(info.l2_bytes, cudaDevAttrL2CacheSize);
    attribute(info.unified_addressing, cudaDevAttrUnifiedAddressing);
    attribute(info.cooperative_launch, cudaDevAttrCooperativeLaunch);
    attribute(info.pci_domain, cudaDevAttrPciDomainId);
    attribute(info.pci_bus, cudaDevAttrPciBusId);
    attribute(info.pci_device, cudaDevAttrPciDeviceId);

    std::ostringstream uuid_output;
    uuid_output << std::hex << std::setfill('0');
    for (int index = 0; index < 16; ++index) {
        uuid_output << std::setw(2)
                    << static_cast<unsigned>(
                           static_cast<unsigned char>(properties.uuid.bytes[index]));
    }
    info.uuid = uuid_output.str();
    return info;
}

class CsvWriter {
public:
    explicit CsvWriter(const fs::path& path) : stream_(path, std::ios::out | std::ios::app) {
        if (!stream_) throw std::runtime_error("cannot open CSV: " + path.string());
        if (fs::file_size(path) == 0) {
            stream_ << "schema_version,timestamp_utc,run_id,experiment_id,phase,mechanism,"
                       "agents,steps,repetition,order_index,status,failure_stage,error_code,"
                       "error_message,wall_ns,device_ns,device_time_scope,expected_checksum,"
                       "observed_checksum,exact_match,seed,block_size\n";
            stream_.flush();
        }
    }

    void write(const Row& row) {
        stream_ << csv_escape(kSchemaVersion) << ',' << csv_escape(row.timestamp_utc) << ','
                << csv_escape(row.run_id) << ',' << csv_escape(row.experiment_id) << ','
                << csv_escape(row.phase) << ',' << csv_escape(row.mechanism) << ','
                << row.agents << ',' << row.steps << ',' << row.repetition << ','
                << row.order_index << ',' << csv_escape(row.status) << ','
                << csv_escape(row.failure_stage) << ',' << row.error_code << ','
                << csv_escape(row.error_message) << ',';
        if (row.wall_ns) stream_ << *row.wall_ns;
        stream_ << ',';
        if (row.device_ns) stream_ << *row.device_ns;
        stream_ << ',' << csv_escape(row.device_time_scope) << ',' << row.expected_checksum << ',';
        if (row.observed_checksum) stream_ << *row.observed_checksum;
        stream_ << ',' << (row.exact_match ? "true" : "false") << ',' << row.seed << ','
                << row.block_size << '\n';
        stream_.flush();
        if (!stream_) throw std::runtime_error("failed while appending CSV row");
    }

private:
    std::ofstream stream_;
};

class GraphBundle {
public:
    cudaGraph_t transition_graph = nullptr;
    cudaGraphExec_t host_exec = nullptr;
    cudaGraphExec_t device_exec = nullptr;
    cudaGraph_t parent_graph = nullptr;
    cudaGraphExec_t parent_exec = nullptr;
    bool host_ready = false;
    bool device_ready = false;
    int host_error_code = 0;
    int device_error_code = 0;
    std::string host_error;
    std::string device_error;

    GraphBundle() = default;
    GraphBundle(const GraphBundle&) = delete;
    GraphBundle& operator=(const GraphBundle&) = delete;

    ~GraphBundle() {
        if (parent_exec) cudaGraphExecDestroy(parent_exec);
        if (parent_graph) cudaGraphDestroy(parent_graph);
        if (device_exec) cudaGraphExecDestroy(device_exec);
        if (host_exec) cudaGraphExecDestroy(host_exec);
        if (transition_graph) cudaGraphDestroy(transition_graph);
    }

    void setup(AgentState* device_states,
               std::size_t count,
               std::uint32_t steps,
               int block_size,
               int* device_launch_status,
               cudaStream_t stream) {
        cudaError_t status = cudaGraphCreate(&transition_graph, 0);
        if (status != cudaSuccess) {
            host_error_code = device_error_code = static_cast<int>(status);
            host_error = device_error = cuda_message(status);
            return;
        }

        cudaGraphNode_t previous = nullptr;
        for (std::uint32_t step = 0; step < steps; ++step) {
            cudaGraphNode_t node = nullptr;
            std::size_t count_argument = count;
            std::uint32_t step_argument = step;
            AgentState* states_argument = device_states;
            void* arguments[] = {&states_argument, &count_argument, &step_argument};
            cudaKernelNodeParams parameters{};
            parameters.func = reinterpret_cast<void*>(transition_kernel);
            parameters.gridDim = dim3(static_cast<unsigned>((count + block_size - 1) / block_size));
            parameters.blockDim = dim3(static_cast<unsigned>(block_size));
            parameters.sharedMemBytes = 0;
            parameters.kernelParams = arguments;
            status = cudaGraphAddKernelNode(
                &node, transition_graph, previous ? &previous : nullptr, previous ? 1 : 0, &parameters);
            if (status != cudaSuccess) {
                host_error_code = device_error_code = static_cast<int>(status);
                host_error = device_error = cuda_message(status);
                return;
            }
            previous = node;
        }

        status = cudaGraphInstantiateWithFlags(&host_exec, transition_graph, 0);
        if (status == cudaSuccess) {
            host_ready = true;
        } else {
            host_error_code = static_cast<int>(status);
            host_error = cuda_message(status);
            cudaGetLastError();
        }

        status = cudaGraphInstantiateWithFlags(
            &device_exec, transition_graph, cudaGraphInstantiateFlagDeviceLaunch);
        if (status != cudaSuccess) {
            device_error_code = static_cast<int>(status);
            device_error = cuda_message(status);
            cudaGetLastError();
            return;
        }
        status = cudaGraphUpload(device_exec, stream);
        if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) {
            device_error_code = static_cast<int>(status);
            device_error = cuda_message(status);
            cudaGetLastError();
            return;
        }

        status = cudaGraphCreate(&parent_graph, 0);
        if (status != cudaSuccess) {
            device_error_code = static_cast<int>(status);
            device_error = cuda_message(status);
            return;
        }
        cudaGraphNode_t launcher_node = nullptr;
        cudaGraphExec_t child_argument = device_exec;
        int* status_argument = device_launch_status;
        void* launcher_arguments[] = {&child_argument, &status_argument};
        cudaKernelNodeParams launcher_parameters{};
        launcher_parameters.func = reinterpret_cast<void*>(launch_device_graph_kernel);
        launcher_parameters.gridDim = dim3(1);
        launcher_parameters.blockDim = dim3(1);
        launcher_parameters.kernelParams = launcher_arguments;
        status = cudaGraphAddKernelNode(
            &launcher_node, parent_graph, nullptr, 0, &launcher_parameters);
        if (status == cudaSuccess) {
            status = cudaGraphInstantiateWithFlags(&parent_exec, parent_graph, 0);
        }
        if (status != cudaSuccess) {
            device_error_code = static_cast<int>(status);
            device_error = cuda_message(status);
            cudaGetLastError();
            return;
        }
        device_ready = true;
    }
};

Measurement gpu_measure(const std::string& mechanism,
                        AgentState* device_initial,
                        AgentState* device_working,
                        int* device_launch_status,
                        std::size_t count,
                        std::uint32_t steps,
                        int block_size,
                        cudaStream_t stream,
                        cudaEvent_t event_start,
                        cudaEvent_t event_stop,
                        cudaGraphExec_t graph_exec,
                        std::uint64_t expected_checksum,
                        const std::vector<AgentState>& oracle,
                        std::vector<AgentState>& host_output) {
    Measurement result;
    const std::size_t bytes = count * sizeof(AgentState);
    cudaError_t status = cudaMemcpyAsync(
        device_working, device_initial, bytes, cudaMemcpyDeviceToDevice, stream);
    if (mechanism == "cuda_device_graph" && status == cudaSuccess) {
        status = cudaMemsetAsync(device_launch_status, 0xff, sizeof(int), stream);
    }
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        result.status = "runtime_failure";
        result.failure_stage = "state_reset";
        result.error_code = static_cast<int>(status);
        result.error_message = cuda_message(status);
        cudaGetLastError();
        return result;
    }

    status = cudaEventRecord(event_start, stream);
    const auto wall_start = Clock::now();
    if (status == cudaSuccess && mechanism == "cuda_host_launch") {
        const unsigned grid = static_cast<unsigned>((count + block_size - 1) / block_size);
        for (std::uint32_t step = 0; step < steps; ++step) {
            transition_kernel<<<grid, block_size, 0, stream>>>(device_working, count, step);
        }
        status = cudaPeekAtLastError();
    } else if (status == cudaSuccess) {
        status = cudaGraphLaunch(graph_exec, stream);
    }
    if (status == cudaSuccess) status = cudaEventRecord(event_stop, stream);
    if (status == cudaSuccess) status = cudaEventSynchronize(event_stop);
    const auto wall_stop = Clock::now();
    if (status != cudaSuccess) {
        result.status = "runtime_failure";
        result.failure_stage = "dispatch_or_synchronize";
        result.error_code = static_cast<int>(status);
        result.error_message = cuda_message(status);
        cudaGetLastError();
        return result;
    }

    float elapsed_ms = 0.0F;
    status = cudaEventElapsedTime(&elapsed_ms, event_start, event_stop);
    if (status != cudaSuccess) {
        result.status = "runtime_failure";
        result.failure_stage = "event_timing";
        result.error_code = static_cast<int>(status);
        result.error_message = cuda_message(status);
        cudaGetLastError();
        return result;
    }
    result.wall_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(wall_stop - wall_start).count();
    result.device_ns = static_cast<std::uint64_t>(elapsed_ms * 1'000'000.0F);
    result.device_time_scope = mechanism == "cuda_device_graph"
                                   ? "complete_parent_and_child_execution_environment"
                                   : "transition_kernels";

    if (mechanism == "cuda_device_graph") {
        int launch_status = -1;
        status = cudaMemcpy(
            &launch_status, device_launch_status, sizeof(int), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess || launch_status != static_cast<int>(cudaSuccess)) {
            result.status = "runtime_failure";
            result.failure_stage = "device_graph_launch";
            result.error_code = status != cudaSuccess ? static_cast<int>(status) : launch_status;
            result.error_message = status != cudaSuccess
                                       ? cuda_message(status)
                                       : "device-side cudaGraphLaunch returned code " +
                                             std::to_string(launch_status);
            cudaGetLastError();
            return result;
        }
    }

    status = cudaMemcpy(host_output.data(), device_working, bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        result.status = "runtime_failure";
        result.failure_stage = "result_copy";
        result.error_code = static_cast<int>(status);
        result.error_message = cuda_message(status);
        cudaGetLastError();
        return result;
    }
    result.observed_checksum = checksum(host_output);
    result.has_observed_checksum = true;
    const std::optional<std::string> difference = first_state_difference(oracle, host_output);
    result.exact_match = !difference.has_value();
    if (!result.exact_match) {
        result.status = "correctness_failure";
        result.failure_stage = "state_comparison";
        result.error_message = *difference;
    } else if (result.observed_checksum != expected_checksum) {
        result.status = "correctness_failure";
        result.failure_stage = "checksum";
        result.error_message = "equal GPU states produced unequal checksums";
        result.exact_match = false;
    }
    return result;
}

Row make_row(const Config& config,
             const std::string& run_id,
             const std::string& phase,
             const std::string& mechanism,
             std::size_t agents,
             std::uint32_t steps,
             int repetition,
             int order_index,
             std::uint64_t expected_checksum,
             const Measurement& measurement) {
    Row row;
    row.timestamp_utc = utc_timestamp();
    row.run_id = run_id;
    row.experiment_id = config.experiment_id;
    row.phase = phase;
    row.mechanism = mechanism;
    row.agents = agents;
    row.steps = steps;
    row.repetition = repetition;
    row.order_index = order_index;
    row.status = measurement.status;
    row.failure_stage = measurement.failure_stage;
    row.error_code = measurement.error_code;
    row.error_message = measurement.error_message;
    if (measurement.status != "setup_failure" && measurement.status != "unsupported") {
        row.wall_ns = measurement.wall_ns;
    }
    row.device_ns = measurement.device_ns;
    row.device_time_scope = measurement.device_time_scope;
    row.expected_checksum = expected_checksum;
    if (measurement.has_observed_checksum) {
        row.observed_checksum = measurement.observed_checksum;
    }
    row.exact_match = measurement.exact_match;
    row.seed = config.seed;
    row.block_size = config.block_size;
    return row;
}

std::string cpu_model() {
    std::ifstream input("/proc/cpuinfo");
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto separator = line.find(':');
            return separator == std::string::npos ? line : line.substr(separator + 2);
        }
    }
    return "unknown";
}

std::string os_description() {
    struct utsname details {};
    if (uname(&details) != 0) return "unknown";
    return std::string(details.sysname) + " " + details.release + " " + details.machine;
}

void write_manifest(const fs::path& path,
                    const Config& config,
                    const HardwareInfo& hardware,
                    const std::string& run_id,
                    const std::string& started_at,
                    const std::string& completed_at,
                    const fs::path& csv_path,
                    const std::map<std::string, std::uint64_t>& status_counts,
                    std::uint64_t measured_rows,
                    std::uint64_t exact_rows,
                    std::uint64_t failure_rows) {
    if (fs::exists(path)) throw std::runtime_error("refusing to overwrite manifest: " + path.string());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create manifest: " + path.string());
    output << "{\n"
           << "  \"schema_version\": \"" << kSchemaVersion << "\",\n"
           << "  \"run_id\": \"" << json_escape(run_id) << "\",\n"
           << "  \"experiment_id\": \"" << json_escape(config.experiment_id) << "\",\n"
           << "  \"started_at_utc\": \"" << started_at << "\",\n"
           << "  \"completed_at_utc\": \"" << completed_at << "\",\n"
           << "  \"csv_file\": \"" << json_escape(csv_path.filename().string()) << "\",\n"
           << "  \"execution_provider\": \""
           << json_escape(environment_or_empty("EXECUTION_PROVIDER")) << "\",\n"
           << "  \"requested_gpu\": \"" << json_escape(environment_or_empty("REQUESTED_GPU"))
           << "\",\n"
           << "  \"source_sha256\": \"" << json_escape(environment_or_empty("SOURCE_SHA256"))
           << "\",\n"
           << "  \"hardware\": {\n"
           << "    \"cuda_available\": " << (hardware.available ? "true" : "false") << ",\n"
           << "    \"cuda_discovery_error\": \"" << json_escape(hardware.discovery_error)
           << "\",\n"
           << "    \"device_count\": " << hardware.device_count << ",\n"
           << "    \"device_index\": " << hardware.device_index << ",\n"
           << "    \"device_name\": \"" << json_escape(hardware.name) << "\",\n"
           << "    \"device_uuid\": \"" << json_escape(hardware.uuid) << "\",\n"
           << "    \"compute_capability\": \"" << hardware.compute_major << '.'
           << hardware.compute_minor << "\",\n"
           << "    \"total_global_memory_bytes\": " << hardware.total_global_memory << ",\n"
           << "    \"multiprocessor_count\": " << hardware.multiprocessors << ",\n"
           << "    \"clock_rate_khz\": " << hardware.clock_rate_khz << ",\n"
           << "    \"memory_clock_rate_khz\": " << hardware.memory_clock_rate_khz << ",\n"
           << "    \"memory_bus_width_bits\": " << hardware.memory_bus_width_bits << ",\n"
           << "    \"l2_cache_bytes\": " << hardware.l2_bytes << ",\n"
           << "    \"unified_addressing\": " << hardware.unified_addressing << ",\n"
           << "    \"cooperative_launch\": " << hardware.cooperative_launch << ",\n"
           << "    \"pci_domain\": " << hardware.pci_domain << ",\n"
           << "    \"pci_bus\": " << hardware.pci_bus << ",\n"
           << "    \"pci_device\": " << hardware.pci_device << "\n"
           << "  },\n"
           << "  \"software\": {\n"
           << "    \"cuda_compile_version\": " << CUDART_VERSION << ",\n"
           << "    \"cuda_runtime_version\": " << hardware.runtime_version << ",\n"
           << "    \"cuda_driver_version\": " << hardware.driver_version << ",\n"
           << "    \"host_compiler\": \"" << json_escape(__VERSION__) << "\",\n"
           << "    \"os\": \"" << json_escape(os_description()) << "\",\n"
           << "    \"cpu_model\": \"" << json_escape(cpu_model()) << "\",\n"
           << "    \"cpu_hardware_threads\": " << std::thread::hardware_concurrency() << "\n"
           << "  },\n"
           << "  \"config\": {\n"
           << "    \"agent_counts\": " << json_array(config.agent_counts) << ",\n"
           << "    \"step_counts\": " << json_array(config.step_counts) << ",\n"
           << "    \"warmups_per_mechanism_cell\": " << config.warmups << ",\n"
           << "    \"repetitions_per_mechanism_cell\": " << config.repetitions << ",\n"
           << "    \"seed\": " << config.seed << ",\n"
           << "    \"block_size\": " << config.block_size << ",\n"
           << "    \"mechanisms\": [\"cpu_cpp\",\"cuda_host_launch\","
              "\"cuda_host_graph\",\"cuda_device_graph\"],\n"
           << "    \"mechanism_order\": \"deterministically shuffled within repetition\",\n"
           << "    \"state_reset_in_timing\": false,\n"
           << "    \"result_copy_in_timing\": false,\n"
           << "    \"checksum_in_timing\": false\n"
           << "  },\n"
           << "  \"results\": {\n"
           << "    \"measured_rows\": " << measured_rows << ",\n"
           << "    \"exact_rows\": " << exact_rows << ",\n"
           << "    \"failure_rows\": " << failure_rows << ",\n"
           << "    \"status_counts\": {";
    bool first = true;
    for (const auto& [status, count] : status_counts) {
        if (!first) output << ',';
        output << "\n      \"" << json_escape(status) << "\": " << count;
        first = false;
    }
    if (!status_counts.empty()) output << '\n';
    output << "    }\n"
           << "  },\n"
           << "  \"limitations\": [\n"
           << "    \"single placement and unlocked clocks; replication is required\",\n"
           << "    \"CPU affinity and NUMA placement are not controlled\",\n"
           << "    \"device graph topology is instantiated and uploaded by the host\",\n"
           << "    \"transition kernel has no external side effects or tool I/O\"\n"
           << "  ]\n"
           << "}\n";
    output.flush();
    if (!output) throw std::runtime_error("failed while writing manifest");
}

int run(const Config& config) {
    fs::create_directories(config.output_dir);
    const std::string started_at = utc_timestamp();
    const std::string compact = utc_timestamp(true);
    const std::string run_id = config.experiment_id + "-" + compact + "-p" +
                               std::to_string(static_cast<long long>(getpid()));
    const fs::path csv_path = config.output_dir / (run_id + ".csv");
    const fs::path manifest_path = config.output_dir / (run_id + ".manifest.json");
    if (fs::exists(csv_path) || fs::exists(manifest_path)) {
        throw std::runtime_error("refusing to overwrite existing run artifacts");
    }
    CsvWriter csv(csv_path);
    HardwareInfo hardware = discover_hardware();
    std::map<std::string, std::uint64_t> status_counts;
    std::uint64_t measured_rows = 0;
    std::uint64_t exact_rows = 0;
    std::uint64_t failure_rows = 0;

    auto persist = [&](const Row& row) {
        csv.write(row);
        ++status_counts[row.status];
        if (row.phase == "measure") ++measured_rows;
        if (row.exact_match) ++exact_rows;
        if (row.status != "ok") ++failure_rows;
    };

    cudaStream_t stream = nullptr;
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    if (hardware.available) {
        cudaSetDevice(hardware.device_index);
        cudaError_t status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (status == cudaSuccess) status = cudaEventCreate(&event_start);
        if (status == cudaSuccess) status = cudaEventCreate(&event_stop);
        if (status != cudaSuccess) {
            hardware.available = false;
            hardware.discovery_error = "CUDA timing resource setup failed: " + cuda_message(status);
            cudaGetLastError();
        }
    }

    for (const std::size_t agents : config.agent_counts) {
        const std::vector<AgentState> initial = initial_states(agents, config.seed);
        AgentState* device_initial = nullptr;
        AgentState* device_working = nullptr;
        int* device_launch_status = nullptr;
        bool buffers_ready = false;
        std::string buffer_error;
        int buffer_error_code = 0;
        if (hardware.available) {
            cudaError_t status = cudaMalloc(&device_initial, agents * sizeof(AgentState));
            if (status == cudaSuccess) {
                status = cudaMalloc(&device_working, agents * sizeof(AgentState));
            }
            if (status == cudaSuccess) status = cudaMalloc(&device_launch_status, sizeof(int));
            if (status == cudaSuccess) {
                status = cudaMemcpy(device_initial,
                                    initial.data(),
                                    agents * sizeof(AgentState),
                                    cudaMemcpyHostToDevice);
            }
            buffers_ready = status == cudaSuccess;
            if (!buffers_ready) {
                buffer_error_code = static_cast<int>(status);
                buffer_error = cuda_message(status);
                cudaGetLastError();
            }
        }
        std::vector<AgentState> host_output(agents);

        for (const std::uint32_t steps : config.step_counts) {
            std::vector<AgentState> oracle = initial;
            reference_apply(oracle, steps);
            const std::uint64_t expected_checksum = checksum(oracle);
            GraphBundle graphs;
            if (buffers_ready) {
                graphs.setup(device_working,
                             agents,
                             steps,
                             config.block_size,
                             device_launch_status,
                             stream);
            }

            std::map<std::string, bool> available{
                {"cpu_cpp", true},
                {"cuda_host_launch", buffers_ready},
                {"cuda_host_graph", buffers_ready && graphs.host_ready},
                {"cuda_device_graph", buffers_ready && graphs.device_ready},
            };
            for (const std::string mechanism :
                 {"cuda_host_launch", "cuda_host_graph", "cuda_device_graph"}) {
                if (available[mechanism]) continue;
                Measurement failure;
                failure.status = hardware.available ? "setup_failure" : "unsupported";
                failure.failure_stage = "mechanism_setup";
                if (!hardware.available) {
                    failure.error_message = hardware.discovery_error;
                } else if (!buffers_ready) {
                    failure.error_code = buffer_error_code;
                    failure.error_message = buffer_error;
                } else if (mechanism == "cuda_host_graph") {
                    failure.error_code = graphs.host_error_code;
                    failure.error_message = graphs.host_error;
                } else if (mechanism == "cuda_device_graph") {
                    failure.error_code = graphs.device_error_code;
                    failure.error_message = graphs.device_error.empty()
                                                ? "device graph unavailable on this platform"
                                                : graphs.device_error;
                }
                persist(make_row(config,
                                 run_id,
                                 "setup",
                                 mechanism,
                                 agents,
                                 steps,
                                 -1,
                                 -1,
                                 expected_checksum,
                                 failure));
            }

            auto measure = [&](const std::string& mechanism) {
                if (mechanism == "cpu_cpp") {
                    return cpu_measure(initial, steps, expected_checksum, oracle);
                }
                cudaGraphExec_t graph = nullptr;
                if (mechanism == "cuda_host_graph") graph = graphs.host_exec;
                if (mechanism == "cuda_device_graph") graph = graphs.parent_exec;
                return gpu_measure(mechanism,
                                   device_initial,
                                   device_working,
                                   device_launch_status,
                                   agents,
                                   steps,
                                   config.block_size,
                                   stream,
                                   event_start,
                                   event_stop,
                                   graph,
                                   expected_checksum,
                                   oracle,
                                   host_output);
            };

            for (int warmup = 0; warmup < config.warmups; ++warmup) {
                for (const std::string mechanism :
                     {"cpu_cpp", "cuda_host_launch", "cuda_host_graph", "cuda_device_graph"}) {
                    if (!available[mechanism]) continue;
                    const Measurement result = measure(mechanism);
                    if (result.status != "ok") {
                        available[mechanism] = false;
                        persist(make_row(config,
                                         run_id,
                                         "warmup",
                                         mechanism,
                                         agents,
                                         steps,
                                         -1,
                                         warmup,
                                         expected_checksum,
                                         result));
                    }
                }
            }

            std::vector<std::string> mechanisms;
            for (const auto& [mechanism, is_available] : available) {
                if (is_available) mechanisms.push_back(mechanism);
            }
            std::mt19937_64 generator(config.seed ^ (agents * 0x9e3779b97f4a7c15ULL) ^
                                      (static_cast<std::uint64_t>(steps) << 32U));
            for (int repetition = 0; repetition < config.repetitions; ++repetition) {
                std::shuffle(mechanisms.begin(), mechanisms.end(), generator);
                for (std::size_t order = 0; order < mechanisms.size(); ++order) {
                    const std::string& mechanism = mechanisms[order];
                    const Measurement result = measure(mechanism);
                    persist(make_row(config,
                                     run_id,
                                     "measure",
                                     mechanism,
                                     agents,
                                     steps,
                                     repetition,
                                     static_cast<int>(order),
                                     expected_checksum,
                                     result));
                }
            }
        }

        if (device_launch_status) cudaFree(device_launch_status);
        if (device_working) cudaFree(device_working);
        if (device_initial) cudaFree(device_initial);
    }

    if (event_stop) cudaEventDestroy(event_stop);
    if (event_start) cudaEventDestroy(event_start);
    if (stream) cudaStreamDestroy(stream);
    const std::string completed_at = utc_timestamp();
    write_manifest(manifest_path,
                   config,
                   hardware,
                   run_id,
                   started_at,
                   completed_at,
                   csv_path,
                   status_counts,
                   measured_rows,
                   exact_rows,
                   failure_rows);
    std::cout << "run_id=" << run_id << '\n'
              << "csv_path=" << fs::absolute(csv_path).string() << '\n'
              << "manifest_path=" << fs::absolute(manifest_path).string() << '\n'
              << "measured_rows=" << measured_rows << '\n'
              << "failure_rows=" << failure_rows << '\n';
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "fatal_error=" << error.what() << '\n';
        return 2;
    }
}
