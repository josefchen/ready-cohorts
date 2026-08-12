#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

#ifndef RESIDENT_POLICY_SOURCE_SHA256
#define RESIDENT_POLICY_SOURCE_SHA256 ""
#endif

#ifndef RESIDENT_POLICY_GIT_COMMIT
#define RESIDENT_POLICY_GIT_COMMIT "unknown"
#endif

#ifndef RESIDENT_POLICY_GIT_DIRTY
#define RESIDENT_POLICY_GIT_DIRTY "unknown"
#endif

constexpr const char* kSchemaVersion = "resident-policy-v1";
constexpr int kDefaultBlockSize = 256;
constexpr std::array<const char*, 3> kMechanisms{
    "host_roundtrip", "device_resident", "no_decision_lower_bound"};

struct alignas(16) AgentState {
    std::uint32_t token;
    std::uint32_t budget;
    std::uint32_t score;
    std::uint32_t route;
};

static_assert(sizeof(AgentState) == 16, "AgentState layout must remain stable");

// The timed CUDA implementation and the host oracle intentionally do not share
// transition or predicate functions. This side contains device-only semantics.
__device__ __forceinline__ std::uint32_t device_rotl(std::uint32_t value, unsigned shift) {
    shift &= 31U;
    return (value << shift) | (value >> ((32U - shift) & 31U));
}

__device__ __forceinline__ std::uint64_t device_mix64(std::uint64_t value) {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

__device__ __forceinline__ AgentState device_route_zero(AgentState state,
                                                         std::uint32_t epoch) {
    const std::uint32_t token =
        state.token * 1664525U + 1013904223U + epoch * 747796405U;
    const std::uint32_t charge = 1U + ((token ^ state.score) & 15U);
    const std::uint32_t budget = state.budget > charge ? state.budget - charge : 0U;
    const std::uint32_t route = (state.route + 3U + (token >> 29U)) & 15U;
    const std::uint32_t score =
        device_rotl(state.score ^ token ^ (budget * 0x9e3779b9U), 5U + (epoch & 7U));
    return AgentState{token, budget, score, route};
}

__device__ __forceinline__ AgentState device_route_one(AgentState state,
                                                        std::uint32_t epoch) {
    std::uint32_t score = state.score ^ (state.token * 0x85ebca6bU);
    score ^= (epoch + 1U) * 0xc2b2ae35U;
    score ^= score >> 16U;
    score *= 0x7feb352dU;
    score ^= score >> 15U;
    const std::uint32_t charge = 2U + ((score >> 3U) & 15U);
    const std::uint32_t budget = state.budget > charge ? state.budget - charge : 0U;
    const std::uint32_t token = state.token + 0x6d2b79f5U + device_rotl(score, 11U);
    const std::uint32_t route = (state.route * 5U + (score >> 27U) + 1U) & 15U;
    score = device_rotl(score ^ token ^ budget, 9U + (route & 7U));
    return AgentState{token, budget, score, route};
}

__device__ __forceinline__ std::uint64_t device_predicate_contribution(
    const AgentState& state, std::size_t index, std::uint32_t epoch) {
    const std::uint64_t left =
        (static_cast<std::uint64_t>(state.token) << 32U) | state.score;
    const std::uint64_t right =
        (static_cast<std::uint64_t>(state.budget) << 32U) | state.route;
    const std::uint64_t salt = 0x9e3779b97f4a7c15ULL * (index + 1ULL) ^
                               0xd1b54a32d192ed03ULL * (epoch + 1ULL);
    return device_mix64(left ^ device_mix64(right + salt));
}

__global__ void route_kernel(AgentState* states,
                             std::size_t count,
                             std::uint32_t epoch,
                             std::uint32_t branch) {
    const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        states[index] = branch == 0U ? device_route_zero(states[index], epoch)
                                    : device_route_one(states[index], epoch);
    }
}

__global__ void predicate_partial_kernel(const AgentState* states,
                                         std::size_t count,
                                         std::uint32_t epoch,
                                         std::uint64_t* partials) {
    extern __shared__ std::uint64_t scratch[];
    const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    std::uint64_t value = 0ULL;
    if (index < count) {
        value = device_predicate_contribution(states[index], index, epoch);
    }
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) scratch[threadIdx.x] ^= scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0U) partials[blockIdx.x] = scratch[0];
}

__global__ void predicate_finalize_kernel(const std::uint64_t* partials,
                                          std::size_t partial_count,
                                          std::uint32_t epoch,
                                          int* predicate) {
    extern __shared__ std::uint64_t scratch[];
    std::uint64_t value = 0ULL;
    for (std::size_t index = threadIdx.x; index < partial_count; index += blockDim.x) {
        value ^= partials[index];
    }
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) scratch[threadIdx.x] ^= scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        const std::uint64_t final_value =
            scratch[0] ^ (0xa24baed4963ee407ULL * (epoch + 1ULL));
        *predicate = static_cast<int>((final_value ^ (final_value >> 32U)) & 1ULL);
    }
}

__global__ void select_and_tail_launch_kernel(const int* predicate,
                                              cudaGraphExec_t branch_zero,
                                              cudaGraphExec_t branch_one,
                                              int* launch_statuses,
                                              int* decisions,
                                              std::uint32_t epoch) {
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
        const int decision = *predicate & 1;
        decisions[epoch] = decision;
        const cudaGraphExec_t selected = decision == 0 ? branch_zero : branch_one;
        launch_statuses[epoch] =
            static_cast<int>(cudaGraphLaunch(selected, cudaStreamGraphTailLaunch));
    }
}

struct Config {
    std::string experiment_id = "resident-policy-pilot";
    fs::path output_dir = "data/raw";
    std::vector<std::size_t> agent_counts{256, 2048, 16384};
    std::vector<std::uint32_t> epoch_counts{2, 8, 32};
    int warmups = 5;
    int calibration_samples = 3;
    int repetitions = 30;
    std::uint64_t min_duration_ns = 100'000'000ULL;
    std::uint64_t max_batch_iterations = 20'000ULL;
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
    int unified_addressing = 0;
    int pci_domain = 0;
    int pci_bus = 0;
    int pci_device = 0;
    int runtime_version = 0;
    int driver_version = 0;
};

struct OracleOutcome {
    std::vector<AgentState> states;
    std::vector<int> decisions;
    std::uint64_t state_checksum = 0;
    std::uint64_t decision_hash = 0;
};

struct Invocation {
    std::string status = "ok";
    std::string failure_stage;
    int error_code = 0;
    std::string error_message;
    std::uint64_t wall_ns = 0;
    std::optional<std::uint64_t> device_ns;
    std::uint64_t observed_state_checksum = 0;
    std::uint64_t observed_decision_hash = 0;
    std::string observed_decisions;
    bool exact_state_match = false;
    bool exact_decision_match = false;
};

struct BatchMeasurement {
    std::string status = "ok";
    std::string failure_stage;
    int error_code = 0;
    std::string error_message;
    std::uint64_t batch_iterations = 0;
    std::uint64_t aggregate_wall_ns = 0;
    std::optional<std::uint64_t> aggregate_device_ns;
    std::uint64_t observed_state_checksum = 0;
    std::uint64_t observed_decision_hash = 0;
    std::string observed_decisions;
    bool exact_state_match = false;
    bool exact_decision_match = false;
    std::uint64_t exact_validation_count = 0;
};

struct Row {
    std::string timestamp_utc;
    std::string run_id;
    std::string experiment_id;
    std::string phase;
    std::string mechanism;
    std::size_t agents = 0;
    std::uint32_t epochs = 0;
    int repetition = -1;
    int order_index = -1;
    std::string status;
    std::string failure_stage;
    int error_code = 0;
    std::string error_message;
    std::uint64_t batch_iterations = 0;
    std::optional<std::uint64_t> aggregate_wall_ns;
    std::optional<double> wall_ns_per_invocation;
    std::optional<std::uint64_t> aggregate_device_ns;
    std::optional<double> device_ns_per_invocation;
    std::uint64_t min_duration_target_ns = 0;
    bool min_duration_reached = false;
    std::uint64_t expected_state_checksum = 0;
    std::optional<std::uint64_t> observed_state_checksum;
    std::uint64_t expected_decision_hash = 0;
    std::optional<std::uint64_t> observed_decision_hash;
    std::string expected_decisions;
    std::string observed_decisions;
    bool exact_state_match = false;
    bool exact_decision_match = false;
    std::uint64_t exact_validation_count = 0;
    std::uint64_t seed = 0;
    int block_size = 0;
    std::size_t predicate_blocks = 0;
};

struct CellAudit {
    std::size_t agents = 0;
    std::uint32_t epochs = 0;
    std::uint64_t common_batch_iterations = 0;
    bool batch_cap_reached = false;
    std::map<std::string, std::uint64_t> median_calibration_wall_ns;
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
        if (ch == '"') escaped.push_back('"');
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
        if (parsed == 0ULL) throw std::invalid_argument("list values must be positive");
        values.push_back(static_cast<T>(parsed));
    }
    if (values.empty()) throw std::invalid_argument("list must not be empty");
    return values;
}

void print_help(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --experiment-id ID       Output identifier (default resident-policy-pilot)\n"
        << "  --output-dir PATH        New artifact directory (default data/raw)\n"
        << "  --agents CSV             Agent counts (default 256,2048,16384)\n"
        << "  --epochs CSV             Decision epochs (default 2,8,32)\n"
        << "  --warmups N              Untimed invocations per mechanism/cell (default 5)\n"
        << "  --calibration-samples N  Single-invocation timings (default 3)\n"
        << "  --repetitions N          Measured batches per mechanism/cell (default 30)\n"
        << "  --min-duration-ms N      Target aggregate time per row (default 100)\n"
        << "  --max-batch N            Safety cap on invocations per row (default 20000)\n"
        << "  --seed N                 Deterministic seed (default 20260811)\n"
        << "  --block-size N           Power-of-two CUDA block size (default 256)\n"
        << "  --smoke                  Tiny bounded GPU smoke test\n"
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
        } else if (argument == "--epochs") {
            config.epoch_counts = parse_list<std::uint32_t>(value(argument));
        } else if (argument == "--warmups") {
            config.warmups = std::stoi(value(argument));
        } else if (argument == "--calibration-samples") {
            config.calibration_samples = std::stoi(value(argument));
        } else if (argument == "--repetitions") {
            config.repetitions = std::stoi(value(argument));
        } else if (argument == "--min-duration-ms") {
            config.min_duration_ns = std::stoull(value(argument)) * 1'000'000ULL;
        } else if (argument == "--max-batch") {
            config.max_batch_iterations = std::stoull(value(argument));
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
        config.agent_counts = {64, 256};
        config.epoch_counts = {2, 4};
        config.warmups = 1;
        config.calibration_samples = 2;
        config.repetitions = 2;
        config.min_duration_ns = 2'000'000ULL;
        config.max_batch_iterations = 1000ULL;
    }
    if (config.experiment_id.empty()) throw std::invalid_argument("experiment id is empty");
    if (config.warmups < 0 || config.calibration_samples <= 0 || config.repetitions <= 0) {
        throw std::invalid_argument("warmups >= 0; calibration samples and repetitions > 0");
    }
    if (config.min_duration_ns == 0ULL || config.max_batch_iterations == 0ULL) {
        throw std::invalid_argument("duration and max batch must be positive");
    }
    if (config.block_size <= 0 || config.block_size > 1024 ||
        (config.block_size & (config.block_size - 1)) != 0) {
        throw std::invalid_argument("block size must be a power of two in [1, 1024]");
    }
    for (const std::size_t count : config.agent_counts) {
        const std::size_t blocks =
            (count + static_cast<std::size_t>(config.block_size) - 1ULL) /
            static_cast<std::size_t>(config.block_size);
        if (blocks > 1024ULL) {
            throw std::invalid_argument("predicate block count exceeds 1024; reduce agents");
        }
    }
    for (const std::uint32_t epochs : config.epoch_counts) {
        if (epochs > 128U) throw std::invalid_argument("epochs are bounded at 128");
    }
    return config;
}

// Independent host-only oracle implementation.
std::uint32_t oracle_rotl(std::uint32_t value, unsigned shift) {
    shift &= 31U;
    return (value << shift) | (value >> ((32U - shift) & 31U));
}

std::uint64_t oracle_mix64(std::uint64_t value) {
    value = value ^ (value >> 30U);
    value = value * 0xbf58476d1ce4e5b9ULL;
    value = value ^ (value >> 27U);
    value = value * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

AgentState oracle_branch_zero(const AgentState& input, std::uint32_t epoch) {
    const std::uint32_t token =
        input.token * 1664525U + 1013904223U + epoch * 747796405U;
    const std::uint32_t charge = ((token ^ input.score) & 15U) + 1U;
    const std::uint32_t remaining = input.budget > charge ? input.budget - charge : 0U;
    const std::uint32_t route = (input.route + 3U + (token >> 29U)) & 15U;
    const unsigned rotation = 5U + (epoch & 7U);
    const std::uint32_t score =
        oracle_rotl(input.score ^ token ^ (remaining * 0x9e3779b9U), rotation);
    return AgentState{token, remaining, score, route};
}

AgentState oracle_branch_one(const AgentState& input, std::uint32_t epoch) {
    std::uint32_t value = input.score ^ (input.token * 0x85ebca6bU);
    value = value ^ ((epoch + 1U) * 0xc2b2ae35U);
    value = value ^ (value >> 16U);
    value = value * 0x7feb352dU;
    value = value ^ (value >> 15U);
    const std::uint32_t charge = ((value >> 3U) & 15U) + 2U;
    const std::uint32_t remaining = input.budget > charge ? input.budget - charge : 0U;
    const std::uint32_t token = input.token + 0x6d2b79f5U + oracle_rotl(value, 11U);
    const std::uint32_t route = (input.route * 5U + (value >> 27U) + 1U) & 15U;
    const std::uint32_t score = oracle_rotl(value ^ token ^ remaining, 9U + (route & 7U));
    return AgentState{token, remaining, score, route};
}

int oracle_predicate(const std::vector<AgentState>& states, std::uint32_t epoch) {
    std::uint64_t aggregate = 0ULL;
    for (std::size_t index = 0; index < states.size(); ++index) {
        const AgentState& state = states[index];
        const std::uint64_t left =
            (static_cast<std::uint64_t>(state.token) << 32U) | state.score;
        const std::uint64_t right =
            (static_cast<std::uint64_t>(state.budget) << 32U) | state.route;
        const std::uint64_t salt = 0x9e3779b97f4a7c15ULL * (index + 1ULL) ^
                                   0xd1b54a32d192ed03ULL * (epoch + 1ULL);
        aggregate ^= oracle_mix64(left ^ oracle_mix64(right + salt));
    }
    const std::uint64_t final_value =
        aggregate ^ (0xa24baed4963ee407ULL * (epoch + 1ULL));
    return static_cast<int>((final_value ^ (final_value >> 32U)) & 1ULL);
}

std::vector<AgentState> make_initial_states(std::size_t count, std::uint64_t seed) {
    std::vector<AgentState> states(count);
    for (std::size_t index = 0; index < count; ++index) {
        std::uint64_t value = seed + 0x9e3779b97f4a7c15ULL * (index + 1ULL);
        value ^= value >> 30U;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27U;
        value *= 0x94d049bb133111ebULL;
        value ^= value >> 31U;
        states[index] = AgentState{
            static_cast<std::uint32_t>(value),
            512U + static_cast<std::uint32_t>((value >> 9U) & 2047ULL),
            static_cast<std::uint32_t>(value >> 32U),
            static_cast<std::uint32_t>((value >> 17U) & 15ULL),
        };
    }
    return states;
}

std::uint64_t state_checksum(const std::vector<AgentState>& states) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const AgentState& state : states) {
        for (const std::uint32_t value : {state.token, state.budget, state.score, state.route}) {
            for (unsigned byte = 0; byte < 4U; ++byte) {
                hash ^= static_cast<std::uint8_t>(value >> (byte * 8U));
                hash *= 1099511628211ULL;
            }
        }
    }
    return hash;
}

std::uint64_t decision_hash(const std::vector<int>& decisions) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const int decision : decisions) {
        hash ^= static_cast<std::uint8_t>(decision);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string decision_string(const std::vector<int>& decisions) {
    std::string result;
    result.reserve(decisions.size());
    for (const int decision : decisions) result.push_back(decision == 0 ? '0' : '1');
    return result;
}

OracleOutcome run_oracle(const std::vector<AgentState>& initial, std::uint32_t epochs) {
    OracleOutcome outcome;
    outcome.states = initial;
    outcome.decisions.reserve(epochs);
    for (std::uint32_t epoch = 0; epoch < epochs; ++epoch) {
        const int branch = oracle_predicate(outcome.states, epoch);
        outcome.decisions.push_back(branch);
        for (AgentState& state : outcome.states) {
            state = branch == 0 ? oracle_branch_zero(state, epoch)
                                : oracle_branch_one(state, epoch);
        }
    }
    outcome.state_checksum = state_checksum(outcome.states);
    outcome.decision_hash = decision_hash(outcome.decisions);
    return outcome;
}

std::optional<std::string> first_state_difference(const std::vector<AgentState>& expected,
                                                  const std::vector<AgentState>& observed) {
    if (expected.size() != observed.size()) return "state vector length mismatch";
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const AgentState& left = expected[index];
        const AgentState& right = observed[index];
        if (left.token == right.token && left.budget == right.budget &&
            left.score == right.score && left.route == right.route) {
            continue;
        }
        std::ostringstream message;
        message << "first mismatch at agent " << index << ": expected=(" << left.token << ','
                << left.budget << ',' << left.score << ',' << left.route << "), observed=("
                << right.token << ',' << right.budget << ',' << right.score << ','
                << right.route << ')';
        return message.str();
    }
    return std::nullopt;
}

std::string cuda_message(cudaError_t status) {
    const char* name = cudaGetErrorName(status);
    const char* message = cudaGetErrorString(status);
    return std::string(name == nullptr ? "cudaErrorUnknown" : name) + ": " +
           (message == nullptr ? "unknown CUDA error" : message);
}

HardwareInfo discover_hardware() {
    HardwareInfo info;
    cudaRuntimeGetVersion(&info.runtime_version);
    cudaDriverGetVersion(&info.driver_version);
    cudaError_t status = cudaGetDeviceCount(&info.device_count);
    if (status != cudaSuccess || info.device_count == 0) {
        info.discovery_error = status == cudaSuccess ? "no CUDA device" : cuda_message(status);
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
    const auto attribute = [&](int& destination, cudaDeviceAttr name) {
        if (cudaDeviceGetAttribute(&destination, name, info.device_index) != cudaSuccess) {
            destination = 0;
            cudaGetLastError();
        }
    };
    attribute(info.compute_major, cudaDevAttrComputeCapabilityMajor);
    attribute(info.compute_minor, cudaDevAttrComputeCapabilityMinor);
    attribute(info.multiprocessors, cudaDevAttrMultiProcessorCount);
    attribute(info.unified_addressing, cudaDevAttrUnifiedAddressing);
    attribute(info.pci_domain, cudaDevAttrPciDomainId);
    attribute(info.pci_bus, cudaDevAttrPciBusId);
    attribute(info.pci_device, cudaDevAttrPciDeviceId);
    std::ostringstream uuid;
    uuid << std::hex << std::setfill('0');
    for (int index = 0; index < 16; ++index) {
        uuid << std::setw(2)
             << static_cast<unsigned>(static_cast<unsigned char>(properties.uuid.bytes[index]));
    }
    info.uuid = uuid.str();
    return info;
}

cudaError_t add_route_node(cudaGraph_t graph,
                           cudaGraphNode_t dependency,
                           AgentState* states,
                           std::size_t count,
                           std::uint32_t epoch,
                           std::uint32_t branch,
                           int block_size,
                           cudaGraphNode_t* result) {
    AgentState* states_argument = states;
    std::size_t count_argument = count;
    std::uint32_t epoch_argument = epoch;
    std::uint32_t branch_argument = branch;
    void* arguments[]{&states_argument, &count_argument, &epoch_argument, &branch_argument};
    cudaKernelNodeParams parameters{};
    parameters.func = reinterpret_cast<void*>(route_kernel);
    parameters.gridDim =
        dim3(static_cast<unsigned>((count + static_cast<std::size_t>(block_size) - 1ULL) /
                                   static_cast<std::size_t>(block_size)));
    parameters.blockDim = dim3(static_cast<unsigned>(block_size));
    parameters.kernelParams = arguments;
    return cudaGraphAddKernelNode(
        result, graph, dependency == nullptr ? nullptr : &dependency, dependency == nullptr ? 0 : 1,
        &parameters);
}

cudaError_t add_predicate_nodes(cudaGraph_t graph,
                                cudaGraphNode_t dependency,
                                AgentState* states,
                                std::size_t count,
                                std::uint32_t epoch,
                                std::uint64_t* partials,
                                std::size_t partial_count,
                                int* predicate,
                                int block_size,
                                cudaGraphNode_t* result) {
    AgentState* states_argument = states;
    std::size_t count_argument = count;
    std::uint32_t epoch_argument = epoch;
    std::uint64_t* partials_argument = partials;
    void* partial_arguments[]{
        &states_argument, &count_argument, &epoch_argument, &partials_argument};
    cudaKernelNodeParams partial_parameters{};
    partial_parameters.func = reinterpret_cast<void*>(predicate_partial_kernel);
    partial_parameters.gridDim = dim3(static_cast<unsigned>(partial_count));
    partial_parameters.blockDim = dim3(static_cast<unsigned>(block_size));
    partial_parameters.sharedMemBytes =
        static_cast<unsigned>(block_size * static_cast<int>(sizeof(std::uint64_t)));
    partial_parameters.kernelParams = partial_arguments;
    cudaGraphNode_t partial_node = nullptr;
    cudaError_t status = cudaGraphAddKernelNode(
        &partial_node,
        graph,
        dependency == nullptr ? nullptr : &dependency,
        dependency == nullptr ? 0 : 1,
        &partial_parameters);
    if (status != cudaSuccess) return status;

    std::size_t partial_count_argument = partial_count;
    int* predicate_argument = predicate;
    void* final_arguments[]{
        &partials_argument, &partial_count_argument, &epoch_argument, &predicate_argument};
    cudaKernelNodeParams final_parameters{};
    final_parameters.func = reinterpret_cast<void*>(predicate_finalize_kernel);
    final_parameters.gridDim = dim3(1);
    final_parameters.blockDim = dim3(static_cast<unsigned>(block_size));
    final_parameters.sharedMemBytes =
        static_cast<unsigned>(block_size * static_cast<int>(sizeof(std::uint64_t)));
    final_parameters.kernelParams = final_arguments;
    return cudaGraphAddKernelNode(result, graph, &partial_node, 1, &final_parameters);
}

cudaError_t add_selector_node(cudaGraph_t graph,
                             cudaGraphNode_t dependency,
                             int* predicate,
                             cudaGraphExec_t branch_zero,
                             cudaGraphExec_t branch_one,
                             int* launch_statuses,
                             int* decisions,
                             std::uint32_t epoch,
                             cudaGraphNode_t* result) {
    int* predicate_argument = predicate;
    cudaGraphExec_t branch_zero_argument = branch_zero;
    cudaGraphExec_t branch_one_argument = branch_one;
    int* statuses_argument = launch_statuses;
    int* decisions_argument = decisions;
    std::uint32_t epoch_argument = epoch;
    void* arguments[]{&predicate_argument,
                      &branch_zero_argument,
                      &branch_one_argument,
                      &statuses_argument,
                      &decisions_argument,
                      &epoch_argument};
    cudaKernelNodeParams parameters{};
    parameters.func = reinterpret_cast<void*>(select_and_tail_launch_kernel);
    parameters.gridDim = dim3(1);
    parameters.blockDim = dim3(1);
    parameters.kernelParams = arguments;
    return cudaGraphAddKernelNode(result, graph, &dependency, 1, &parameters);
}

class GraphSet {
public:
    std::vector<cudaGraph_t> host_predicate_graphs;
    std::vector<cudaGraphExec_t> host_predicate_execs;
    std::vector<std::array<cudaGraph_t, 2>> host_route_graphs;
    std::vector<std::array<cudaGraphExec_t, 2>> host_route_execs;
    cudaGraph_t no_decision_graph = nullptr;
    cudaGraphExec_t no_decision_exec = nullptr;
    cudaGraph_t resident_root_graph = nullptr;
    cudaGraphExec_t resident_root_exec = nullptr;
    std::vector<std::array<cudaGraph_t, 2>> resident_path_graphs;
    std::vector<std::array<cudaGraphExec_t, 2>> resident_path_execs;
    bool host_ready = false;
    bool no_decision_ready = false;
    bool resident_ready = false;
    int host_error_code = 0;
    int no_decision_error_code = 0;
    int resident_error_code = 0;
    std::string host_error;
    std::string no_decision_error;
    std::string resident_error;

    GraphSet() = default;
    GraphSet(const GraphSet&) = delete;
    GraphSet& operator=(const GraphSet&) = delete;

    ~GraphSet() {
        if (resident_root_exec) cudaGraphExecDestroy(resident_root_exec);
        if (resident_root_graph) cudaGraphDestroy(resident_root_graph);
        for (auto& pair : resident_path_execs) {
            for (cudaGraphExec_t exec : pair) {
                if (exec) cudaGraphExecDestroy(exec);
            }
        }
        for (auto& pair : resident_path_graphs) {
            for (cudaGraph_t graph : pair) {
                if (graph) cudaGraphDestroy(graph);
            }
        }
        if (no_decision_exec) cudaGraphExecDestroy(no_decision_exec);
        if (no_decision_graph) cudaGraphDestroy(no_decision_graph);
        for (auto& pair : host_route_execs) {
            for (cudaGraphExec_t exec : pair) {
                if (exec) cudaGraphExecDestroy(exec);
            }
        }
        for (auto& pair : host_route_graphs) {
            for (cudaGraph_t graph : pair) {
                if (graph) cudaGraphDestroy(graph);
            }
        }
        for (cudaGraphExec_t exec : host_predicate_execs) {
            if (exec) cudaGraphExecDestroy(exec);
        }
        for (cudaGraph_t graph : host_predicate_graphs) {
            if (graph) cudaGraphDestroy(graph);
        }
    }

    void setup(AgentState* states,
               std::size_t count,
               std::uint32_t epochs,
               int block_size,
               std::uint64_t* partials,
               std::size_t partial_count,
               int* predicate,
               int* launch_statuses,
               int* decisions,
               const std::vector<int>& oracle_decisions,
               bool unified_addressing,
               cudaStream_t stream) {
        setup_host(states, count, epochs, block_size, partials, partial_count, predicate);
        setup_no_decision(states, count, epochs, block_size, oracle_decisions);
        if (!unified_addressing) {
            resident_error = "device graph launch requires unified addressing";
            return;
        }
        setup_resident(states,
                       count,
                       epochs,
                       block_size,
                       partials,
                       partial_count,
                       predicate,
                       launch_statuses,
                       decisions,
                       stream);
    }

private:
    void setup_host(AgentState* states,
                    std::size_t count,
                    std::uint32_t epochs,
                    int block_size,
                    std::uint64_t* partials,
                    std::size_t partial_count,
                    int* predicate) {
        host_predicate_graphs.resize(epochs, nullptr);
        host_predicate_execs.resize(epochs, nullptr);
        host_route_graphs.resize(epochs, {nullptr, nullptr});
        host_route_execs.resize(epochs, {nullptr, nullptr});
        cudaError_t status = cudaSuccess;
        for (std::uint32_t epoch = 0; epoch < epochs && status == cudaSuccess; ++epoch) {
            status = cudaGraphCreate(&host_predicate_graphs[epoch], 0);
            cudaGraphNode_t final_node = nullptr;
            if (status == cudaSuccess) {
                status = add_predicate_nodes(host_predicate_graphs[epoch],
                                             nullptr,
                                             states,
                                             count,
                                             epoch,
                                             partials,
                                             partial_count,
                                             predicate,
                                             block_size,
                                             &final_node);
            }
            if (status == cudaSuccess) {
                status = cudaGraphInstantiateWithFlags(
                    &host_predicate_execs[epoch], host_predicate_graphs[epoch], 0);
            }
            for (std::uint32_t branch = 0; branch < 2U && status == cudaSuccess; ++branch) {
                status = cudaGraphCreate(&host_route_graphs[epoch][branch], 0);
                cudaGraphNode_t route = nullptr;
                if (status == cudaSuccess) {
                    status = add_route_node(host_route_graphs[epoch][branch],
                                            nullptr,
                                            states,
                                            count,
                                            epoch,
                                            branch,
                                            block_size,
                                            &route);
                }
                if (status == cudaSuccess) {
                    status = cudaGraphInstantiateWithFlags(
                        &host_route_execs[epoch][branch], host_route_graphs[epoch][branch], 0);
                }
            }
        }
        if (status == cudaSuccess) {
            host_ready = true;
        } else {
            host_error_code = static_cast<int>(status);
            host_error = cuda_message(status);
            cudaGetLastError();
        }
    }

    void setup_no_decision(AgentState* states,
                           std::size_t count,
                           std::uint32_t epochs,
                           int block_size,
                           const std::vector<int>& oracle_decisions) {
        cudaError_t status = cudaGraphCreate(&no_decision_graph, 0);
        cudaGraphNode_t previous = nullptr;
        for (std::uint32_t epoch = 0; epoch < epochs && status == cudaSuccess; ++epoch) {
            cudaGraphNode_t route = nullptr;
            status = add_route_node(no_decision_graph,
                                    previous,
                                    states,
                                    count,
                                    epoch,
                                    static_cast<std::uint32_t>(oracle_decisions[epoch]),
                                    block_size,
                                    &route);
            previous = route;
        }
        if (status == cudaSuccess) {
            status = cudaGraphInstantiateWithFlags(&no_decision_exec, no_decision_graph, 0);
        }
        if (status == cudaSuccess) {
            no_decision_ready = true;
        } else {
            no_decision_error_code = static_cast<int>(status);
            no_decision_error = cuda_message(status);
            cudaGetLastError();
        }
    }

    void setup_resident(AgentState* states,
                        std::size_t count,
                        std::uint32_t epochs,
                        int block_size,
                        std::uint64_t* partials,
                        std::size_t partial_count,
                        int* predicate,
                        int* launch_statuses,
                        int* decisions,
                        cudaStream_t stream) {
        resident_path_graphs.resize(epochs, {nullptr, nullptr});
        resident_path_execs.resize(epochs, {nullptr, nullptr});
        cudaError_t status = cudaSuccess;
        for (std::int64_t epoch = static_cast<std::int64_t>(epochs) - 1;
             epoch >= 0 && status == cudaSuccess;
             --epoch) {
            for (std::uint32_t branch = 0; branch < 2U && status == cudaSuccess; ++branch) {
                cudaGraph_t& graph = resident_path_graphs[static_cast<std::size_t>(epoch)][branch];
                cudaGraphExec_t& exec =
                    resident_path_execs[static_cast<std::size_t>(epoch)][branch];
                status = cudaGraphCreate(&graph, 0);
                cudaGraphNode_t previous = nullptr;
                if (status == cudaSuccess) {
                    status = add_route_node(graph,
                                            nullptr,
                                            states,
                                            count,
                                            static_cast<std::uint32_t>(epoch),
                                            branch,
                                            block_size,
                                            &previous);
                }
                if (status == cudaSuccess && static_cast<std::uint32_t>(epoch + 1) < epochs) {
                    cudaGraphNode_t final_predicate = nullptr;
                    status = add_predicate_nodes(graph,
                                                 previous,
                                                 states,
                                                 count,
                                                 static_cast<std::uint32_t>(epoch + 1),
                                                 partials,
                                                 partial_count,
                                                 predicate,
                                                 block_size,
                                                 &final_predicate);
                    if (status == cudaSuccess) {
                        const auto& next =
                            resident_path_execs[static_cast<std::size_t>(epoch + 1)];
                        status = add_selector_node(graph,
                                                   final_predicate,
                                                   predicate,
                                                   next[0],
                                                   next[1],
                                                   launch_statuses,
                                                   decisions,
                                                   static_cast<std::uint32_t>(epoch + 1),
                                                   &previous);
                    }
                }
                if (status == cudaSuccess) {
                    status = cudaGraphInstantiateWithFlags(
                        &exec, graph, cudaGraphInstantiateFlagDeviceLaunch);
                }
                if (status == cudaSuccess) status = cudaGraphUpload(exec, stream);
            }
        }
        if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
        if (status == cudaSuccess) status = cudaGraphCreate(&resident_root_graph, 0);
        cudaGraphNode_t predicate_node = nullptr;
        if (status == cudaSuccess) {
            status = add_predicate_nodes(resident_root_graph,
                                         nullptr,
                                         states,
                                         count,
                                         0U,
                                         partials,
                                         partial_count,
                                         predicate,
                                         block_size,
                                         &predicate_node);
        }
        cudaGraphNode_t selector = nullptr;
        if (status == cudaSuccess) {
            status = add_selector_node(resident_root_graph,
                                       predicate_node,
                                       predicate,
                                       resident_path_execs[0][0],
                                       resident_path_execs[0][1],
                                       launch_statuses,
                                       decisions,
                                       0U,
                                       &selector);
        }
        if (status == cudaSuccess) {
            status = cudaGraphInstantiateWithFlags(&resident_root_exec, resident_root_graph, 0);
        }
        if (status == cudaSuccess) {
            resident_ready = true;
        } else {
            resident_error_code = static_cast<int>(status);
            resident_error = cuda_message(status);
            cudaGetLastError();
        }
    }
};

struct DeviceBuffers {
    AgentState* initial = nullptr;
    AgentState* working = nullptr;
    std::uint64_t* partials = nullptr;
    int* predicate = nullptr;
    int* decisions = nullptr;
    int* launch_statuses = nullptr;
    int* pinned_predicate = nullptr;
    bool ready = false;
    int error_code = 0;
    std::string error;

    DeviceBuffers() = default;
    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;

    ~DeviceBuffers() {
        if (pinned_predicate) cudaFreeHost(pinned_predicate);
        if (launch_statuses) cudaFree(launch_statuses);
        if (decisions) cudaFree(decisions);
        if (predicate) cudaFree(predicate);
        if (partials) cudaFree(partials);
        if (working) cudaFree(working);
        if (initial) cudaFree(initial);
    }

    void setup(const std::vector<AgentState>& host_initial,
               std::size_t partial_count,
               std::uint32_t epochs) {
        cudaError_t status = cudaMalloc(&initial, host_initial.size() * sizeof(AgentState));
        if (status == cudaSuccess) {
            status = cudaMalloc(&working, host_initial.size() * sizeof(AgentState));
        }
        if (status == cudaSuccess) {
            status = cudaMalloc(&partials, partial_count * sizeof(std::uint64_t));
        }
        if (status == cudaSuccess) status = cudaMalloc(&predicate, sizeof(int));
        if (status == cudaSuccess) status = cudaMalloc(&decisions, epochs * sizeof(int));
        if (status == cudaSuccess) {
            status = cudaMalloc(&launch_statuses, epochs * sizeof(int));
        }
        if (status == cudaSuccess) status = cudaMallocHost(&pinned_predicate, sizeof(int));
        if (status == cudaSuccess) {
            status = cudaMemcpy(initial,
                                host_initial.data(),
                                host_initial.size() * sizeof(AgentState),
                                cudaMemcpyHostToDevice);
        }
        if (status == cudaSuccess) {
            ready = true;
        } else {
            error_code = static_cast<int>(status);
            error = cuda_message(status);
            cudaGetLastError();
        }
    }
};

Invocation run_invocation(const std::string& mechanism,
                          GraphSet& graphs,
                          DeviceBuffers& buffers,
                          const std::vector<AgentState>& initial,
                          const OracleOutcome& oracle,
                          std::size_t partial_count,
                          std::uint32_t epochs,
                          int block_size,
                          cudaStream_t stream,
                          cudaEvent_t event_start,
                          cudaEvent_t event_stop,
                          std::vector<AgentState>& host_output) {
    Invocation result;
    const std::size_t state_bytes = initial.size() * sizeof(AgentState);
    cudaError_t status = cudaMemcpyAsync(
        buffers.working, buffers.initial, state_bytes, cudaMemcpyDeviceToDevice, stream);
    if (status == cudaSuccess) {
        status = cudaMemsetAsync(buffers.decisions, 0xff, epochs * sizeof(int), stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemsetAsync(buffers.launch_statuses, 0xff, epochs * sizeof(int), stream);
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

    std::vector<int> observed_decisions;
    observed_decisions.reserve(epochs);
    status = cudaEventRecord(event_start, stream);
    const auto wall_start = Clock::now();
    if (status == cudaSuccess && mechanism == "host_roundtrip") {
        for (std::uint32_t epoch = 0; epoch < epochs && status == cudaSuccess; ++epoch) {
            status = cudaGraphLaunch(graphs.host_predicate_execs[epoch], stream);
            if (status == cudaSuccess) {
                status = cudaMemcpyAsync(buffers.pinned_predicate,
                                         buffers.predicate,
                                         sizeof(int),
                                         cudaMemcpyDeviceToHost,
                                         stream);
            }
            if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess) break;
            const std::uint32_t branch =
                static_cast<std::uint32_t>(*buffers.pinned_predicate & 1);
            observed_decisions.push_back(static_cast<int>(branch));
            status = cudaGraphLaunch(graphs.host_route_execs[epoch][branch], stream);
        }
    } else if (status == cudaSuccess && mechanism == "device_resident") {
        status = cudaGraphLaunch(graphs.resident_root_exec, stream);
    } else if (status == cudaSuccess && mechanism == "no_decision_lower_bound") {
        status = cudaGraphLaunch(graphs.no_decision_exec, stream);
    } else if (status == cudaSuccess) {
        status = cudaErrorInvalidValue;
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
    result.wall_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(wall_stop - wall_start).count();
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
    result.device_ns = static_cast<std::uint64_t>(elapsed_ms * 1'000'000.0F);

    if (mechanism == "device_resident") {
        observed_decisions.resize(epochs);
        std::vector<int> launch_statuses(epochs, -1);
        status = cudaMemcpy(observed_decisions.data(),
                            buffers.decisions,
                            epochs * sizeof(int),
                            cudaMemcpyDeviceToHost);
        if (status == cudaSuccess) {
            status = cudaMemcpy(launch_statuses.data(),
                                buffers.launch_statuses,
                                epochs * sizeof(int),
                                cudaMemcpyDeviceToHost);
        }
        if (status == cudaSuccess) {
            for (std::uint32_t epoch = 0; epoch < epochs; ++epoch) {
                if (launch_statuses[epoch] != static_cast<int>(cudaSuccess)) {
                    result.status = "runtime_failure";
                    result.failure_stage = "device_tail_launch";
                    result.error_code = launch_statuses[epoch];
                    result.error_message = "device-side cudaGraphLaunch failed at epoch " +
                                           std::to_string(epoch);
                    return result;
                }
            }
        }
    } else if (mechanism == "no_decision_lower_bound") {
        // The lower bound replays the oracle path and performs no predicate at runtime.
        observed_decisions = oracle.decisions;
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_output.data(),
                            buffers.working,
                            state_bytes,
                            cudaMemcpyDeviceToHost);
    }
    if (status != cudaSuccess) {
        result.status = "runtime_failure";
        result.failure_stage = "validation_copy";
        result.error_code = static_cast<int>(status);
        result.error_message = cuda_message(status);
        cudaGetLastError();
        return result;
    }

    result.observed_state_checksum = state_checksum(host_output);
    result.observed_decision_hash = decision_hash(observed_decisions);
    result.observed_decisions = decision_string(observed_decisions);
    const std::optional<std::string> state_difference =
        first_state_difference(oracle.states, host_output);
    result.exact_state_match = !state_difference.has_value();
    result.exact_decision_match = observed_decisions == oracle.decisions;
    if (!result.exact_state_match || !result.exact_decision_match) {
        result.status = "correctness_failure";
        result.failure_stage = !result.exact_decision_match ? "decision_trace" : "state_fields";
        result.error_message = !result.exact_decision_match
                                   ? "observed decision trace differs from host-only oracle"
                                   : *state_difference;
    } else if (result.observed_state_checksum != oracle.state_checksum ||
               result.observed_decision_hash != oracle.decision_hash) {
        result.status = "correctness_failure";
        result.failure_stage = "checksum_after_exact_comparison";
        result.error_message = "field-equal result produced an unequal checksum";
        result.exact_state_match = false;
        result.exact_decision_match = false;
    }
    return result;
}

BatchMeasurement run_batch(const std::string& mechanism,
                           std::uint64_t initial_iterations,
                           std::uint64_t min_duration_ns,
                           std::uint64_t max_batch_iterations,
                           GraphSet& graphs,
                           DeviceBuffers& buffers,
                           const std::vector<AgentState>& initial,
                           const OracleOutcome& oracle,
                           std::size_t partial_count,
                           std::uint32_t epochs,
                           int block_size,
                           cudaStream_t stream,
                           cudaEvent_t event_start,
                           cudaEvent_t event_stop,
                           std::vector<AgentState>& host_output) {
    BatchMeasurement batch;
    std::uint64_t device_total = 0;
    bool all_device_times = true;
    for (std::uint64_t iteration = 0;
         iteration < max_batch_iterations &&
         (iteration < initial_iterations || batch.aggregate_wall_ns < min_duration_ns);
         ++iteration) {
        const Invocation result = run_invocation(mechanism,
                                                 graphs,
                                                 buffers,
                                                 initial,
                                                 oracle,
                                                 partial_count,
                                                 epochs,
                                                 block_size,
                                                 stream,
                                                 event_start,
                                                 event_stop,
                                                 host_output);
        batch.aggregate_wall_ns += result.wall_ns;
        if (result.device_ns) {
            device_total += *result.device_ns;
        } else {
            all_device_times = false;
        }
        batch.observed_state_checksum = result.observed_state_checksum;
        batch.observed_decision_hash = result.observed_decision_hash;
        batch.observed_decisions = result.observed_decisions;
        batch.exact_state_match = result.exact_state_match;
        batch.exact_decision_match = result.exact_decision_match;
        ++batch.batch_iterations;
        if (result.status == "ok" && result.exact_state_match && result.exact_decision_match) {
            ++batch.exact_validation_count;
        } else {
            batch.status = result.status;
            batch.failure_stage = result.failure_stage;
            batch.error_code = result.error_code;
            batch.error_message = "batch iteration " + std::to_string(iteration) + ": " +
                                  result.error_message;
            break;
        }
    }
    if (all_device_times && batch.exact_validation_count == batch.batch_iterations) {
        batch.aggregate_device_ns = device_total;
    }
    return batch;
}

class CsvWriter {
public:
    explicit CsvWriter(const fs::path& path) : stream_(path, std::ios::out | std::ios::app) {
        if (!stream_) throw std::runtime_error("cannot create CSV: " + path.string());
        if (fs::file_size(path) == 0ULL) {
            stream_
                << "schema_version,timestamp_utc,run_id,experiment_id,phase,mechanism,agents,"
                   "epochs,repetition,order_index,status,failure_stage,error_code,error_message,"
                   "batch_iterations,aggregate_wall_ns,wall_ns_per_invocation,"
                   "aggregate_device_ns,device_ns_per_invocation,min_duration_target_ns,"
                   "min_duration_reached,expected_state_checksum,observed_state_checksum,"
                   "expected_decision_hash,observed_decision_hash,expected_decisions,"
                   "observed_decisions,exact_state_match,exact_decision_match,"
                   "exact_validation_count,seed,block_size,predicate_blocks\n";
            stream_.flush();
        }
    }

    void write(const Row& row) {
        stream_ << csv_escape(kSchemaVersion) << ',' << csv_escape(row.timestamp_utc) << ','
                << csv_escape(row.run_id) << ',' << csv_escape(row.experiment_id) << ','
                << csv_escape(row.phase) << ',' << csv_escape(row.mechanism) << ','
                << row.agents << ',' << row.epochs << ',' << row.repetition << ','
                << row.order_index << ',' << csv_escape(row.status) << ','
                << csv_escape(row.failure_stage) << ',' << row.error_code << ','
                << csv_escape(row.error_message) << ',' << row.batch_iterations << ',';
        if (row.aggregate_wall_ns) stream_ << *row.aggregate_wall_ns;
        stream_ << ',';
        if (row.wall_ns_per_invocation) stream_ << std::setprecision(17) << *row.wall_ns_per_invocation;
        stream_ << ',';
        if (row.aggregate_device_ns) stream_ << *row.aggregate_device_ns;
        stream_ << ',';
        if (row.device_ns_per_invocation) {
            stream_ << std::setprecision(17) << *row.device_ns_per_invocation;
        }
        stream_ << ',' << row.min_duration_target_ns << ','
                << (row.min_duration_reached ? "true" : "false") << ','
                << row.expected_state_checksum << ',';
        if (row.observed_state_checksum) stream_ << *row.observed_state_checksum;
        stream_ << ',' << row.expected_decision_hash << ',';
        if (row.observed_decision_hash) stream_ << *row.observed_decision_hash;
        stream_ << ',' << csv_escape(row.expected_decisions) << ','
                << csv_escape(row.observed_decisions) << ','
                << (row.exact_state_match ? "true" : "false") << ','
                << (row.exact_decision_match ? "true" : "false") << ','
                << row.exact_validation_count << ',' << row.seed << ',' << row.block_size << ','
                << row.predicate_blocks << '\n';
        stream_.flush();
        if (!stream_) throw std::runtime_error("failed while appending CSV row");
    }

private:
    std::ofstream stream_;
};

Row make_row(const Config& config,
             const std::string& run_id,
             const std::string& phase,
             const std::string& mechanism,
             std::size_t agents,
             std::uint32_t epochs,
             int repetition,
             int order_index,
             std::size_t predicate_blocks,
             const OracleOutcome& oracle,
             const BatchMeasurement& measurement) {
    Row row;
    row.timestamp_utc = utc_timestamp();
    row.run_id = run_id;
    row.experiment_id = config.experiment_id;
    row.phase = phase;
    row.mechanism = mechanism;
    row.agents = agents;
    row.epochs = epochs;
    row.repetition = repetition;
    row.order_index = order_index;
    row.status = measurement.status;
    row.failure_stage = measurement.failure_stage;
    row.error_code = measurement.error_code;
    row.error_message = measurement.error_message;
    row.batch_iterations = measurement.batch_iterations;
    if (measurement.batch_iterations > 0ULL && measurement.aggregate_wall_ns > 0ULL) {
        row.aggregate_wall_ns = measurement.aggregate_wall_ns;
        row.wall_ns_per_invocation =
            static_cast<double>(measurement.aggregate_wall_ns) /
            static_cast<double>(measurement.batch_iterations);
    }
    row.aggregate_device_ns = measurement.aggregate_device_ns;
    if (measurement.aggregate_device_ns && measurement.batch_iterations > 0ULL) {
        row.device_ns_per_invocation =
            static_cast<double>(*measurement.aggregate_device_ns) /
            static_cast<double>(measurement.batch_iterations);
    }
    row.min_duration_target_ns = config.min_duration_ns;
    row.min_duration_reached = measurement.aggregate_wall_ns >= config.min_duration_ns;
    row.expected_state_checksum = oracle.state_checksum;
    if (measurement.exact_validation_count > 0ULL) {
        row.observed_state_checksum = measurement.observed_state_checksum;
        row.observed_decision_hash = measurement.observed_decision_hash;
    }
    row.expected_decision_hash = oracle.decision_hash;
    row.expected_decisions = decision_string(oracle.decisions);
    row.observed_decisions = measurement.observed_decisions;
    row.exact_state_match = measurement.exact_state_match;
    row.exact_decision_match = measurement.exact_decision_match;
    row.exact_validation_count = measurement.exact_validation_count;
    row.seed = config.seed;
    row.block_size = config.block_size;
    row.predicate_blocks = predicate_blocks;
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
                    const std::vector<CellAudit>& cells,
                    const std::map<std::string, std::uint64_t>& status_counts,
                    std::uint64_t measured_rows,
                    std::uint64_t exact_rows,
                    std::uint64_t failure_rows) {
    if (fs::exists(path)) throw std::runtime_error("refusing to overwrite manifest");
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create manifest: " + path.string());
    const std::string environment_source = environment_or_empty("SOURCE_SHA256");
    const std::string source_sha =
        environment_source.empty() ? RESIDENT_POLICY_SOURCE_SHA256 : environment_source;
    output << "{\n"
           << "  \"schema_version\": \"" << kSchemaVersion << "\",\n"
           << "  \"run_id\": \"" << json_escape(run_id) << "\",\n"
           << "  \"experiment_id\": \"" << json_escape(config.experiment_id) << "\",\n"
           << "  \"started_at_utc\": \"" << started_at << "\",\n"
           << "  \"completed_at_utc\": \"" << completed_at << "\",\n"
           << "  \"csv_file\": \"" << json_escape(csv_path.filename().string()) << "\",\n"
           << "  \"provenance\": {\n"
           << "    \"execution_provider\": \""
           << json_escape(environment_or_empty("EXECUTION_PROVIDER")) << "\",\n"
           << "    \"requested_gpu\": \"" << json_escape(environment_or_empty("REQUESTED_GPU"))
           << "\",\n"
           << "    \"placement_id\": \"" << json_escape(environment_or_empty("PLACEMENT_ID"))
           << "\",\n"
           << "    \"image_digest\": \"" << json_escape(environment_or_empty("IMAGE_DIGEST"))
           << "\",\n"
           << "    \"source_sha256\": \"" << json_escape(source_sha) << "\",\n"
           << "    \"binary_sha256\": \""
           << json_escape(environment_or_empty("BINARY_SHA256")) << "\",\n"
           << "    \"git_commit\": \"" << json_escape(RESIDENT_POLICY_GIT_COMMIT) << "\",\n"
           << "    \"git_dirty_at_compile\": \"" << json_escape(RESIDENT_POLICY_GIT_DIRTY)
           << "\",\n"
           << "    \"compile_date_utc_unverified\": \"" << __DATE__ << " " << __TIME__
           << "\"\n"
           << "  },\n"
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
           << "    \"unified_addressing\": " << hardware.unified_addressing << ",\n"
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
           << "    \"epoch_counts\": " << json_array(config.epoch_counts) << ",\n"
           << "    \"warmups_per_mechanism_cell\": " << config.warmups << ",\n"
           << "    \"calibration_samples_per_mechanism_cell\": "
           << config.calibration_samples << ",\n"
           << "    \"repetitions_per_mechanism_cell\": " << config.repetitions << ",\n"
           << "    \"min_duration_target_ns\": " << config.min_duration_ns << ",\n"
           << "    \"max_batch_iterations\": " << config.max_batch_iterations << ",\n"
           << "    \"seed\": " << config.seed << ",\n"
           << "    \"block_size\": " << config.block_size << ",\n"
           << "    \"mechanisms\": [\"host_roundtrip\",\"device_resident\","
              "\"no_decision_lower_bound\"],\n"
           << "    \"mechanism_order\": \"deterministically shuffled within cell and repetition\",\n"
           << "    \"batch_policy\": \"one calibrated common lower-bound count per cell; each row extends until its aggregate wall time reaches the target, with a safety cap\",\n"
           << "    \"state_reset_in_timing\": false,\n"
           << "    \"result_copy_or_validation_in_timing\": false,\n"
           << "    \"host_predicate_copy_and_sync_in_timing\": true,\n"
           << "    \"graph_instantiation_or_upload_in_timing\": false\n"
           << "  },\n"
           << "  \"cells\": [";
    for (std::size_t index = 0; index < cells.size(); ++index) {
        const CellAudit& cell = cells[index];
        if (index != 0) output << ',';
        output << "\n    {\"agents\":" << cell.agents << ",\"epochs\":" << cell.epochs
               << ",\"common_batch_iterations\":" << cell.common_batch_iterations
               << ",\"batch_cap_reached\":"
               << (cell.batch_cap_reached ? "true" : "false")
               << ",\"median_calibration_wall_ns\":{";
        bool first_calibration = true;
        for (const auto& [mechanism, duration] : cell.median_calibration_wall_ns) {
            if (!first_calibration) output << ',';
            output << "\"" << json_escape(mechanism) << "\":" << duration;
            first_calibration = false;
        }
        output << "}}";
    }
    if (!cells.empty()) output << '\n';
    output << "  ],\n"
           << "  \"results\": {\n"
           << "    \"measured_rows\": " << measured_rows << ",\n"
           << "    \"exact_rows\": " << exact_rows << ",\n"
           << "    \"failure_rows\": " << failure_rows << ",\n"
           << "    \"status_counts\": {";
    bool first_status = true;
    for (const auto& [status, count] : status_counts) {
        if (!first_status) output << ',';
        output << "\n      \"" << json_escape(status) << "\": " << count;
        first_status = false;
    }
    if (!status_counts.empty()) output << '\n';
    output << "    }\n"
           << "  },\n"
           << "  \"semantic_contract\": {\n"
           << "    \"host_roundtrip\": \"identical GPU predicate and route kernels; one 4-byte D2H predicate synchronization and host graph selection per epoch\",\n"
           << "    \"device_resident\": \"GPU predicate followed by a one-thread selector that tail-launches one of two pre-uploaded device graphs per epoch\",\n"
           << "    \"no_decision_lower_bound\": \"one host-launched graph containing only the oracle-selected route kernels; predicate and selection removed\",\n"
           << "    \"oracle\": \"separately written host-only predicate and route functions; every state field and every decision compared after every invocation\"\n"
           << "  },\n"
           << "  \"limitations\": [\n"
           << "    \"CUDA graph topology is fixed, instantiated, and uploaded by the host\",\n"
           << "    \"the policy selects one global binary route; it is not per-agent divergent scheduling\",\n"
           << "    \"the lower bound removes predicate work and is a floor, not a competing scheduler\",\n"
           << "    \"tool I/O, authority checks, memory isolation, and failures are outside the timed kernel\",\n"
           << "    \"CPU affinity, NUMA placement, and accelerator clocks are not controlled\",\n"
           << "    \"a placement is the independent sampling unit; rows and batch invocations are technical repeats\"\n"
           << "  ]\n"
           << "}\n";
    output.flush();
    if (!output) throw std::runtime_error("failed while writing manifest");
}

int run(const Config& config) {
    fs::create_directories(config.output_dir);
    const std::string started_at = utc_timestamp();
    const std::string run_id = config.experiment_id + "-" + utc_timestamp(true) + "-p" +
                               std::to_string(static_cast<long long>(getpid()));
    const fs::path csv_path = config.output_dir / (run_id + ".csv");
    const fs::path manifest_path = config.output_dir / (run_id + ".manifest.json");
    if (fs::exists(csv_path) || fs::exists(manifest_path)) {
        throw std::runtime_error("refusing to overwrite an existing artifact");
    }
    CsvWriter csv(csv_path);
    HardwareInfo hardware = discover_hardware();
    std::map<std::string, std::uint64_t> status_counts;
    std::vector<CellAudit> cell_audits;
    std::uint64_t measured_rows = 0ULL;
    std::uint64_t exact_rows = 0ULL;
    std::uint64_t failure_rows = 0ULL;
    const auto persist = [&](const Row& row) {
        csv.write(row);
        ++status_counts[row.status];
        if (row.phase == "measure") ++measured_rows;
        if (row.phase == "measure" && row.status == "ok" && row.exact_state_match &&
            row.exact_decision_match && row.exact_validation_count == row.batch_iterations) {
            ++exact_rows;
        }
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
            hardware.discovery_error = "CUDA timing setup failed: " + cuda_message(status);
            cudaGetLastError();
        }
    }

    for (const std::size_t agents : config.agent_counts) {
        const std::uint64_t cell_seed =
            config.seed ^ (static_cast<std::uint64_t>(agents) * 0x9e3779b97f4a7c15ULL);
        const std::vector<AgentState> initial = make_initial_states(agents, cell_seed);
        const std::size_t predicate_blocks =
            (agents + static_cast<std::size_t>(config.block_size) - 1ULL) /
            static_cast<std::size_t>(config.block_size);
        for (const std::uint32_t epochs : config.epoch_counts) {
            const OracleOutcome oracle = run_oracle(initial, epochs);
            DeviceBuffers buffers;
            if (hardware.available) buffers.setup(initial, predicate_blocks, epochs);
            GraphSet graphs;
            if (buffers.ready) {
                graphs.setup(buffers.working,
                             agents,
                             epochs,
                             config.block_size,
                             buffers.partials,
                             predicate_blocks,
                             buffers.predicate,
                             buffers.launch_statuses,
                             buffers.decisions,
                             oracle.decisions,
                             hardware.unified_addressing != 0,
                             stream);
            }
            std::map<std::string, bool> available{
                {"host_roundtrip", buffers.ready && graphs.host_ready},
                {"device_resident", buffers.ready && graphs.resident_ready},
                {"no_decision_lower_bound", buffers.ready && graphs.no_decision_ready},
            };
            for (const char* mechanism_ptr : kMechanisms) {
                const std::string mechanism = mechanism_ptr;
                if (available[mechanism]) continue;
                BatchMeasurement failure;
                failure.status = hardware.available ? "setup_failure" : "unsupported";
                failure.failure_stage = "mechanism_setup";
                if (!hardware.available) {
                    failure.error_message = hardware.discovery_error;
                } else if (!buffers.ready) {
                    failure.error_code = buffers.error_code;
                    failure.error_message = buffers.error;
                } else if (mechanism == "host_roundtrip") {
                    failure.error_code = graphs.host_error_code;
                    failure.error_message = graphs.host_error;
                } else if (mechanism == "device_resident") {
                    failure.error_code = graphs.resident_error_code;
                    failure.error_message = graphs.resident_error;
                } else {
                    failure.error_code = graphs.no_decision_error_code;
                    failure.error_message = graphs.no_decision_error;
                }
                persist(make_row(config,
                                 run_id,
                                 "setup",
                                 mechanism,
                                 agents,
                                 epochs,
                                 -1,
                                 -1,
                                 predicate_blocks,
                                 oracle,
                                 failure));
            }

            std::vector<AgentState> host_output(agents);
            const auto one = [&](const std::string& mechanism) {
                return run_invocation(mechanism,
                                      graphs,
                                      buffers,
                                      initial,
                                      oracle,
                                      predicate_blocks,
                                      epochs,
                                      config.block_size,
                                      stream,
                                      event_start,
                                      event_stop,
                                      host_output);
            };

            for (int warmup = 0; warmup < config.warmups; ++warmup) {
                for (const char* mechanism_ptr : kMechanisms) {
                    const std::string mechanism = mechanism_ptr;
                    if (!available[mechanism]) continue;
                    const Invocation result = one(mechanism);
                    if (result.status != "ok") {
                        available[mechanism] = false;
                        BatchMeasurement failure;
                        failure.status = result.status;
                        failure.failure_stage = result.failure_stage;
                        failure.error_code = result.error_code;
                        failure.error_message = result.error_message;
                        failure.batch_iterations = 1ULL;
                        failure.aggregate_wall_ns = result.wall_ns;
                        failure.aggregate_device_ns = result.device_ns;
                        failure.observed_state_checksum = result.observed_state_checksum;
                        failure.observed_decision_hash = result.observed_decision_hash;
                        failure.observed_decisions = result.observed_decisions;
                        failure.exact_state_match = result.exact_state_match;
                        failure.exact_decision_match = result.exact_decision_match;
                        persist(make_row(config,
                                         run_id,
                                         "warmup",
                                         mechanism,
                                         agents,
                                         epochs,
                                         -1,
                                         warmup,
                                         predicate_blocks,
                                         oracle,
                                         failure));
                    }
                }
            }

            CellAudit audit;
            audit.agents = agents;
            audit.epochs = epochs;
            std::uint64_t common_batch = 1ULL;
            for (const char* mechanism_ptr : kMechanisms) {
                const std::string mechanism = mechanism_ptr;
                if (!available[mechanism]) continue;
                std::vector<std::uint64_t> durations;
                for (int sample = 0; sample < config.calibration_samples; ++sample) {
                    const Invocation result = one(mechanism);
                    if (result.status != "ok") {
                        available[mechanism] = false;
                        BatchMeasurement failure;
                        failure.status = result.status;
                        failure.failure_stage = result.failure_stage;
                        failure.error_code = result.error_code;
                        failure.error_message = result.error_message;
                        failure.batch_iterations = 1ULL;
                        failure.aggregate_wall_ns = result.wall_ns;
                        failure.aggregate_device_ns = result.device_ns;
                        failure.observed_state_checksum = result.observed_state_checksum;
                        failure.observed_decision_hash = result.observed_decision_hash;
                        failure.observed_decisions = result.observed_decisions;
                        failure.exact_state_match = result.exact_state_match;
                        failure.exact_decision_match = result.exact_decision_match;
                        persist(make_row(config,
                                         run_id,
                                         "calibration",
                                         mechanism,
                                         agents,
                                         epochs,
                                         -1,
                                         sample,
                                         predicate_blocks,
                                         oracle,
                                         failure));
                        break;
                    }
                    durations.push_back(std::max<std::uint64_t>(1ULL, result.wall_ns));
                }
                if (!available[mechanism]) continue;
                std::sort(durations.begin(), durations.end());
                const std::uint64_t median = durations[durations.size() / 2ULL];
                audit.median_calibration_wall_ns[mechanism] = median;
                const long double required =
                    std::ceil(static_cast<long double>(config.min_duration_ns) /
                              static_cast<long double>(median));
                const std::uint64_t required_iterations =
                    required > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())
                        ? config.max_batch_iterations
                        : static_cast<std::uint64_t>(required);
                common_batch = std::max(common_batch, required_iterations);
            }
            audit.batch_cap_reached = common_batch > config.max_batch_iterations;
            common_batch = std::min(common_batch, config.max_batch_iterations);
            audit.common_batch_iterations = common_batch;
            cell_audits.push_back(audit);

            std::mt19937_64 generator(
                config.seed ^ (static_cast<std::uint64_t>(agents) * 0xd1b54a32d192ed03ULL) ^
                (static_cast<std::uint64_t>(epochs) << 32U));
            for (int repetition = 0; repetition < config.repetitions; ++repetition) {
                std::vector<std::string> mechanisms;
                for (const char* mechanism_ptr : kMechanisms) {
                    if (available[mechanism_ptr]) mechanisms.emplace_back(mechanism_ptr);
                }
                std::shuffle(mechanisms.begin(), mechanisms.end(), generator);
                for (std::size_t order = 0; order < mechanisms.size(); ++order) {
                    const std::string& mechanism = mechanisms[order];
                    const BatchMeasurement result = run_batch(mechanism,
                                                              common_batch,
                                                              config.min_duration_ns,
                                                              config.max_batch_iterations,
                                                              graphs,
                                                              buffers,
                                                              initial,
                                                              oracle,
                                                              predicate_blocks,
                                                              epochs,
                                                              config.block_size,
                                                              stream,
                                                              event_start,
                                                              event_stop,
                                                              host_output);
                    persist(make_row(config,
                                     run_id,
                                     "measure",
                                     mechanism,
                                     agents,
                                     epochs,
                                     repetition,
                                     static_cast<int>(order),
                                     predicate_blocks,
                                     oracle,
                                     result));
                    if (result.status != "ok") available[mechanism] = false;
                }
            }
        }
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
                   cell_audits,
                   status_counts,
                   measured_rows,
                   exact_rows,
                   failure_rows);
    std::cout << "run_id=" << run_id << '\n'
              << "csv_path=" << fs::absolute(csv_path).string() << '\n'
              << "manifest_path=" << fs::absolute(manifest_path).string() << '\n'
              << "measured_rows=" << measured_rows << '\n'
              << "exact_rows=" << exact_rows << '\n'
              << "failure_rows=" << failure_rows << '\n';
    return failure_rows == 0ULL ? 0 : 1;
}

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "fatal_error=" << error.what() << '\n';
        return 2;
    }
}
