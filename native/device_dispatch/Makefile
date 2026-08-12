NVCC ?= nvcc

TARGET := build/device_dispatch_pilot
SOURCE := device_dispatch_pilot.cu

# CUDA 13 no longer emits offline code for pre-Turing devices. These targets
# cover the Modal GPU classes intended for the pilot: T4 through H100.
ARCH_FLAGS := \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90 \
	-gencode arch=compute_90,code=compute_90

NVCC_FLAGS ?= -O3 -std=c++17 -lineinfo -rdc=true --threads 0

.PHONY: all compile-smoke smoke clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $< -o $@ -lcudadevrt

# This verifies compilation and argument parsing without requiring a GPU.
compile-smoke: $(TARGET)
	./$(TARGET) --help

# This is deliberately tiny, but it does require a CUDA-capable GPU.
smoke: $(TARGET)
	./$(TARGET) --smoke --output-dir smoke-results

clean:
	rm -f $(TARGET)
