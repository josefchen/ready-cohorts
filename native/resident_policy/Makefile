NVCC ?= nvcc

TARGET := build/resident_policy_pilot
SOURCE := resident_policy_pilot.cu
SOURCE_SHA256 := $(shell sha256sum $(SOURCE) | awk '{print $$1}')
GIT_COMMIT := $(shell git rev-parse --verify HEAD 2>/dev/null || printf unknown)
GIT_DIRTY := $(shell test -n "$$(git status --porcelain --untracked-files=no 2>/dev/null)" && printf true || printf false)

ARCH_FLAGS := \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90 \
	-gencode arch=compute_90,code=compute_90

PROVENANCE_FLAGS := \
	-DRESIDENT_POLICY_SOURCE_SHA256=\"$(SOURCE_SHA256)\" \
	-DRESIDENT_POLICY_GIT_COMMIT=\"$(GIT_COMMIT)\" \
	-DRESIDENT_POLICY_GIT_DIRTY=\"$(GIT_DIRTY)\"

NVCC_FLAGS ?= -O3 -std=c++17 -lineinfo -rdc=true --threads 0

.PHONY: all compile-smoke smoke clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(PROVENANCE_FLAGS) $< -o $@ -lcudadevrt

compile-smoke: $(TARGET)
	./$(TARGET) --help

smoke: $(TARGET)
	@binary_sha="$$(sha256sum $(TARGET) | awk '{print $$1}')"; \
	SOURCE_SHA256="$(SOURCE_SHA256)" BINARY_SHA256="$$binary_sha" \
	EXECUTION_PROVIDER=local REQUESTED_GPU=local \
	./$(TARGET) --smoke --output-dir smoke-results

clean:
	rm -f $(TARGET)
