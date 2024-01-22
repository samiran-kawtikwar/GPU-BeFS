NVCC ?= nvcc
GCC ?= g++

ARCH := $(shell ~/get_SM.sh)
BUILD_DIR ?=./build

all: $(BUILD_DIR)/obj/main.o $(BUILD_DIR)/obj/gurobi_solver.o
	$(NVCC) -o $(BUILD_DIR)/main.exe $(BUILD_DIR)/obj/gurobi_solver.o $(BUILD_DIR)/obj/main.o -L${GUROBI_HOME}/lib -lgurobi_c++ -lgurobi110 -lcuda -lgomp -O3 -I${GUROBI_HOME}/include -arch=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=compute_$(ARCH)

$(BUILD_DIR)/obj/main.o: main.cu
	mkdir -p $(BUILD_DIR)/obj/
	$(NVCC) -c main.cu -o $(BUILD_DIR)/obj/main.o -L${GUROBI_HOME}/lib -lgomp -O3 -I${GUROBI_HOME}/include -arch=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=compute_$(ARCH)

$(BUILD_DIR)/obj/gurobi_solver.o: RCAP/gurobi_solver.cpp
	$(NVCC) -c RCAP/gurobi_solver.cpp -o $(BUILD_DIR)/obj/gurobi_solver.o -L${GUROBI_HOME}/lib -lgurobi_c++ -lgurobi110 -O3 -I${GUROBI_HOME}/include
clean:
	$(RM) -r $(BUILD_DIR)
	@echo SM_VALUE IS $(ARCH)
-include $(DEPS)