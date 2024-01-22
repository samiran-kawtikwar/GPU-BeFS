NVCC ?= nvcc
GCC ?= g++

ARCH := $(shell ~/get_SM.sh)
BUILD_DIR ?=./build

# Find all source files
EXCLUDED_DIRS := scratch build scripts tests logs  # Add the directories you want to exclude here
CU_FILES := $(shell find . -name '*.cu' $(addprefix -not -path "./", $(addsuffix "/*", $(EXCLUDED_DIRS))))
CPP_FILES := $(shell find . -name '*.cpp' $(addprefix -not -path "./", $(addsuffix "/*", $(EXCLUDED_DIRS))))

# Define object files for both cu and cpp
CU_OBJ_FILES := $(patsubst %.cu,$(BUILD_DIR)/obj/%.cu.o,$(notdir $(CU_FILES)))
CPP_OBJ_FILES := $(patsubst %.cpp,$(BUILD_DIR)/obj/%.cpp.o,$(CPP_FILES))

# cuda flags
CUDAFLAGS ?= -g -Xcompiler -fopenmp -lineinfo -O3 -arch=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) \
						-gencode=arch=compute_$(ARCH),code=compute_$(ARCH)
CUDAINC	?=

CPPFLAGS ?= -O3
CPPINC ?= -I${GUROBI_HOME}/include

LDIR ?= -L${GUROBI_HOME}/lib
LDFLAGS ?= -lcuda -lgomp -lgurobi_c++ -lgurobi100

all: $(BUILD_DIR)/main.exe

$(BUILD_DIR)/main.exe: $(CU_OBJ_FILES) $(CPP_OBJ_FILES)
	$(NVCC) -o $@ $(CU_OBJ_FILES) $(CPP_OBJ_FILES) $(LDIR) $(LDFLAGS)

# Pattern rule for cu files
$(BUILD_DIR)/obj/%.cu.o: %.cu
	mkdir -p $(BUILD_DIR)/obj/
	@echo cu obj files are: $(CU_OBJ_FILES)
	@echo cu files are: $(CU_FILES)
	$(NVCC) $(CUDAFLAGS) $(CUDAINC) -c $< -o $@


# Pattern rule for cpp files
$(BUILD_DIR)/obj/%.cpp.o: %.cpp
	@mkdir -p $(BUILD_DIR)/obj/$(dir $<) 
	@echo cpp obj files are: $(CPP_OBJ_FILES)
	@echo cpp files are: $(CPP_FILES)
	$(GCC) $(CPPFLAGS) $(CPPINC) -c $< -o $@


clean:
	$(RM) -r $(BUILD_DIR)
	@echo SM_VALUE IS $(ARCH)