# Compiler
HIPCC = hipcc
CXX = g++

# Directories
SRC_DIR = src
BUILD_DIR = build
LIB_DIR = $(BUILD_DIR)/lib
BIN_DIR = $(BUILD_DIR)/bin
OBJ_DIR = $(BUILD_DIR)/obj

# Source files
SOURCES = $(notdir $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*/*.cpp))
OBJECTS = $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SOURCES)))

# Include directories
INCLUDE_FLAGS = -I$(SRC_DIR) -Icommon

# Linker flags
LIBS = -lrocprofiler-sdk -lrocprofiler-sdk-roctx -lpthread
LDFLAGS = -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib $(LIBS)

# Compiler flags
CXXFLAGS = -std=c++17 -fPIC -Wall
HIPFLAGS = -std=c++17 -fPIC -Wall

# Targets
all: dirs rocm_callback

rocm_callback: $(BIN_DIR)/rocm_callback $(LIB_DIR)/librocm_callback.so

dirs: $(BUILD_DIR) $(LIB_DIR) $(BIN_DIR) $(OBJ_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Build shared library
$(LIB_DIR)/librocm_callback.so: $(OBJ_DIR)/rocm_callback.o
	$(HIPCC) -shared -o $@ $^ $(LDFLAGS)

# Build main application
$(BIN_DIR)/rocm_callback: $(OBJECTS) $(LIB_DIR)/librocm_callback.so
	$(HIPCC) -o $@ $(OBJECTS) $(LDFLAGS) -L$(LIB_DIR) -lrocm_callback

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(HIPCC) $(HIPFLAGS) $(INCLUDE_FLAGS) -c $< -o $@

# Run tests
test: all
	LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $(BIN_DIR)/rocm_callback

# Clean build
clean:
	rm -rf $(BUILD_DIR)
