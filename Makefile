.PHONY: all build test clean install run

# Build variables
BINARY_NAME=octopus
SRC_DIR=src
BUILD_DIR=build

all: build

build:
	@echo "Building Octopus..."
	@mkdir -p $(BUILD_DIR)
	@cd $(SRC_DIR) && go build -o ../$(BUILD_DIR)/$(BINARY_NAME) .

test:
	@echo "Running tests..."
	@cd $(SRC_DIR) && go test -v ./...

clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR)
	@cd $(SRC_DIR) && go clean

install:
	@echo "Installing dependencies..."
	@cd $(SRC_DIR) && go mod download

run: build
	@echo "Running Octopus..."
	@$(BUILD_DIR)/$(BINARY_NAME)

run-single:
	@echo "Running Octopus (single node)..."
	@$(BUILD_DIR)/$(BINARY_NAME) --node=0

run-multi:
	@echo "Running Octopus (multiple nodes)..."
	@$(BUILD_DIR)/$(BINARY_NAME) --node=0 &
	@$(BUILD_DIR)/$(BINARY_NAME) --node=1 &
	@$(BUILD_DIR)/$(BINARY_NAME) --node=2 &
	@$(BUILD_DIR)/$(BINARY_NAME) --node=3

fmt:
	@echo "Formatting code..."
	@cd $(SRC_DIR) && go fmt ./...

vet:
	@echo "Running go vet..."
	@cd $(SRC_DIR) && go vet ./...

lint: vet
	@echo "Linting..."

proto:
	@echo "Generating protobuf code..."
	@cd proto && protoc --go_out=../src *.proto

benchmark:
	@echo "Running benchmarks..."
	@cd $(SRC_DIR) && go test -bench=. -benchmem ./...

help:
	@echo "Available targets:"
	@echo "  build        - Build the Octopus binary"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean build artifacts"
	@echo "  install      - Install dependencies"
	@echo "  run          - Run Octopus"
	@echo "  run-single   - Run Octopus with single node"
	@echo "  run-multi    - Run Octopus with multiple nodes"
	@echo "  fmt          - Format code"
	@echo "  vet          - Run go vet"
	@echo "  lint         - Lint code"
	@echo "  proto        - Generate protobuf code"
	@echo "  benchmark    - Run benchmarks"
