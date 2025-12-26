# APXM Makefile - Build orchestration
#
# This Makefile provides convenient targets for common development tasks.

.PHONY: all build build-release test test-all clean fmt lint check help

# Default target
all: build

# Build all crates (debug mode, excludes compiler by default)
build:
	cargo build --workspace --exclude apxm-compiler

# Build all crates including compiler (requires MLIR)
build-all:
	cargo build --workspace

# Build in release mode
build-release:
	cargo build --release --workspace --exclude apxm-compiler

# Build only the compiler (requires MLIR)
build-compiler:
	cargo build --package apxm-compiler

# Build only the runtime
build-runtime:
	cargo build --package apxm-runtime

# Run tests (excludes compiler tests by default)
test:
	cargo test --workspace --exclude apxm-compiler

# Run all tests including compiler (requires MLIR)
test-all:
	cargo test --workspace

# Run only runtime tests
test-runtime:
	cargo test --package apxm-runtime

# Clean build artifacts
clean:
	cargo clean

# Format code
fmt:
	cargo fmt

# Check formatting without modifying
fmt-check:
	cargo fmt --check

# Run clippy linter
lint:
	cargo clippy --workspace -- -D warnings

# Fast compile check
check:
	cargo check --workspace

# Help target
help:
	@echo "APXM Development Makefile"
	@echo ""
	@echo "Build targets:"
	@echo "  build        - Build workspace (excludes compiler)"
	@echo "  build-all    - Build all crates including compiler"
	@echo "  build-release- Build in release mode"
	@echo "  build-compiler - Build only compiler (requires MLIR)"
	@echo "  build-runtime  - Build only runtime"
	@echo ""
	@echo "Test targets:"
	@echo "  test         - Run tests (excludes compiler)"
	@echo "  test-all     - Run all tests including compiler"
	@echo "  test-runtime - Run only runtime tests"
	@echo ""
	@echo "Development targets:"
	@echo "  clean        - Clean build artifacts"
	@echo "  fmt          - Format code"
	@echo "  fmt-check    - Check formatting"
	@echo "  lint         - Run clippy linter"
	@echo "  check        - Fast compile check"
