# === Variables ===
PYTHON = python
RUFF = ruff
SOURCE_DIR = geomas

# === Commands ===
.PHONY: lint fix format check sort all

lint:
	@echo "🔍 Running Ruff linting..."
	$(RUFF) check $(SOURCE_DIR) $(TESTS_DIR)

fix:
	@echo "🔧 Fixing linting errors..."
	$(RUFF) check --fix $(SOURCE_DIR) $(TESTS_DIR)

format:
	@echo "✨ Formatting code..."
	$(RUFF) format $(SOURCE_DIR) $(TESTS_DIR)

sort:
	@echo "📦 Sorting imports..."
	$(RUFF) check --select I --fix $(SOURCE_DIR) $(TESTS_DIR)

all: lint fix format check