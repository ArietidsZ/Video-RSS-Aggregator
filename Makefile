.PHONY: proto proto-go proto-python clean

PROTO_DIR := proto
GEN_DIR := gen

proto: proto-go proto-python

proto-go:
	@echo "Generating Go code..."
	@mkdir -p $(GEN_DIR)/go
	protoc --proto_path=$(PROTO_DIR) \
		--go_out=$(GEN_DIR)/go --go_opt=paths=source_relative \
		--go-grpc_out=$(GEN_DIR)/go --go-grpc_opt=paths=source_relative \
		$(PROTO_DIR)/*.proto

proto-python:
	@echo "Generating Python code..."
	@mkdir -p $(GEN_DIR)/python/video_rss
	python3 -m grpc_tools.protoc -I$(PROTO_DIR) \
		--python_out=$(GEN_DIR)/python/video_rss \
		--grpc_python_out=$(GEN_DIR)/python/video_rss \
		$(PROTO_DIR)/*.proto
	@touch $(GEN_DIR)/python/video_rss/__init__.py

clean:
	rm -rf $(GEN_DIR)
