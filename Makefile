.PHONY: dev gen-proto run-perf

dev:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip -r requirements.txt

gen-proto:
	python -m grpc_tools.protoc -I=api --python_out=. --grpc_python_out=. api/primerl.proto

run-perf:
	python perf/bench_matrix.py --out perf/results.csv
	python perf/perf_gate.py --baseline perf/baseline.csv --current perf/results.csv
