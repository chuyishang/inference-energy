# Progress Log

## Current status
- README reformatted with structured protocol and toolkit usage.
- Implemented GPU power logger, vLLM load generator, and CLI entrypoint; packaging exposes `inference-energy`.
- `python3 -m compileall` passes for current codebase.
- Added analysis utilities to integrate power logs, subtract idle, and attribute energy to tokens; CLI now exposes `analyze`.
- Documented CLI workflow in README, including analyze command.

## Next steps
- Add a worked example once sample logs are available.
- If possible, validate the CLI end-to-end locally once `python` is available.
- Provide guidance for single-GPU runs (use `CUDA_VISIBLE_DEVICES` with vLLM).
