# PrimeRL gRPC API

## Summary
The `PrimeRL` service exposes three RPCs to manage stateful decode sessions.

### StartEpisode
- **Request**: `StartReq`
  - `env_id`: environment identifier (string)
  - `model`: model alias (string)
  - `prompt_fp`: optional fingerprint bytes (blake2b digest)
  - `prompt`: raw prompt text (string)
  - `pin_prefill`: whether to retain prefill cache on server (bool)
- **Response**: `StartResp`
  - `session_id`: UUID for episode
  - `cache_hit`: prefix cache hit flag

### Step (bidirectional streaming)
- **Request stream**: `StepReq`
  - `session_id`: active session id
  - `obs`: observation text
  - `max_new_tokens`: decode budget for this call
  - `grammar_id`: optional grammar spec id
  - `speculative`: enable speculation if supported
- **Response stream**: `StepResp`
  - `token`: generated token text
  - `t_us`: microsecond timestamp since decode start
  - `kv_bytes`: current KV residency
  - `boundary`: true when grammar/tool boundary reached

### EndEpisode
- **Request**: `EndReq`
  - `session_id`: session to close
- **Response**: `EndResp`
  - `evicted`: true if session KV cache was removed

Run `make gen-proto` to regenerate Python stubs after updating `api/primerl.proto`.
