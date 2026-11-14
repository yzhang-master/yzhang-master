# PUSS

AI systems engineer working on model training, inference optimization, and agent infrastructure.

This profile organizes my portfolio by the engineering stages the projects represent. The implementations use current libraries and honest CPU verification; GPU benchmark tables remain pending until they are measured on real hardware.

## Engineering timeline

| Stage | Role | Focus |
|---|---|---|
| 2022/02 | Senior ML Engineer | Efficient transformer training and deployment |
| 2023/01 | Senior AI Engineer | LLM fine-tuning, serving, evaluation, and agents |
| 2025/01 | Staff AI Engineer | Frontier training, kernels, serving, and agent infrastructure |

## Selected work

### Efficient training and deployment

- [μP coordinate checks and learning-rate transfer](https://github.com/yzhang-master/mup-coordinate-check-lr-transfer)
- [MinHash-LSH near-duplicate removal](https://github.com/yzhang-master/minhash-lsh-near-dedup)
- [DDP, AMP, and activation checkpointing](https://github.com/yzhang-master/ddp-amp-activation-checkpointing-memory-bench)
- [INT8 outlier-decomposition matmul](https://github.com/yzhang-master/llm-int8-outlier-decomposition-matmul)
- [FlashAttention forward and backward in Triton](https://github.com/yzhang-master/flashattention-triton-fwd-bwd)
- [BERT ONNX, TensorRT, and dynamic batching](https://github.com/yzhang-master/bert-onnx-tensorrt-dynamic-batching)
- [Late-interaction retrieval with product quantization](https://github.com/yzhang-master/late-interaction-retrieval-pq)
- [Hard-negative InfoNCE embedding fine-tuning](https://github.com/yzhang-master/hard-negative-infonce-embedding-ft)
- [BM25, dense retrieval, RRF, and cross-encoder reranking](https://github.com/yzhang-master/bm25-dense-rrf-crossencoder)
- [NCCL all-reduce and topology benchmarks](https://github.com/yzhang-master/nccl-allreduce-topology-bench)

### LLM training and evaluation

- [ZeRO-3 vs FSDP2 memory timelines](https://github.com/yzhang-master/zero3-vs-fsdp2-memory-timeline)
- [NF4 QLoRA and paged AdamW from scratch](https://github.com/yzhang-master/qlora-nf4-double-quant-from-scratch)
- [Unified DPO, IPO, and KTO trainer](https://github.com/yzhang-master/dpo-ipo-kto-unified-trainer)
- [YaRN and NTK-aware RoPE context extension](https://github.com/yzhang-master/yarn-rope-128k-context-extension)
- [Ring attention for context parallelism](https://github.com/yzhang-master/ring-attention-context-parallel)
- [HF, Megatron, and torch DCP checkpoint resharding](https://github.com/yzhang-master/hf-megatron-dcp-checkpoint-reshard)
- [LLM judge bias measurement and calibration](https://github.com/yzhang-master/llm-judge-position-bias-calibration)
- [Benchmark contamination detection with n-grams and Min-K%](https://github.com/yzhang-master/benchmark-contamination-ngram-minkprob)

### Inference and serving

- [AWQ vs GPTQ INT4 with a Marlin GPU path](https://github.com/yzhang-master/awq-vs-gptq-int4-marlin-bench)
- [Radix-tree prefix caching with LRU eviction](https://github.com/yzhang-master/radix-tree-prefix-cache-lru)
- [Continuous batching with chunked prefill](https://github.com/yzhang-master/continuous-batching-chunked-prefill)
- [EAGLE draft-head training and speculative decoding](https://github.com/yzhang-master/eagle-draft-head-training)
- [Paged Triton decode attention](https://github.com/yzhang-master/paged-attention-triton-decode)
- [FP8 and INT4 KV-cache quantization](https://github.com/yzhang-master/kv-cache-fp8-int4-quant)
- [JSON Schema FSM and token masks](https://github.com/yzhang-master/json-schema-fsm-token-mask)

### Agent systems

- [Typed-tool ReAct agent with repair retries](https://github.com/yzhang-master/react-agent-typed-tool-schemas)
- [Docker-validated test-repair agent](https://github.com/yzhang-master/swe-test-repair-agent-docker)
- [Planner, parallel executors, and verifying critic](https://github.com/yzhang-master/planner-executor-critic-dag)
- [Browser accessibility-tree and set-of-marks grounding](https://github.com/yzhang-master/browser-agent-a11y-tree-grounding)
- [Episodic memory consolidation](https://github.com/yzhang-master/episodic-memory-consolidation-agent)
- [Function-calling SFT and 3B training path](https://github.com/yzhang-master/function-calling-sft-3b)
- [Indirect prompt-injection detection and capability enforcement](https://github.com/yzhang-master/indirect-prompt-injection-detector)
- [Read-only MCP SQL server](https://github.com/yzhang-master/mcp-sql-server-readonly-guard)
- [Agent trajectory and pass^k reliability evaluation](https://github.com/yzhang-master/agent-trajectory-eval-pass-k)
- [Docker and Firecracker code sandbox](https://github.com/yzhang-master/firecracker-microvm-code-sandbox)
- [OpenTelemetry GenAI tracing and deterministic replay](https://github.com/yzhang-master/otel-genai-agent-tracing)
- [Streaming voice-agent pipeline](https://github.com/yzhang-master/streaming-voice-agent-sub500ms)
- [Prompt-cache-aware context compaction](https://github.com/yzhang-master/prompt-cache-aware-context-compaction)

### Frontier training and inference

- [FSDP2 FP8 Llama pretraining](https://github.com/yzhang-master/fsdp2-fp8-pretrain-llama-1b)
- [GRPO with verifiable math rewards](https://github.com/yzhang-master/grpo-gsm8k-verifiable-rewards)
- [MoE expert parallelism over all-to-all](https://github.com/yzhang-master/moe-expert-parallel-all2all)
- [Monte Carlo process reward modeling](https://github.com/yzhang-master/prm-monte-carlo-step-labels)
- [Blockwise FP8 GEMM for Hopper](https://github.com/yzhang-master/fp8-blockwise-gemm-cutlass-hopper)
- [NVFP4 weight quantization for Blackwell](https://github.com/yzhang-master/nvfp4-weight-quant-blackwell)
- [H100/B200 decode roofline analysis](https://github.com/yzhang-master/llm-decode-roofline-h100-b200)
- [MLA absorbed-weight decode kernel](https://github.com/yzhang-master/mla-absorbed-weights-decode-kernel)
- [Fused top-k and grouped MoE GEMM](https://github.com/yzhang-master/moe-fused-topk-grouped-gemm)
- [TP allreduce/compute overlap](https://github.com/yzhang-master/tp-allreduce-compute-overlap)
- [Prefill/decode disaggregation with KV transfer](https://github.com/yzhang-master/prefill-decode-disagg-rdma-kv)

### Integrated systems

- [FSDP2, TP, CP, and FP8 pretraining stack](https://github.com/yzhang-master/fsdp2-tp-cp-fp8-pretraining-stack)
- [Verifier, pass@k, judge-bias, and contamination evaluation harness](https://github.com/yzhang-master/llm-eval-harness-judge-contamination)

### Kernel optimization

- [Fused Triton RMSNorm, RoPE, and SwiGLU](https://github.com/yzhang-master/triton-fused-rmsnorm-rope-swiglu)

## Verification

Each implementation repository includes CPU tests, reproduction commands, and a discussion of limitations. Performance numbers are published only when measured; unmeasured GPU results are explicitly marked pending.
