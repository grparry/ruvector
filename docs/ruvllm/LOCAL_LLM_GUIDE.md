# Local LLM Inference Guide

How to run local inference with RuvLLM, connect it to the tiered router, and
close the self-learning loop so the local tier improves over time.

## Quick start

### 1. Download a model

```bash
# List available models
cargo run -p ruvllm --example download_test_model -- --list

# Download a small model for testing (~400MB)
cargo run -p ruvllm --example download_test_model -- qwen-0.5b

# Download the purpose-built RuvLTRA Small (~662MB, includes SONA weights)
cargo run -p ruvllm --example download_test_model -- ruvltra-small
```

Available models:

| Model | Size | Params | Best for |
|-------|------|--------|----------|
| `qwen-0.5b` | ~400MB | 0.5B | Fastest tests, smallest footprint |
| `tinyllama` | ~669MB | 1.1B | General testing |
| `ruvltra-small` | ~662MB | 0.5B | Edge devices, includes SONA weights |
| `ruvltra-medium` | ~2.1GB | 3B | General purpose, 256K context |
| `phi-3-mini` | ~2.2GB | 3.8B | Higher quality outputs |
| `gemma-2b` | ~1.5GB | 2B | Good instruction following |

Models download as GGUF files to `./test_models/` by default. Override with
`--output <DIR>` or set `RUVLLM_MODELS_DIR`.

### 2. Load and generate (direct backend)

```rust
use ruvllm::{CandleBackend, ModelConfig, ModelArchitecture, GenerateParams, LlmBackend};

// Create backend (auto-selects Metal on macOS, CPU otherwise)
let mut backend = CandleBackend::new()?;

// Load a GGUF model from disk
backend.load_model("./test_models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", ModelConfig {
    architecture: ModelArchitecture::Llama,
    max_sequence_length: 2048,
    ..Default::default()
})?;

// Generate
let response = backend.generate("Explain quicksort in one paragraph.", GenerateParams {
    max_tokens: 256,
    temperature: 0.7,
    ..Default::default()
})?;
println!("{}", response);
```

### 3. Streaming generation

```rust
use ruvllm::{CandleBackend, GenerateParams, LlmBackend};

// After loading a model...
let stream = backend.generate_stream_v2("Write a haiku about Rust.", GenerateParams {
    max_tokens: 64,
    temperature: 0.9,
    ..Default::default()
})?;

for event in stream {
    match event {
        Ok(ruvllm::StreamEvent::Token(tok)) => print!("{}", tok.text),
        Ok(ruvllm::StreamEvent::Done(_)) => break,
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Cargo features

| Feature | What it enables | Dependencies added |
|---------|-----------------|-------------------|
| `candle` (default) | CandleBackend, GGUF loading, HuggingFace Hub | candle-core/nn/transformers, tokenizers, hf-hub |
| `metal` | Metal GPU acceleration on Apple Silicon | candle metal backends |
| `cuda` | CUDA GPU acceleration on NVIDIA | candle CUDA backends |
| `ollama` | OllamaBackend, UnifiedInferenceBackend, TieredRouter | reqwest |
| `parallel` | Multi-threaded GEMM/GEMV via rayon | rayon |
| `mmap` | Memory-mapped GGUF loading | memmap2 |
| `coreml` | Apple Neural Engine via Core ML | objc2, objc2-core-ml |
| `openai-compat` | OpenAI/Ollama compatible HTTP server | axum |

Recommended feature sets:

```toml
# Mac with Apple Silicon (best performance)
ruvllm = { version = "2.0", features = ["inference-metal", "ollama", "parallel", "mmap"] }

# NVIDIA GPU
ruvllm = { version = "2.0", features = ["inference-cuda", "ollama", "parallel", "mmap"] }

# CPU only (Linux server, CI)
ruvllm = { version = "2.0", features = ["candle", "ollama", "parallel"] }

# Edge / embedded (minimal deps)
ruvllm = { version = "2.0", features = ["minimal"] }
```

## Device selection

```rust
use ruvllm::{CandleBackend, DeviceType};

// Explicit device selection
let backend = CandleBackend::with_device(DeviceType::Metal)?;   // Apple Silicon GPU
let backend = CandleBackend::with_device(DeviceType::Cuda(0))?; // NVIDIA GPU 0
let backend = CandleBackend::with_device(DeviceType::Cpu)?;     // CPU fallback

// Auto-select (Metal on macOS, CPU otherwise)
let backend = CandleBackend::new()?;
```

## Generation parameters

```rust
use ruvllm::GenerateParams;

let params = GenerateParams {
    max_tokens: 512,           // Max output length
    temperature: 0.7,          // 0.0 = deterministic, 1.0+ = creative
    top_p: 0.9,                // Nucleus sampling
    top_k: 50,                 // Top-k sampling (0 = disabled)
    repetition_penalty: 1.1,   // Penalize repeated tokens
    frequency_penalty: 0.0,    // Penalize frequent tokens
    presence_penalty: 0.0,     // Penalize tokens already present
    stop_sequences: vec![],    // Stop generation at these strings
    seed: Some(42),            // Reproducibility
};
```

## Model configuration

```rust
use ruvllm::{ModelConfig, ModelArchitecture, Quantization};

let config = ModelConfig {
    architecture: ModelArchitecture::Llama,  // Llama, Qwen2, Phi3, Gemma, etc.
    quantization: Some(Quantization::Q4_K),  // Matches GGUF quantization format
    use_flash_attention: true,               // Memory-efficient attention
    max_sequence_length: 4096,               // Context window
    num_kv_heads: None,                      // Auto-detect from model
    hidden_size: None,                       // Auto-detect from model
    num_layers: None,                        // Auto-detect from model
    ..Default::default()
};
```

## Ollama backend

Requires a running Ollama server (`ollama serve` or `systemctl start ollama`).

```rust
use ruvllm::backends::{OllamaBackend, OllamaConfig};
use std::time::Duration;

let config = OllamaConfig {
    base_url: "http://localhost:11434".to_string(),
    model: "llama3.1:8b".to_string(),
    timeout: Duration::from_secs(120),
    connect_timeout: Duration::from_secs(5),
    keep_alive: Some(Duration::from_secs(300)),
};

let backend = OllamaBackend::new(config);

// Health check
let healthy = backend.health_check().await?;

// Generate (uses configured default model)
let response = backend.generate("llama3.1:8b", "Hello!", None).await?;
println!("{}", response.response);

// Streaming
let mut rx = backend.generate_stream("llama3.1:8b", "Tell me a story", None).await?;
while let Some(chunk) = rx.recv().await {
    print!("{}", chunk.text);
}
```

## Tiered router

Combines local, Ollama, and Claude backends behind a single interface that
routes based on task complexity.

```rust
use ruvllm::claude_flow::tiered_router::{TieredRouter, TieredRouterConfig};
use ruvllm::claude_flow::model_router::InferenceTier;
use ruvllm::backends::unified_backend::{
    UnifiedInferenceBackend, UnifiedRequest, LocalCandleAdapter, OllamaAdapter, ClaudeAdapter,
};
use ruvllm::claude_flow::ClaudeModel;
use std::sync::Arc;

// Configure routing thresholds
let config = TieredRouterConfig {
    local_threshold: 0.35,     // Below this → Local
    claude_threshold: 0.70,    // Above this → Claude
    local_token_limit: 500,    // Token budget for Local tier
    ollama_token_limit: 2000,  // Token budget for Ollama tier
    enable_fallback: true,
    enable_distillation: true,
    distillation_quality_threshold: 0.5,
    ..Default::default()
};

let mut router = TieredRouter::new(config);

// Register backends (any subset — you don't need all three)
let mut candle = CandleBackend::new()?;
candle.load_model("./test_models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", ModelConfig {
    architecture: ModelArchitecture::Llama,
    ..Default::default()
})?;
router.register(InferenceTier::Local, Arc::new(LocalCandleAdapter::new(candle)));
router.register(InferenceTier::Ollama, Arc::new(OllamaAdapter::with_defaults()?));
router.register(InferenceTier::CloudClaude, Arc::new(
    ClaudeAdapter::from_env(ClaudeModel::Sonnet)?
));

// Route automatically — complexity analysis picks the tier
let response = router.route_and_generate("fix typo in README").await?;
// → Local tier (simple task, score < 0.35)

let response = router.route_and_generate(
    "Design a distributed authentication system with OAuth2, JWT, security audit"
).await?;
// → Claude tier (complex task, score > 0.70)

// Check what happened
let stats = router.stats();
println!("Requests by tier: {:?}", stats.requests_by_tier);
println!("Fallbacks: {}", stats.fallback_count);
println!("Distillation events: {}", stats.distillation_count);
```

## Closing the self-learning loop

The self-learning router doc (`SELF_LEARNING_ROUTER.md`) identifies two gaps
that prevent the local tier from actually improving:

> 1. Uses hash-based pseudo-embeddings instead of real embeddings
> 2. Needs a real local model behind the MicroLoRA

Both gaps are addressed by wiring CandleBackend into the distillation pipeline.

### The problem

`SonaDistillationSink` currently uses `text_to_pseudo_embedding()` — a
deterministic hash that maps text to vectors. These vectors are consistent
(same input → same output) but don't capture semantic meaning, so the
ReasoningBank's K-means++ clustering groups by surface form rather than intent.

Meanwhile, the learning bridge (`SonaLoraBridge`) applies pattern gradients to
a `MicroLoRA`, but without a real model behind it, the weight updates have
nowhere to take effect during inference.

### The fix

**Real embeddings**: CandleBackend exposes `get_embeddings()` via the
`LlmBackend` trait. Once a model is loaded, the forward pass through the
model's encoder produces real hidden-state embeddings that capture semantic
meaning. Replace `text_to_pseudo_embedding` calls in `SonaDistillationSink`
with `backend.get_embeddings(text)`:

```rust
// Before (hash-based, no semantic signal):
let embedding = text_to_pseudo_embedding(&event.prompt, 768);

// After (model-derived, semantically meaningful):
let embedding = backend.get_embeddings(&event.prompt)?;
```

This requires:
- A loaded CandleBackend accessible from the distillation sink
- The `get_embeddings()` implementation to extract actual hidden states
  (currently returns a zero vector placeholder — needs the final hidden state
  mean-pooled across tokens)

**Real local model**: The `LocalCandleAdapter` in the tiered router already
wraps a real CandleBackend with a loaded model. When the tiered router's
Local tier serves a request, it runs through the model's full forward pass.
The `MicroLoRA` adapter sits in this forward pass path — when
`SonaLoraBridge` applies weight updates, they take effect on the next
Local-tier inference.

The wiring looks like this:

```
                     ┌──────────────────────────┐
                     │      TieredRouter         │
                     │                           │
  request ──────────►│  analyze complexity       │
                     │       │                   │
                     │  ┌────▼────┐              │
                     │  │  Local  │──► CandleBackend + MicroLoRA ──► response
                     │  │  tier   │         ▲                           │
                     │  └─────────┘         │                           │
                     │  ┌─────────┐     weight updates                 │
                     │  │ Ollama  │         │                           │
                     │  │  tier   │    SonaLoraBridge                   │
                     │  └─────────┘         ▲                           │
                     │  ┌─────────┐     ReasoningBank patterns         │
                     │  │ Claude  │         ▲                           │
                     │  │  tier   │    SonaIntegration                  │
                     │  └─────────┘         ▲                           │
                     │                      │                           │
                     │              SonaDistillationSink                │
                     │                      ▲                           │
                     │                      │                           │
                     │              DistillationEvent ◄─────────────────┘
                     │         (prompt + response from higher tier)     │
                     └──────────────────────────────────────────────────┘
```

### Steps to complete

1. **Implement real `get_embeddings()`**: In `CandleBackend`, extract the
   final hidden state from the model's forward pass and mean-pool across
   tokens. This gives a real semantic embedding for any input text.

2. **Inject backend into distillation sink**: Give `SonaDistillationSink` an
   `Arc<Mutex<CandleBackend>>` (or the `LlmBackend` trait object) so it can
   call `get_embeddings()` instead of `text_to_pseudo_embedding()`.

3. **Attach MicroLoRA to CandleBackend**: Ensure the `MicroLoRA` adapter
   used by `SonaLoraBridge` is the same instance that sits in the
   CandleBackend's forward pass, so weight updates from pattern application
   directly affect Local-tier inference quality.

4. **Persist learned state**: Save MicroLoRA weights and ReasoningBank
   patterns to disk so learning survives restarts. `MicroLoRA` is < 1MB and
   serializable; the ReasoningBank patterns are already Ruvector-backed.

5. **Add evaluation**: Compare Local-tier output quality before and after
   learning rounds to confirm the loop is actually improving responses,
   not just accumulating noise.

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `RUVLLM_MODELS_DIR` | Default model download directory | `./test_models` |
| `HF_TOKEN` | HuggingFace token for gated model downloads | None |
| `ANTHROPIC_API_KEY` | Claude API key (for Claude tier) | None |
| `OLLAMA_HOST` | Ollama server address | `http://localhost:11434` |

## Running tests with a real model

```bash
# Download a model first
cargo run -p ruvllm --example download_test_model -- tinyllama

# Run the real model integration tests
TEST_MODEL_PATH=./test_models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  cargo test -p ruvllm --test real_model_test -- --ignored
```
