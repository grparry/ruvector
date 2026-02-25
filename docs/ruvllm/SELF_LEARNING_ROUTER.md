# Self-Learning Inference Router

## What it is

A tiered LLM proxy that learns from its own routing decisions. When a request
escalates from a cheaper backend to a more capable one, the system captures that
interaction as a training signal. Over time, the router's internal
representations adapt so it makes better decisions and the local tier becomes
more capable for previously-escalated patterns.

## How it works

```
User request
  │
  ▼
TieredRouter ──► analyze complexity
  │
  ├─ Local tier (fast, free)
  │    └─ if unavailable or insufficient → fallback ↓
  ├─ Ollama tier (small model, free)
  │    └─ if unavailable or insufficient → fallback ↓
  └─ Claude tier (large model, paid)
         │
         ▼
    Response returned to caller
         │
         ▼
    DistillationEvent captured
         │
         ▼
    SonaDistillationSink → Trajectory
         │
         ▼
    SONA Learning Engine
    ├─ Instant loop: MicroLoRA gradient accumulation
    ├─ Background loop: pattern extraction via ReasoningBank
    └─ Deep loop: long-term consolidation (weekly)
         │
         ▼
    SonaLoraBridge
    ├─ Path A: ReasoningBank centroids → MicroLoRA weight updates
    └─ Path B: SONA engine transform → MicroLoRA weight updates
         │
         ▼
    Next request's forward pass is different
```

## The learning loop

1. **Route**: A request arrives. The router estimates complexity and picks a
   starting tier. If the chosen tier is unavailable or fails, it escalates.

2. **Distill**: When a response comes from a higher tier than initially
   recommended, a `DistillationEvent` fires containing the prompt, response,
   quality estimate, and routing metadata.

3. **Record**: The `SonaDistillationSink` converts the event into a SONA
   `Trajectory` with pseudo-embeddings derived from the prompt and response
   text, then records it in the SONA integration layer.

4. **Learn (instant)**: On every trajectory, the SONA engine's instant loop
   generates a `LearningSignal` and accumulates gradients in its internal
   MicroLoRA. For single-step trajectories, the query embedding serves as the
   gradient direction scaled by quality. For multi-step trajectories, REINFORCE
   with baseline provides the gradient estimate.

5. **Learn (background)**: Periodically (configurable, default hourly), the
   background loop clusters accumulated trajectories in the ReasoningBank using
   K-means++ and extracts learned patterns — centroids that represent "average
   shapes" of successful responses for each cluster.

6. **Apply**: The `SonaLoraBridge` connects SONA's learned state to the ruvllm
   `MicroLoRA`:
   - **Path A**: Queries the ReasoningBank for patterns similar to the current
     input, converts each centroid into an adaptation gradient, and applies
     weight updates.
   - **Path B**: Runs the input through the SONA engine's MicroLoRA transform
     and feeds the result as a gradient estimate to the ruvllm MicroLoRA.

7. **Repeat**: The next request's forward pass runs through updated weights.
   Over time, the local tier's representations shift toward patterns that
   previously required escalation.

## Key components

| Component | Crate | Role |
|---|---|---|
| `TieredRouter` | ruvllm | Routes requests across Local/Ollama/Claude tiers with fallback |
| `SonaDistillationSink` | ruvllm | Converts escalation events into SONA trajectories |
| `SonaIntegration` | ruvllm | Manages trajectory buffer, EWC++, ReasoningBank, and SONA engine |
| `SonaLoraBridge` | ruvllm | Applies learned patterns and transforms to MicroLoRA weights |
| `MicroLoRA` | ruvllm | Ultra-lightweight rank 1-2 LoRA adapters (<1MB, <1ms forward) |
| `SonaEngine` | sona | Three-loop learning: instant, background, deep |
| `ReasoningBank` | sona | K-means++ clustering and pattern storage |
| `LearningSignal` | sona | REINFORCE gradient estimation from trajectories |

## Cold-start configuration

By default, SONA requires 100+ trajectories before learning activates. For
development, testing, or low-traffic deployments, these thresholds are
configurable:

```rust
SonaConfig {
    instant_flush_threshold: 1,        // Flush after every trajectory (default: 100)
    min_background_trajectories: 1,    // Allow background loop with any count (default: 100)
    min_cluster_size: 1,               // Allow single-member clusters (default: 5)
    background_interval_secs: 0,       // No timer delay (default: 3600)
    ..SonaConfig::default()
}
```

Production defaults are unchanged and require meaningful data volume before
learning activates, which prevents noise from corrupting weights.

## What it does not do

- **Generate text**: It routes to backends that generate text. The LoRA adapts
  routing representations, not a language model's decoder weights.
- **Replace human review**: Weight updates apply automatically within the LoRA
  adapter. There is no approval gate for individual weight changes.
- **Use real embeddings**: Currently uses deterministic hash-based
  pseudo-embeddings. A real embedding model would improve pattern quality.
- **Learn to serve locally**: Making the local tier actually handle requests it
  previously couldn't requires a real local model behind the MicroLoRA, not just
  routing adaptation.

## What it needs to become production-ready

1. **Real embedding model** for trajectory recording and pattern matching
2. **Real local model** behind the MicroLoRA so weight shifts change generation
3. **Multi-step trajectories** from chain-of-thought or multi-turn conversations
   to unlock full REINFORCE gradient signal
4. **Evaluation harness** to confirm weight shifts improve output quality
5. **Persistence** for learned patterns and LoRA weights across restarts
6. **Monitoring** for weight magnitude, pattern quality, and routing accuracy
   over time
