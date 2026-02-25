//! Bridge between TieredRouter distillation events and SONA learning.
//!
//! `SonaDistillationSink` implements the `DistillationSink` trait so the
//! tiered router's escalation events feed directly into SONA's trajectory
//! recording, closing the first link in the self-improvement loop:
//!
//! ```text
//! TieredRouter (escalation) → DistillationEvent → SonaDistillationSink
//!     → Trajectory → SonaIntegration.record_trajectory()
//!         → Instant loop (MicroLoRA)
//!         → Background loop (EWC++, BaseLoRA)
//!         → Deep loop (PatternBank consolidation)
//! ```

use crate::InferenceTier;
use crate::claude_flow::tiered_router::{DistillationEvent, DistillationSink};
use crate::sona::integration::{SonaIntegration, Trajectory};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Maps `InferenceTier` to the SONA model index convention:
///   0 = Local (tiny/free), 1 = Ollama (small/free), 2 = CloudClaude (large/paid)
fn tier_to_model_index(tier: InferenceTier) -> usize {
    match tier {
        InferenceTier::Local => 0,
        InferenceTier::Ollama => 1,
        InferenceTier::CloudClaude => 2,
    }
}

/// Generates a deterministic pseudo-embedding from text.
///
/// This is a simple hash-based embedding for bridging purposes.
/// In production, the real embedding model would be used, but for
/// the learning loop to work we just need consistent vectors that
/// map similar prompts to nearby regions.
pub fn text_to_pseudo_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut embedding = vec![0.0f32; dim];
    let bytes = text.as_bytes();

    // Fill with deterministic values derived from text bytes
    for (i, &b) in bytes.iter().enumerate() {
        let idx = i % dim;
        // Use a simple hash-like mixing to spread values
        let val = ((b as f32) / 255.0) * 2.0 - 1.0; // normalize to [-1, 1]
        embedding[idx] += val;
    }

    // Normalize to unit length
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for x in &mut embedding {
            *x /= magnitude;
        }
    }

    embedding
}

/// A `DistillationSink` that forwards escalation events to SONA for learning.
///
/// When the tiered router escalates a request (e.g., Local → Ollama → Claude),
/// this sink converts the `DistillationEvent` into a SONA `Trajectory` and
/// records it. This triggers SONA's instant learning loop, which updates
/// MicroLoRA weights so the local model can handle similar requests next time.
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::sona::{SonaIntegration, SonaConfig};
/// use ruvllm::sona::distillation_bridge::SonaDistillationSink;
/// use ruvllm::claude_flow::tiered_router::TieredRouter;
///
/// let sona = Arc::new(SonaIntegration::new(SonaConfig::default()));
/// let sink = Arc::new(SonaDistillationSink::new(sona.clone()));
///
/// let mut router = TieredRouter::with_defaults();
/// router.set_distillation_sink(sink.clone());
/// ```
pub struct SonaDistillationSink {
    sona: Arc<SonaIntegration>,
    /// Embedding dimension for generated pseudo-embeddings
    embedding_dim: usize,
    /// Count of events successfully forwarded
    forwarded_count: AtomicU64,
    /// Count of events that failed to forward
    error_count: AtomicU64,
}

impl SonaDistillationSink {
    /// Create a new sink that forwards events to the given SONA integration.
    pub fn new(sona: Arc<SonaIntegration>) -> Self {
        Self {
            sona,
            embedding_dim: 768, // Match SonaConfig default
            forwarded_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
        }
    }

    /// Create with a custom embedding dimension.
    pub fn with_embedding_dim(sona: Arc<SonaIntegration>, embedding_dim: usize) -> Self {
        Self {
            sona,
            embedding_dim,
            forwarded_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
        }
    }

    /// Number of events successfully forwarded to SONA.
    pub fn forwarded_count(&self) -> u64 {
        self.forwarded_count.load(Ordering::SeqCst)
    }

    /// Number of events that failed to forward.
    pub fn error_count(&self) -> u64 {
        self.error_count.load(Ordering::SeqCst)
    }

    /// Convert a DistillationEvent into a SONA Trajectory.
    fn event_to_trajectory(&self, event: &DistillationEvent) -> Trajectory {
        let query_embedding = text_to_pseudo_embedding(&event.prompt, self.embedding_dim);
        let response_embedding = text_to_pseudo_embedding(&event.response, self.embedding_dim);

        // Build routing features vector from the event metadata
        let routing_features = vec![
            event.complexity,
            event.quality_estimate,
            tier_to_model_index(event.recommended_tier) as f32 / 2.0, // normalized 0-1
            tier_to_model_index(event.actual_tier) as f32 / 2.0,
            if event.was_fallback { 1.0 } else { 0.0 },
            event.input_tokens as f32 / 10000.0, // normalized
            event.output_tokens as f32 / 10000.0,
            event.latency_ms as f32 / 10000.0,
        ];

        Trajectory {
            request_id: format!("distill-{}", self.forwarded_count.load(Ordering::SeqCst)),
            session_id: "tiered-router".to_string(),
            query_embedding,
            response_embedding,
            quality_score: event.quality_estimate,
            routing_features,
            model_index: tier_to_model_index(event.actual_tier),
            timestamp: chrono::Utc::now(),
        }
    }
}

#[async_trait::async_trait]
impl DistillationSink for SonaDistillationSink {
    async fn on_distillation(&self, event: DistillationEvent) {
        let trajectory = self.event_to_trajectory(&event);

        match self.sona.record_trajectory(trajectory) {
            Ok(()) => {
                self.forwarded_count.fetch_add(1, Ordering::SeqCst);
                tracing::debug!(
                    prompt_len = event.prompt.len(),
                    actual_tier = ?event.actual_tier,
                    quality = event.quality_estimate,
                    "Distillation event forwarded to SONA"
                );
            }
            Err(e) => {
                self.error_count.fetch_add(1, Ordering::SeqCst);
                tracing::warn!(
                    error = %e,
                    "Failed to forward distillation event to SONA"
                );
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sona::SonaConfig;

    #[test]
    fn test_tier_to_model_index() {
        assert_eq!(tier_to_model_index(InferenceTier::Local), 0);
        assert_eq!(tier_to_model_index(InferenceTier::Ollama), 1);
        assert_eq!(tier_to_model_index(InferenceTier::CloudClaude), 2);
    }

    #[test]
    fn test_pseudo_embedding_deterministic() {
        let e1 = text_to_pseudo_embedding("hello world", 64);
        let e2 = text_to_pseudo_embedding("hello world", 64);
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_pseudo_embedding_normalized() {
        let e = text_to_pseudo_embedding("test input for embedding", 64);
        let magnitude: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.01, "expected unit length, got {}", magnitude);
    }

    #[test]
    fn test_pseudo_embedding_different_texts_differ() {
        let e1 = text_to_pseudo_embedding("fix typo", 64);
        let e2 = text_to_pseudo_embedding("design distributed system", 64);
        assert_ne!(e1, e2);
    }

    #[test]
    fn test_event_to_trajectory() {
        let sona = Arc::new(SonaIntegration::new(SonaConfig::default()));
        let sink = SonaDistillationSink::with_embedding_dim(sona, 64);

        let event = DistillationEvent {
            prompt: "fix typo in readme".to_string(),
            response: "Here is the fixed readme content.".to_string(),
            recommended_tier: InferenceTier::Local,
            actual_tier: InferenceTier::Ollama,
            was_fallback: true,
            complexity: 0.25,
            quality_estimate: 0.7,
            input_tokens: 50,
            output_tokens: 100,
            latency_ms: 200,
            timestamp: std::time::SystemTime::now(),
        };

        let trajectory = sink.event_to_trajectory(&event);

        assert_eq!(trajectory.session_id, "tiered-router");
        assert_eq!(trajectory.model_index, 1); // Ollama
        assert_eq!(trajectory.quality_score, 0.7);
        assert_eq!(trajectory.query_embedding.len(), 64);
        assert_eq!(trajectory.response_embedding.len(), 64);
        assert_eq!(trajectory.routing_features.len(), 8);
        // was_fallback should be 1.0
        assert_eq!(trajectory.routing_features[4], 1.0);
    }

    #[tokio::test]
    async fn test_sink_forwards_to_sona() {
        let sona = Arc::new(SonaIntegration::new(SonaConfig {
            embedding_dim: 64,
            hidden_dim: 64,
            quality_threshold: 0.0, // Accept all for testing
            ..SonaConfig::default()
        }));
        let sink = SonaDistillationSink::with_embedding_dim(sona.clone(), 64);

        let event = DistillationEvent {
            prompt: "fix typo".to_string(),
            response: "Fixed the typo.".to_string(),
            recommended_tier: InferenceTier::Local,
            actual_tier: InferenceTier::Ollama,
            was_fallback: true,
            complexity: 0.2,
            quality_estimate: 0.8,
            input_tokens: 20,
            output_tokens: 30,
            latency_ms: 150,
            timestamp: std::time::SystemTime::now(),
        };

        // Before
        assert_eq!(sink.forwarded_count(), 0);
        assert_eq!(sona.stats().total_trajectories, 0);

        // Fire the event
        sink.on_distillation(event).await;

        // After
        assert_eq!(sink.forwarded_count(), 1);
        assert_eq!(sink.error_count(), 0);
        assert_eq!(sona.stats().total_trajectories, 1);
    }

    #[tokio::test]
    async fn test_multiple_events_accumulate() {
        let sona = Arc::new(SonaIntegration::new(SonaConfig {
            embedding_dim: 64,
            hidden_dim: 64,
            quality_threshold: 0.0,
            ..SonaConfig::default()
        }));
        let sink = SonaDistillationSink::with_embedding_dim(sona.clone(), 64);

        for i in 0..5 {
            let event = DistillationEvent {
                prompt: format!("task number {}", i),
                response: format!("response for task {}", i),
                recommended_tier: InferenceTier::Local,
                actual_tier: InferenceTier::CloudClaude,
                was_fallback: true,
                complexity: 0.5,
                quality_estimate: 0.9,
                input_tokens: 100,
                output_tokens: 200,
                latency_ms: 1500,
                timestamp: std::time::SystemTime::now(),
            };
            sink.on_distillation(event).await;
        }

        assert_eq!(sink.forwarded_count(), 5);
        assert_eq!(sona.stats().total_trajectories, 5);
    }

    #[tokio::test]
    async fn test_sink_with_tiered_router() {
        use crate::claude_flow::tiered_router::{TieredRouter, TieredRouterConfig};
        use crate::backends::unified_backend::{
            UnifiedInferenceBackend, UnifiedRequest, UnifiedResponse, UnifiedStreamToken,
        };

        /// Mock backend for testing
        struct MockBackend {
            name: String,
        }

        #[async_trait::async_trait]
        impl UnifiedInferenceBackend for MockBackend {
            fn name(&self) -> &str {
                &self.name
            }
            async fn is_available(&self) -> bool {
                true
            }
            async fn generate(&self, request: &UnifiedRequest) -> crate::error::Result<UnifiedResponse> {
                Ok(UnifiedResponse {
                    text: format!("Response to: {}", request.prompt),
                    input_tokens: 10,
                    output_tokens: 20,
                    ttft_ms: 50,
                    total_ms: 100,
                    backend_name: self.name.clone(),
                })
            }
            async fn generate_stream(
                &self,
                _request: &UnifiedRequest,
            ) -> crate::error::Result<tokio::sync::mpsc::Receiver<crate::error::Result<UnifiedStreamToken>>> {
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                drop(tx);
                Ok(rx)
            }
        }

        let sona = Arc::new(SonaIntegration::new(SonaConfig {
            embedding_dim: 64,
            hidden_dim: 64,
            quality_threshold: 0.0,
            ..SonaConfig::default()
        }));
        let sink = Arc::new(SonaDistillationSink::with_embedding_dim(sona.clone(), 64));

        let config = TieredRouterConfig {
            distillation_quality_threshold: 0.0, // mock backends complete in ~0ms
            ..TieredRouterConfig::default()
        };
        let mut router = TieredRouter::new(config);
        router.set_distillation_sink(sink.clone());

        // Only register Ollama — simple tasks will escalate from Local → Ollama
        router.register(
            InferenceTier::Ollama,
            Arc::new(MockBackend {
                name: "test-ollama".to_string(),
            }),
        );

        // Route a simple task — should escalate and trigger distillation → SONA
        let response = router.route_and_generate("fix typo").await.unwrap();
        assert!(!response.text.is_empty());

        // Check router stats to see if distillation was attempted
        let stats = router.stats();

        // Verify the event made it through to SONA
        assert_eq!(
            stats.distillation_count, 1,
            "Router should have distilled 1 event, stats: {:?}",
            stats
        );
        assert_eq!(sink.forwarded_count(), 1, "Sink should have forwarded 1 event");
        assert_eq!(sona.stats().total_trajectories, 1);
    }
}
