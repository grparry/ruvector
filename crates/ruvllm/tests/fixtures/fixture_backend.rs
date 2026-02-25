//! Fixture-based backend for E2E testing of the self-improvement loop.
//!
//! Provides a `FixtureBackend` that implements `UnifiedInferenceBackend` using
//! pre-defined prompt/response pairs from `self_improvement_fixtures.json`.
//! No real models needed — responses come from fixture data with realistic
//! token counts and latency simulation.

use ruvllm::backends::unified_backend::{
    UnifiedInferenceBackend, UnifiedRequest, UnifiedResponse, UnifiedStreamToken,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// Fixture data types
// ============================================================================

/// A single fixture task loaded from JSON
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct FixtureTask {
    pub id: String,
    pub category: String,
    pub prompt: String,
    pub expected_tier: String,
    pub high_quality_response: String,
    pub degraded_response: String,
    pub key_phrases: Vec<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_max_tokens() -> usize {
    200
}

/// Root fixture file structure
#[derive(Debug, Clone, Deserialize)]
pub struct FixtureFile {
    pub tasks: Vec<FixtureTask>,
}

/// Controls what kind of response the backend returns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureMode {
    /// Returns high_quality_response (simulates Ollama/Claude)
    HighQuality,
    /// Returns degraded_response (simulates weak Local)
    Degraded,
    /// is_available() returns false (forces fallback)
    Unavailable,
}

// ============================================================================
// FixtureBackend
// ============================================================================

/// A test backend that returns fixture responses without any real model.
///
/// Matches incoming prompts against fixture data via exact match first,
/// then Jaccard similarity fallback.
pub struct FixtureBackend {
    name: String,
    mode: FixtureMode,
    /// Exact-match lookup: prompt text → fixture task
    lookup: HashMap<String, FixtureTask>,
    /// All fixture tasks (for Jaccard fallback)
    all_tasks: Vec<FixtureTask>,
    /// Count of generate() calls
    generate_count: AtomicUsize,
}

impl FixtureBackend {
    /// Create a new fixture backend with the given mode.
    pub fn new(name: &str, mode: FixtureMode, tasks: &[FixtureTask]) -> Self {
        let mut lookup = HashMap::new();
        for task in tasks {
            lookup.insert(task.prompt.clone(), task.clone());
        }
        Self {
            name: name.to_string(),
            mode,
            lookup,
            all_tasks: tasks.to_vec(),
            generate_count: AtomicUsize::new(0),
        }
    }

    /// Load fixtures from the embedded JSON file.
    pub fn load_fixtures() -> Vec<FixtureTask> {
        let json = include_str!("self_improvement_fixtures.json");
        let file: FixtureFile = serde_json::from_str(json)
            .expect("Failed to parse self_improvement_fixtures.json");
        file.tasks
    }

    /// How many times generate() was called.
    pub fn generate_count(&self) -> usize {
        self.generate_count.load(Ordering::SeqCst)
    }

    /// Find the best matching fixture task for a prompt.
    fn find_task(&self, prompt: &str) -> Option<&FixtureTask> {
        // 1. Exact match
        if let Some(task) = self.lookup.get(prompt) {
            return Some(task);
        }

        // 2. Jaccard similarity fallback
        let prompt_words = word_set(prompt);
        let mut best_score = 0.0f32;
        let mut best_task = None;

        for task in &self.all_tasks {
            let task_words = word_set(&task.prompt);
            let score = jaccard_similarity(&prompt_words, &task_words);
            if score > best_score {
                best_score = score;
                best_task = Some(task);
            }
        }

        // Require at least 30% overlap
        if best_score >= 0.3 {
            best_task
        } else {
            None
        }
    }

    /// Get response text for a matched task
    fn response_text(&self, task: &FixtureTask) -> String {
        match self.mode {
            FixtureMode::HighQuality => task.high_quality_response.clone(),
            FixtureMode::Degraded => task.degraded_response.clone(),
            FixtureMode::Unavailable => unreachable!("Unavailable backend should not generate"),
        }
    }
}

#[async_trait::async_trait]
impl UnifiedInferenceBackend for FixtureBackend {
    fn name(&self) -> &str {
        &self.name
    }

    async fn is_available(&self) -> bool {
        self.mode != FixtureMode::Unavailable
    }

    async fn generate(&self, request: &UnifiedRequest) -> ruvllm::error::Result<UnifiedResponse> {
        self.generate_count.fetch_add(1, Ordering::SeqCst);

        let response_text = if let Some(task) = self.find_task(&request.prompt) {
            self.response_text(task)
        } else {
            // Fallback: generic response for unmatched prompts
            match self.mode {
                FixtureMode::HighQuality => {
                    format!("High quality response to: {}", &request.prompt[..request.prompt.len().min(80)])
                }
                FixtureMode::Degraded => "ok".to_string(),
                FixtureMode::Unavailable => unreachable!(),
            }
        };

        // Realistic token counts (response.len() / 4) and latency
        let output_tokens = response_text.len() / 4;
        let (ttft_ms, total_ms) = match self.mode {
            FixtureMode::Degraded => (20, 50),
            FixtureMode::HighQuality => (100, 300),
            FixtureMode::Unavailable => unreachable!(),
        };

        Ok(UnifiedResponse {
            text: response_text,
            input_tokens: request.prompt.len() / 4,
            output_tokens,
            ttft_ms,
            total_ms,
            backend_name: self.name.clone(),
        })
    }

    async fn generate_stream(
        &self,
        _request: &UnifiedRequest,
    ) -> ruvllm::error::Result<tokio::sync::mpsc::Receiver<ruvllm::error::Result<UnifiedStreamToken>>> {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(tx); // Immediately close — streaming not needed for tests
        Ok(rx)
    }
}

// ============================================================================
// Scoring utilities
// ============================================================================

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

/// Capture LoRA forward-pass outputs for all fixture prompts.
///
/// Returns a Vec of forward outputs (one Vec<f32> per task), which can be
/// compared before/after learning to measure adaptation.
pub fn capture_forward_outputs(
    tasks: &[FixtureTask],
    lora: &ruvllm::lora::micro_lora::MicroLoRA,
    dim: usize,
) -> Vec<Vec<f32>> {
    use ruvllm::lora::micro_lora::TargetModule;
    use ruvllm::sona::distillation_bridge::text_to_pseudo_embedding;

    tasks
        .iter()
        .map(|task| {
            let input = text_to_pseudo_embedding(&task.prompt, dim);
            lora.forward(&input, &TargetModule::QProj)
        })
        .collect()
}

/// Compare pre/post forward outputs and return per-prompt change magnitudes.
///
/// For each prompt, computes the L2 norm of (post_output - pre_output).
/// A non-zero value means the LoRA weights have shifted for that prompt.
pub fn compute_adaptation_magnitudes(
    pre_outputs: &[Vec<f32>],
    post_outputs: &[Vec<f32>],
) -> Vec<f32> {
    pre_outputs
        .iter()
        .zip(post_outputs.iter())
        .map(|(pre, post)| {
            let diff_sq: f32 = pre
                .iter()
                .zip(post.iter())
                .map(|(a, b)| (b - a) * (b - a))
                .sum();
            diff_sq.sqrt()
        })
        .collect()
}

/// Compute the L2 norm (magnitude) of a vector.
pub fn vector_magnitude(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Key-phrase scorer: what fraction of expected key phrases appear in text.
pub fn key_phrase_score(text: &str, key_phrases: &[String]) -> f32 {
    if key_phrases.is_empty() {
        return 0.0;
    }
    let text_lower = text.to_lowercase();
    let matches = key_phrases
        .iter()
        .filter(|kp| text_lower.contains(&kp.to_lowercase()))
        .count();
    matches as f32 / key_phrases.len() as f32
}

// ============================================================================
// Helpers
// ============================================================================

/// Extract lowercase word set from text.
fn word_set(text: &str) -> std::collections::HashSet<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Jaccard similarity between two word sets.
fn jaccard_similarity(
    a: &std::collections::HashSet<String>,
    b: &std::collections::HashSet<String>,
) -> f32 {
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_fixtures() {
        let tasks = FixtureBackend::load_fixtures();
        assert_eq!(tasks.len(), 24, "Should have 24 fixture tasks");

        let simple = tasks.iter().filter(|t| t.category == "simple").count();
        let medium = tasks.iter().filter(|t| t.category == "medium").count();
        let complex = tasks.iter().filter(|t| t.category == "complex").count();
        assert_eq!(simple, 8, "Should have 8 simple tasks");
        assert_eq!(medium, 8, "Should have 8 medium tasks");
        assert_eq!(complex, 8, "Should have 8 complex tasks");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_key_phrase_score() {
        let text = "Replace 'teh' with 'the' to correct the substitution error";
        let phrases = vec![
            "replace".to_string(),
            "the".to_string(),
            "corrected".to_string(),
            "substitution".to_string(),
        ];
        let score = key_phrase_score(text, &phrases);
        // "replace", "the", and "substitution" match, "corrected" does not
        assert!(score >= 0.5, "Expected >= 0.5, got {}", score);
    }

    #[test]
    fn test_jaccard_similarity() {
        let a: std::collections::HashSet<String> =
            ["fix", "typo", "in", "readme"].iter().map(|s| s.to_string()).collect();
        let b: std::collections::HashSet<String> =
            ["fix", "typo", "error"].iter().map(|s| s.to_string()).collect();
        let sim = jaccard_similarity(&a, &b);
        // Intersection: {fix, typo} = 2, Union: {fix, typo, in, readme, error} = 5
        assert!((sim - 0.4).abs() < 1e-6, "Expected 0.4, got {}", sim);
    }

    #[tokio::test]
    async fn test_fixture_backend_high_quality() {
        let tasks = FixtureBackend::load_fixtures();
        let backend = FixtureBackend::new("test-hq", FixtureMode::HighQuality, &tasks);

        assert!(backend.is_available().await);

        let request = UnifiedRequest {
            prompt: "fix typo: change 'teh' to 'the' in line 5".to_string(),
            max_tokens: 100,
            ..UnifiedRequest::default()
        };
        let response = backend.generate(&request).await.unwrap();
        assert!(response.text.contains("Replace"), "Should return high quality response");
        assert_eq!(backend.generate_count(), 1);
    }

    #[tokio::test]
    async fn test_fixture_backend_degraded() {
        let tasks = FixtureBackend::load_fixtures();
        let backend = FixtureBackend::new("test-deg", FixtureMode::Degraded, &tasks);

        let request = UnifiedRequest {
            prompt: "fix typo: change 'teh' to 'the' in line 5".to_string(),
            max_tokens: 100,
            ..UnifiedRequest::default()
        };
        let response = backend.generate(&request).await.unwrap();
        assert_eq!(response.text, "teh fix done");
    }

    #[tokio::test]
    async fn test_fixture_backend_unavailable() {
        let tasks = FixtureBackend::load_fixtures();
        let backend = FixtureBackend::new("test-unavail", FixtureMode::Unavailable, &tasks);
        assert!(!backend.is_available().await);
    }
}
