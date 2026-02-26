//! 10 exotic applications of ruvector-verified beyond dimension checks.
//!
//! Each module demonstrates a real-world domain where proof-carrying vector
//! operations provide structural safety that runtime assertions cannot.

pub mod weapons_filter;
pub mod medical_diagnostics;
pub mod financial_routing;
pub mod agent_contracts;
pub mod sensor_swarm;
pub mod quantization_proof;
pub mod verified_memory;
pub mod vector_signatures;
pub mod simulation_integrity;
pub mod legal_forensics;

/// Shared proof receipt that all domains produce.
#[derive(Debug, Clone)]
pub struct ProofReceipt {
    /// Domain identifier (e.g. "weapons", "medical", "trade").
    pub domain: String,
    /// Human-readable description of what was proved.
    pub claim: String,
    /// Proof term ID in the environment.
    pub proof_id: u32,
    /// 82-byte attestation bytes.
    pub attestation_bytes: Vec<u8>,
    /// Proof tier used (reflex/standard/deep).
    pub tier: String,
    /// Whether the gate passed.
    pub gate_passed: bool,
}
