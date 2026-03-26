use super::encounter::EncounterId;
use super::profile::iso8601_now;
use crate::core::error::{KwaversError, KwaversResult};
use std::time::{SystemTime, UNIX_EPOCH};

fn iso8601_future_days(days: i64) -> String {
    let now = SystemTime::now();
    let duration = now.duration_since(UNIX_EPOCH).unwrap_or_default();

    let seconds = duration.as_secs();
    // Approximate days since epoch (ignoring leap seconds for this estimate)
    let total_days = (seconds / 86400) as i64 + days;
    let year = 1970 + total_days / 365;
    let month = (total_days % 365) / 30 + 1;
    let day = (total_days % 365) % 30 + 1;

    format!("{:04}-{:02}-{:02}T00:00:00Z", year, month, day)
}

/// Treatment plan status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreatmentStatus {
    /// Plan created, not yet started
    Planned,
    /// In progress
    Active,
    /// Completed successfully
    Completed,
    /// Cancelled or abandoned
    Cancelled,
    /// On hold pending review
    OnHold,
}

impl TreatmentStatus {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Planned => "Planned",
            Self::Active => "Active",
            Self::Completed => "Completed",
            Self::Cancelled => "Cancelled",
            Self::OnHold => "On Hold",
        }
    }
}

/// Treatment plan
#[derive(Debug, Clone)]
pub struct TreatmentPlan {
    /// Unique plan identifier
    pub plan_id: String,
    /// Associated encounter
    pub encounter_id: EncounterId,
    /// Treatment description
    pub treatment_description: String,
    /// Target indication (diagnosis)
    pub target_indication: String,
    /// Expected duration (days)
    pub expected_duration_days: u32,
    /// Start date (ISO 8601)
    pub start_date: String,
    /// Planned end date (ISO 8601)
    pub planned_end_date: String,
    /// Status
    pub status: TreatmentStatus,
    /// Number of planned sessions
    pub planned_sessions: u32,
    /// Completed sessions
    pub completed_sessions: u32,
    /// Clinical objectives
    pub objectives: Vec<String>,
    /// Success criteria
    pub success_criteria: Vec<String>,
}

impl TreatmentPlan {
    /// Create a new treatment plan
    pub fn new(
        encounter_id: EncounterId,
        treatment_description: impl Into<String>,
        target_indication: impl Into<String>,
        planned_sessions: u32,
    ) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);

        Self {
            plan_id: format!("PLAN_{:08}", id),
            encounter_id,
            treatment_description: treatment_description.into(),
            target_indication: target_indication.into(),
            expected_duration_days: planned_sessions * 7, // Estimate 1 session per week
            start_date: iso8601_now(),
            planned_end_date: iso8601_future_days(planned_sessions as i64 * 7),
            status: TreatmentStatus::Planned,
            planned_sessions,
            completed_sessions: 0,
            objectives: Vec::new(),
            success_criteria: Vec::new(),
        }
    }

    /// Add an objective
    pub fn add_objective(&mut self, objective: impl Into<String>) {
        self.objectives.push(objective.into());
    }

    /// Add success criteria
    pub fn add_success_criteria(&mut self, criteria: impl Into<String>) {
        self.success_criteria.push(criteria.into());
    }

    /// Mark a session as completed
    pub fn complete_session(&mut self) -> KwaversResult<()> {
        if self.completed_sessions >= self.planned_sessions {
            return Err(KwaversError::InvalidInput(
                "All planned sessions already completed".to_string(),
            ));
        }

        self.completed_sessions += 1;

        if self.completed_sessions == self.planned_sessions {
            self.status = TreatmentStatus::Completed;
        } else if self.status == TreatmentStatus::Planned {
            self.status = TreatmentStatus::Active;
        }

        Ok(())
    }
}
