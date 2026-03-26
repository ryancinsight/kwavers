pub mod pipeline;
pub mod queue;
pub mod scheduler;
pub mod task;

pub use pipeline::PipelineCoordinator;
pub use queue::{PoolMetrics, ThreadPoolConfig, WorkQueue};
pub use scheduler::RealTimeScheduler;
pub use task::{TaskMetrics, TaskPriority, WorkItem};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::KwaversResult;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_work_item_creation() {
        let work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync> = Arc::new(|| Ok(()));
        let item = WorkItem::new(1, TaskPriority::Normal, work, 1000);

        assert_eq!(item.task_id, 1);
        assert_eq!(item.priority, TaskPriority::Normal);
        assert!(item.deadline.is_none());
    }

    #[test]
    fn test_work_item_deadline() {
        let work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync> = Arc::new(|| Ok(()));
        let item = WorkItem::new(1, TaskPriority::High, work, 1000).with_deadline(100, 1000); // Created at 1000, deadline is 1100

        assert!(item.deadline.is_some());
        assert!(!item.is_overdue(1050));
        assert!(item.is_overdue(1150));
    }

    #[test]
    fn test_scheduler_submit() {
        let scheduler = RealTimeScheduler::new();
        let work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync> = Arc::new(|| Ok(()));

        let task_id = scheduler.submit(TaskPriority::Normal, work).unwrap();
        assert_eq!(task_id, 0);

        let metrics = scheduler.metrics();
        assert_eq!(metrics.total_submitted, 1);
        assert_eq!(metrics.current_queue_depth, 1);
    }

    #[test]
    fn test_scheduler_priority_ordering() {
        let scheduler = RealTimeScheduler::new();
        let work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync> = Arc::new(|| Ok(()));

        scheduler.submit(TaskPriority::Low, Arc::clone(&work)).ok();
        scheduler
            .submit(TaskPriority::Critical, Arc::clone(&work))
            .ok();
        scheduler
            .submit(TaskPriority::Normal, Arc::clone(&work))
            .ok();

        // First task should be Critical
        if let Some(item) = scheduler.next_task() {
            assert_eq!(item.priority, TaskPriority::Critical);
        }
    }

    #[test]
    fn test_scheduler_execution() {
        let scheduler = RealTimeScheduler::new();
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = Arc::clone(&counter);

        let work = Arc::new(move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });

        scheduler.submit(TaskPriority::Normal, work).ok();

        if let Some(item) = scheduler.next_task() {
            scheduler.execute_task(item).ok();
        }

        let metrics = scheduler.metrics();
        assert_eq!(metrics.total_completed, 1);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_thread_pool_config_defaults() {
        let config = ThreadPoolConfig::default();
        assert!(config.num_threads > 0);
        assert!(config.work_stealing_enabled);
    }

    #[test]
    fn test_work_queue_creation() {
        let config = ThreadPoolConfig::default();
        let queue = WorkQueue::new(config);

        assert!(queue.config.num_threads > 0);
    }

    #[test]
    fn test_work_queue_submit() {
        let config = ThreadPoolConfig {
            num_threads: 2,
            ..Default::default()
        };
        let queue = WorkQueue::new(config);
        let work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync> = Arc::new(|| Ok(()));

        let task_id = queue.submit(TaskPriority::Normal, work).unwrap();
        assert_eq!(task_id, 0);
    }

    #[test]
    fn test_work_queue_metrics() {
        let config = ThreadPoolConfig {
            num_threads: 1,
            ..Default::default()
        };
        let queue = WorkQueue::new(config);
        let work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync> = Arc::new(|| Ok(()));

        queue.submit(TaskPriority::Normal, work).ok();
        let metrics = queue.metrics();

        assert!(metrics.active_threads <= queue.config.num_threads);
    }

    #[test]
    fn test_pipeline_coordinator_creation() {
        let coordinator = PipelineCoordinator::new(3).unwrap();
        assert_eq!(coordinator.num_stages(), 3);
    }

    #[test]
    fn test_pipeline_coordinator_invalid_stages() {
        let result = PipelineCoordinator::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_coordinator_submit() {
        let coordinator = PipelineCoordinator::new(2).unwrap();
        let work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync> = Arc::new(|| Ok(()));

        let task_id = coordinator
            .submit_to_stage(0, TaskPriority::Normal, work)
            .unwrap();
        assert_eq!(task_id, 0);

        let result = coordinator.submit_to_stage(5, TaskPriority::Normal, Arc::new(|| Ok(())));
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_stage_metrics() {
        let coordinator = PipelineCoordinator::new(2).unwrap();
        let work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync> = Arc::new(|| Ok(()));

        coordinator
            .submit_to_stage(0, TaskPriority::Normal, work)
            .ok();

        let metrics = coordinator.stage_metrics(0).unwrap();
        assert!(metrics.active_threads <= coordinator.stages[0].config.num_threads); // Work may be executing

        let result = coordinator.stage_metrics(5);
        assert!(result.is_err());
    }
}
