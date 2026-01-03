#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheduler - Periodic task management
"""
import time
import threading
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Task:
    """Scheduled task"""
    def __init__(
        self,
        name: str,
        func: Callable,
        interval_seconds: int,
        run_immediately: bool = False
    ):
        self.name = name
        self.func = func
        self.interval = interval_seconds
        self.last_run: Optional[datetime] = None
        self.next_run: datetime = datetime.now() if run_immediately else datetime.now() + timedelta(seconds=interval_seconds)
        self.run_count = 0
        self.error_count = 0
        self.enabled = True
    
    def should_run(self) -> bool:
        return self.enabled and datetime.now() >= self.next_run
    
    def run(self):
        try:
            logger.info(f"Running task: {self.name}")
            self.func()
            self.run_count += 1
            self.last_run = datetime.now()
            self.next_run = datetime.now() + timedelta(seconds=self.interval)
            logger.info(f"Task {self.name} completed. Next run: {self.next_run}")
        except Exception as e:
            self.error_count += 1
            logger.error(f"Task {self.name} failed: {e}")
            self.next_run = datetime.now() + timedelta(seconds=self.interval)


class Scheduler:
    """
    Simple task scheduler for periodic operations
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
    
    def add_task(
        self,
        name: str,
        func: Callable,
        interval_seconds: int,
        run_immediately: bool = False
    ):
        """Add a task to the scheduler"""
        task = Task(name, func, interval_seconds, run_immediately)
        self.tasks[name] = task
        logger.info(f"Added task: {name} (interval: {interval_seconds}s)")
    
    def remove_task(self, name: str):
        """Remove a task"""
        if name in self.tasks:
            del self.tasks[name]
            logger.info(f"Removed task: {name}")
    
    def enable_task(self, name: str):
        """Enable a task"""
        if name in self.tasks:
            self.tasks[name].enabled = True
    
    def disable_task(self, name: str):
        """Disable a task"""
        if name in self.tasks:
            self.tasks[name].enabled = False
    
    def _loop(self):
        """Main scheduler loop"""
        while self.is_running:
            for task in self.tasks.values():
                if task.should_run():
                    task.run()
            time.sleep(1)  # Check every second
    
    def start(self, background: bool = True):
        """Start the scheduler"""
        self.is_running = True
        logger.info("Scheduler starting...")
        
        if background:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        else:
            self._loop()
    
    def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")
    
    def get_status(self) -> Dict:
        """Get scheduler status"""
        return {
            "running": self.is_running,
            "tasks": {
                name: {
                    "enabled": task.enabled,
                    "run_count": task.run_count,
                    "error_count": task.error_count,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "next_run": task.next_run.isoformat()
                }
                for name, task in self.tasks.items()
            }
        }


# Default scheduler instance
_scheduler: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    """Get or create global scheduler"""
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler()
    return _scheduler


def setup_default_tasks():
    """Setup default scheduled tasks"""
    from ..orchestrator import Orchestrator
    
    orch = Orchestrator()
    scheduler = get_scheduler()
    
    # Market Gate check every 30 minutes
    scheduler.add_task(
        name="market_gate_check",
        func=orch.check_market_gate,
        interval_seconds=30 * 60,
        run_immediately=True
    )
    
    # VCP scan every hour
    scheduler.add_task(
        name="vcp_scan",
        func=orch.run_scan_cycle,
        interval_seconds=60 * 60,
        run_immediately=False
    )
    
    return scheduler
