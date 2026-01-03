#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notifier - Signal notification system
Supports multiple notification channels
"""
import os
import json
import requests
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Notification:
    """Notification message"""
    title: str
    message: str
    priority: str = "normal"  # low, normal, high, urgent
    channel: str = "default"
    metadata: Optional[Dict] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class NotificationChannel:
    """Base notification channel"""
    
    def send(self, notification: Notification) -> bool:
        raise NotImplementedError


class ConsoleChannel(NotificationChannel):
    """Console/Log notification channel"""
    
    def send(self, notification: Notification) -> bool:
        priority_emoji = {
            "low": "ℹ️",
            "normal": "📌",
            "high": "⚠️",
            "urgent": "🚨"
        }
        emoji = priority_emoji.get(notification.priority, "📌")
        
        print(f"\n{emoji} [{notification.timestamp}] {notification.title}")
        print(f"   {notification.message}")
        
        if notification.metadata:
            for k, v in notification.metadata.items():
                print(f"   • {k}: {v}")
        
        return True


class WebhookChannel(NotificationChannel):
    """Generic webhook notification channel"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send(self, notification: Notification) -> bool:
        try:
            payload = {
                "title": notification.title,
                "message": notification.message,
                "priority": notification.priority,
                "timestamp": notification.timestamp,
                "metadata": notification.metadata or {}
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False


class FileChannel(NotificationChannel):
    """File-based notification (for logging)"""
    
    def __init__(self, filepath: str = "notifications.log"):
        self.filepath = filepath
    
    def send(self, notification: Notification) -> bool:
        try:
            with open(self.filepath, "a") as f:
                entry = {
                    "timestamp": notification.timestamp,
                    "title": notification.title,
                    "message": notification.message,
                    "priority": notification.priority,
                    "metadata": notification.metadata
                }
                f.write(json.dumps(entry) + "\n")
            return True
        except Exception as e:
            logger.error(f"File notification failed: {e}")
            return False


class Notifier:
    """
    Central notification manager
    """
    
    def __init__(self):
        self.channels: Dict[str, NotificationChannel] = {}
        self.default_channel = "console"
        
        # Add default console channel
        self.add_channel("console", ConsoleChannel())
    
    def add_channel(self, name: str, channel: NotificationChannel):
        """Add a notification channel"""
        self.channels[name] = channel
        logger.info(f"Added notification channel: {name}")
    
    def remove_channel(self, name: str):
        """Remove a notification channel"""
        if name in self.channels:
            del self.channels[name]
    
    def send(
        self,
        title: str,
        message: str,
        priority: str = "normal",
        channel: str = None,
        metadata: Dict = None
    ) -> bool:
        """Send a notification"""
        channel_name = channel or self.default_channel
        
        notification = Notification(
            title=title,
            message=message,
            priority=priority,
            channel=channel_name,
            metadata=metadata
        )
        
        if channel_name not in self.channels:
            logger.warning(f"Channel not found: {channel_name}")
            return False
        
        return self.channels[channel_name].send(notification)
    
    def broadcast(
        self,
        title: str,
        message: str,
        priority: str = "normal",
        metadata: Dict = None
    ) -> Dict[str, bool]:
        """Send to all channels"""
        results = {}
        
        for name, channel in self.channels.items():
            notification = Notification(
                title=title,
                message=message,
                priority=priority,
                channel=name,
                metadata=metadata
            )
            results[name] = channel.send(notification)
        
        return results
    
    def notify_signal(self, signal) -> bool:
        """Send VCP signal notification"""
        grade = signal.market_regime.split("|")[-1] if "|" in signal.market_regime else "?"
        
        return self.send(
            title=f"🎯 VCP Signal: {signal.symbol}",
            message=f"{signal.signal_type} detected on {signal.timeframe}",
            priority="high" if signal.score >= 70 else "normal",
            metadata={
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "type": signal.signal_type,
                "score": signal.score,
                "grade": grade,
                "pivot": f"${signal.pivot_high:,.2f}",
            }
        )
    
    def notify_market_gate(self, gate_result) -> bool:
        """Send Market Gate notification"""
        priority_map = {
            "GREEN": "normal",
            "YELLOW": "normal",
            "RED": "high"
        }
        
        return self.send(
            title=f"🚦 Market Gate: {gate_result.gate}",
            message=f"Score: {gate_result.score}/100",
            priority=priority_map.get(gate_result.gate, "normal"),
            metadata={
                "gate": gate_result.gate,
                "score": gate_result.score,
                "reasons": ", ".join(gate_result.reasons[:3])
            }
        )


# Default notifier instance
_notifier: Optional[Notifier] = None


def get_notifier() -> Notifier:
    """Get or create global notifier"""
    global _notifier
    if _notifier is None:
        _notifier = Notifier()
    return _notifier
