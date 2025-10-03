"""
Quality Monitoring for Credibility Analyzer
Implements performance monitoring and drift detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time
import json
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
from dataclasses import dataclass
from enum import Enum

class AlertLevel(Enum):
    """Alert levels for quality monitoring."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class QualityMetric:
    """Quality metric data structure."""
    name: str
    value: float
    timestamp: datetime
    threshold: Optional[float] = None
    status: str = "normal"

@dataclass
class Alert:
    """Alert data structure."""
    level: AlertLevel
    message: str
    metric_name: str
    value: float
    threshold: float
    timestamp: datetime

class QualityMonitor:
    """
    Monitors analyzer performance and detects drift.
    """
    
    def __init__(self, window_size: int = 1000, alert_thresholds: Dict[str, float] = None):
        self.window_size = window_size
        self.logger = self._setup_logging()
        
        # Metrics storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.current_metrics = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'avg_confidence': 0.0,
            'avg_credibility_score': 0.0
        }
        
        # Drift detection
        self.baseline_metrics = {}
        self.drift_thresholds = {
            'accuracy_drift': 0.05,
            'confidence_drift': 0.1,
            'response_time_drift': 0.2,
            'prediction_distribution_drift': 0.15
        }
        
        # Alert system
        self.alert_thresholds = alert_thresholds or {
            'accuracy': 0.85,
            'confidence': 0.7,
            'response_time': 2.0,
            'error_rate': 0.1,
            'throughput': 1.0
        }
        
        self.alerts = deque(maxlen=1000)
        self.alert_callbacks = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for quality monitor."""
        logger = logging.getLogger('quality_monitor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Quality monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Quality monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check for drift
                self._detect_drift()
                
                # Check alerts
                self._check_alerts()
                
                # Sleep for monitoring interval
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def record_prediction(self, prediction_result: Dict[str, Any], 
                         processing_time: float, success: bool = True) -> None:
        """Record a prediction for quality monitoring."""
        with self.lock:
            timestamp = datetime.now()
            
            # Update performance stats
            self.performance_stats['total_requests'] += 1
            if success:
                self.performance_stats['successful_requests'] += 1
            else:
                self.performance_stats['failed_requests'] += 1
            
            # Update averages
            total_requests = self.performance_stats['total_requests']
            if total_requests > 0:
                # Exponential moving average for response time
                alpha = 0.1
                current_avg = self.performance_stats['avg_response_time']
                self.performance_stats['avg_response_time'] = (
                    alpha * processing_time + (1 - alpha) * current_avg
                )
                
                # Update confidence and credibility averages
                if success and 'confidence' in prediction_result:
                    current_avg_conf = self.performance_stats['avg_confidence']
                    self.performance_stats['avg_confidence'] = (
                        alpha * prediction_result['confidence'] + (1 - alpha) * current_avg_conf
                    )
                
                if success and 'credibility_score' in prediction_result:
                    current_avg_score = self.performance_stats['avg_credibility_score']
                    self.performance_stats['avg_credibility_score'] = (
                        alpha * prediction_result['credibility_score'] + (1 - alpha) * current_avg_score
                    )
            
            # Record metrics
            self._record_metric('response_time', processing_time, timestamp)
            
            if success:
                if 'confidence' in prediction_result:
                    self._record_metric('confidence', prediction_result['confidence'], timestamp)
                
                if 'credibility_score' in prediction_result:
                    self._record_metric('credibility_score', prediction_result['credibility_score'], timestamp)
                
                # Record prediction distribution
                label = prediction_result.get('label', 'Unknown')
                fake_count = 1 if label == 'Fake' else 0
                real_count = 1 if label == 'Real' else 0
                
                self._record_metric('fake_predictions', fake_count, timestamp)
                self._record_metric('real_predictions', real_count, timestamp)
    
    def _record_metric(self, name: str, value: float, timestamp: datetime) -> None:
        """Record a metric value."""
        metric = QualityMetric(
            name=name,
            value=value,
            timestamp=timestamp,
            threshold=self.alert_thresholds.get(name)
        )
        
        self.metrics_history[name].append(metric)
        self.current_metrics[name] = value
    
    def _update_system_metrics(self) -> None:
        """Update system-level metrics."""
        timestamp = datetime.now()
        
        # Calculate derived metrics
        total_requests = self.performance_stats['total_requests']
        successful_requests = self.performance_stats['successful_requests']
        
        if total_requests > 0:
            success_rate = successful_requests / total_requests
            error_rate = 1.0 - success_rate
            
            self._record_metric('success_rate', success_rate, timestamp)
            self._record_metric('error_rate', error_rate, timestamp)
            
            # Calculate throughput (requests per minute)
            if len(self.metrics_history['response_time']) > 1:
                recent_metrics = list(self.metrics_history['response_time'])[-10:]
                if recent_metrics:
                    avg_response_time = np.mean([m.value for m in recent_metrics])
                    throughput = 60.0 / avg_response_time if avg_response_time > 0 else 0
                    self._record_metric('throughput', throughput, timestamp)
    
    def _detect_drift(self) -> None:
        """Detect performance drift."""
        for metric_name, history in self.metrics_history.items():
            if len(history) < 10:  # Need minimum data
                continue
            
            # Calculate current vs baseline
            if metric_name not in self.baseline_metrics:
                # Establish baseline
                baseline_values = [m.value for m in list(history)[:50]]
                if baseline_values:
                    self.baseline_metrics[metric_name] = {
                        'mean': np.mean(baseline_values),
                        'std': np.std(baseline_values)
                    }
                continue
            
            # Compare recent performance to baseline
            recent_values = [m.value for m in list(history)[-20:]]
            if not recent_values:
                continue
            
            current_mean = np.mean(recent_values)
            baseline_mean = self.baseline_metrics[metric_name]['mean']
            baseline_std = self.baseline_metrics[metric_name]['std']
            
            # Calculate drift
            if baseline_std > 0:
                drift_score = abs(current_mean - baseline_mean) / baseline_std
                
                # Check drift threshold
                drift_threshold = self.drift_thresholds.get(metric_name, 0.1)
                
                if drift_score > drift_threshold:
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"Performance drift detected in {metric_name}",
                        metric_name,
                        current_mean,
                        drift_threshold
                    )
    
    def _check_alerts(self) -> None:
        """Check for alert conditions."""
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name not in self.current_metrics:
                continue
            
            current_value = self.current_metrics[metric_name]
            
            # Determine alert level
            if current_value < threshold * 0.5:
                alert_level = AlertLevel.CRITICAL
            elif current_value < threshold * 0.8:
                alert_level = AlertLevel.ERROR
            elif current_value < threshold:
                alert_level = AlertLevel.WARNING
            else:
                continue
            
            # Create alert
            self._create_alert(
                alert_level,
                f"{metric_name} below threshold: {current_value:.3f} < {threshold:.3f}",
                metric_name,
                current_value,
                threshold
            )
    
    def _create_alert(self, level: AlertLevel, message: str, 
                     metric_name: str, value: float, threshold: float) -> None:
        """Create and store an alert."""
        alert = Alert(
            level=level,
            message=message,
            metric_name=metric_name,
            value=value,
            threshold=threshold,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # Log alert
        if level == AlertLevel.CRITICAL:
            self.logger.critical(message)
        elif level == AlertLevel.ERROR:
            self.logger.error(message)
        elif level == AlertLevel.WARNING:
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback) -> None:
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        with self.lock:
            return {
                'performance_stats': self.performance_stats.copy(),
                'current_metrics': self.current_metrics.copy(),
                'baseline_metrics': self.baseline_metrics.copy(),
                'alert_count': len(self.alerts),
                'recent_alerts': [alert.__dict__ for alert in list(self.alerts)[-5:]],
                'is_monitoring': self.is_monitoring
            }
    
    def get_metric_trends(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get metric trends over time."""
        if metric_name not in self.metrics_history:
            return {'error': f'Metric {metric_name} not found'}
        
        history = self.metrics_history[metric_name]
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent metrics
        recent_metrics = [m for m in history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {'error': f'No data for {metric_name} in last {hours} hours'}
        
        values = [m.value for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]
        
        return {
            'metric_name': metric_name,
            'values': values,
            'timestamps': timestamps,
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'trend': self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        if abs(slope) < 0.001:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics to JSON file."""
        if filename is None:
            filename = f"quality_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'performance_stats': self.performance_stats,
            'current_metrics': self.current_metrics,
            'baseline_metrics': self.baseline_metrics,
            'alert_thresholds': self.alert_thresholds,
            'drift_thresholds': self.drift_thresholds,
            'metrics_history': {
                name: [
                    {
                        'value': m.value,
                        'timestamp': m.timestamp.isoformat(),
                        'threshold': m.threshold,
                        'status': m.status
                    }
                    for m in list(history)
                ]
                for name, history in self.metrics_history.items()
            },
            'alerts': [
                {
                    'level': alert.level.value,
                    'message': alert.message,
                    'metric_name': alert.metric_name,
                    'value': alert.value,
                    'threshold': alert.threshold,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in list(self.alerts)
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filename}")
        return filename
    
    def load_metrics(self, filename: str) -> None:
        """Load metrics from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Load performance stats
        self.performance_stats = data.get('performance_stats', self.performance_stats)
        
        # Load baseline metrics
        self.baseline_metrics = data.get('baseline_metrics', {})
        
        # Load metrics history
        for name, history_data in data.get('metrics_history', {}).items():
            self.metrics_history[name] = deque(maxlen=self.window_size)
            for metric_data in history_data:
                metric = QualityMetric(
                    name=name,
                    value=metric_data['value'],
                    timestamp=datetime.fromisoformat(metric_data['timestamp']),
                    threshold=metric_data.get('threshold'),
                    status=metric_data.get('status', 'normal')
                )
                self.metrics_history[name].append(metric)
        
        self.logger.info(f"Metrics loaded from {filename}")

def main():
    """Test the quality monitor."""
    print("📊 Testing Quality Monitor")
    print("=" * 50)
    
    # Initialize monitor
    monitor = QualityMonitor(window_size=100)
    
    print("🧪 Simulating predictions...")
    
    # Simulate some predictions
    for i in range(50):
        # Simulate varying performance
        processing_time = np.random.normal(0.2, 0.05)
        confidence = np.random.normal(0.8, 0.1)
        credibility_score = np.random.normal(0.6, 0.2)
        
        # Simulate occasional failures
        success = np.random.random() > 0.05
        
        prediction_result = {
            'label': 'Real' if np.random.random() > 0.3 else 'Fake',
            'confidence': max(0.0, min(1.0, confidence)),
            'credibility_score': max(0.0, min(1.0, credibility_score))
        }
        
        monitor.record_prediction(prediction_result, processing_time, success)
        
        # Add some delay
        time.sleep(0.01)
    
    # Start monitoring
    print("\n🔄 Starting monitoring...")
    monitor.start_monitoring()
    
    # Let it run for a bit
    time.sleep(5)
    
    # Get summary
    print("\n📊 Performance Summary:")
    summary = monitor.get_performance_summary()
    
    print(f"   Total requests: {summary['performance_stats']['total_requests']}")
    print(f"   Success rate: {summary['performance_stats']['successful_requests'] / summary['performance_stats']['total_requests']:.1%}")
    print(f"   Avg response time: {summary['performance_stats']['avg_response_time']:.3f}s")
    print(f"   Avg confidence: {summary['performance_stats']['avg_confidence']:.3f}")
    print(f"   Alerts: {summary['alert_count']}")
    
    # Get trends
    print("\n📈 Metric Trends:")
    for metric_name in ['response_time', 'confidence', 'credibility_score']:
        trend = monitor.get_metric_trends(metric_name, hours=1)
        if 'error' not in trend:
            print(f"   {metric_name}: {trend['mean']:.3f} ± {trend['std']:.3f} ({trend['trend']})")
    
    # Stop monitoring
    print("\n⏹️ Stopping monitoring...")
    monitor.stop_monitoring()
    
    # Export metrics
    print("\n💾 Exporting metrics...")
    filename = monitor.export_metrics()
    print(f"   Exported to: {filename}")

if __name__ == "__main__":
    main()
