use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub severity: AlertSeverity,
    pub component: String,
    pub metric: String,
    pub message: String,
    pub current_value: f64,
    pub threshold: f64,
    pub status: AlertStatus,
    pub resolution_time: Option<DateTime<Utc>>,
    pub escalation_level: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertStatus {
    Active,
    Acknowledged,
    Resolved,
    Suppressed,
}

#[derive(Debug, Clone)]
pub struct AlertRule {
    pub name: String,
    pub component: String,
    pub metric: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub message_template: String,
    pub cooldown_minutes: i32,
    pub escalation_minutes: i32,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub enum AlertCondition {
    GreaterThan(f64),
    LessThan(f64),
    PercentageIncrease { threshold: f64, window_minutes: i32 },
    PercentageDecrease { threshold: f64, window_minutes: i32 },
    AbsenceOfData { minutes: i32 },
    ErrorRateSpike { threshold: f64, window_minutes: i32 },
}

#[derive(Debug, Clone)]
pub struct NotificationChannel {
    pub name: String,
    pub channel_type: ChannelType,
    pub config: HashMap<String, String>,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub enum ChannelType {
    Email,
    Slack,
    Discord,
    Webhook,
    PagerDuty,
}

pub struct AlertingSystem {
    alert_rules: Vec<AlertRule>,
    notification_channels: Vec<NotificationChannel>,
    active_alerts: HashMap<String, Alert>,
    alert_history: Vec<Alert>,
    suppression_rules: Vec<SuppressionRule>,
}

#[derive(Debug, Clone)]
pub struct SuppressionRule {
    pub name: String,
    pub component_pattern: String,
    pub metric_pattern: String,
    pub start_time: String,  // HH:MM format
    pub end_time: String,    // HH:MM format
    pub days_of_week: Vec<String>,
    pub enabled: bool,
}

impl AlertingSystem {
    pub fn new() -> Self {
        Self {
            alert_rules: Self::load_default_rules(),
            notification_channels: Self::load_default_channels(),
            active_alerts: HashMap::new(),
            alert_history: Vec::new(),
            suppression_rules: Self::load_default_suppressions(),
        }
    }

    fn load_default_rules() -> Vec<AlertRule> {
        vec![
            // CPU Alerts
            AlertRule {
                name: "high_cpu_usage".to_string(),
                component: "*".to_string(),
                metric: "cpu_usage".to_string(),
                condition: AlertCondition::GreaterThan(85.0),
                severity: AlertSeverity::Warning,
                message_template: "High CPU usage detected on {component}: {value}%".to_string(),
                cooldown_minutes: 10,
                escalation_minutes: 30,
                enabled: true,
            },
            AlertRule {
                name: "critical_cpu_usage".to_string(),
                component: "*".to_string(),
                metric: "cpu_usage".to_string(),
                condition: AlertCondition::GreaterThan(95.0),
                severity: AlertSeverity::Critical,
                message_template: "CRITICAL: CPU usage on {component} reached {value}%".to_string(),
                cooldown_minutes: 5,
                escalation_minutes: 15,
                enabled: true,
            },

            // Memory Alerts
            AlertRule {
                name: "high_memory_usage".to_string(),
                component: "*".to_string(),
                metric: "memory_usage".to_string(),
                condition: AlertCondition::GreaterThan(80.0),
                severity: AlertSeverity::Warning,
                message_template: "High memory usage detected on {component}: {value}%".to_string(),
                cooldown_minutes: 15,
                escalation_minutes: 45,
                enabled: true,
            },
            AlertRule {
                name: "critical_memory_usage".to_string(),
                component: "*".to_string(),
                metric: "memory_usage".to_string(),
                condition: AlertCondition::GreaterThan(90.0),
                severity: AlertSeverity::Critical,
                message_template: "CRITICAL: Memory usage on {component} reached {value}%".to_string(),
                cooldown_minutes: 5,
                escalation_minutes: 20,
                enabled: true,
            },

            // Error Rate Alerts
            AlertRule {
                name: "high_error_rate".to_string(),
                component: "*".to_string(),
                metric: "error_rate".to_string(),
                condition: AlertCondition::GreaterThan(5.0),
                severity: AlertSeverity::Warning,
                message_template: "High error rate detected on {component}: {value}%".to_string(),
                cooldown_minutes: 10,
                escalation_minutes: 30,
                enabled: true,
            },
            AlertRule {
                name: "critical_error_rate".to_string(),
                component: "*".to_string(),
                metric: "error_rate".to_string(),
                condition: AlertCondition::GreaterThan(15.0),
                severity: AlertSeverity::Critical,
                message_template: "CRITICAL: Error rate on {component} reached {value}%".to_string(),
                cooldown_minutes: 2,
                escalation_minutes: 10,
                enabled: true,
            },

            // Response Time Alerts
            AlertRule {
                name: "slow_response_time".to_string(),
                component: "api-server".to_string(),
                metric: "avg_response_time".to_string(),
                condition: AlertCondition::GreaterThan(2000.0), // 2 seconds
                severity: AlertSeverity::Warning,
                message_template: "Slow response time on {component}: {value}ms".to_string(),
                cooldown_minutes: 15,
                escalation_minutes: 45,
                enabled: true,
            },

            // Request Rate Alerts
            AlertRule {
                name: "request_rate_spike".to_string(),
                component: "api-server".to_string(),
                metric: "request_rate".to_string(),
                condition: AlertCondition::PercentageIncrease { threshold: 100.0, window_minutes: 10 },
                severity: AlertSeverity::Warning,
                message_template: "Request rate spike detected on {component}: {value} req/s".to_string(),
                cooldown_minutes: 20,
                escalation_minutes: 60,
                enabled: true,
            },

            // Service Availability
            AlertRule {
                name: "service_down".to_string(),
                component: "*".to_string(),
                metric: "health_check".to_string(),
                condition: AlertCondition::LessThan(1.0),
                severity: AlertSeverity::Critical,
                message_template: "Service {component} appears to be down or unhealthy".to_string(),
                cooldown_minutes: 1,
                escalation_minutes: 5,
                enabled: true,
            },

            // Data Processing Alerts
            AlertRule {
                name: "processing_backlog".to_string(),
                component: "processing-queue".to_string(),
                metric: "queue_size".to_string(),
                condition: AlertCondition::GreaterThan(1000.0),
                severity: AlertSeverity::Warning,
                message_template: "Processing queue backlog: {value} items pending".to_string(),
                cooldown_minutes: 30,
                escalation_minutes: 90,
                enabled: true,
            },

            // GPU Alerts
            AlertRule {
                name: "gpu_memory_high".to_string(),
                component: "*".to_string(),
                metric: "gpu_memory_usage".to_string(),
                condition: AlertCondition::GreaterThan(85.0),
                severity: AlertSeverity::Warning,
                message_template: "High GPU memory usage on {component}: {value}%".to_string(),
                cooldown_minutes: 10,
                escalation_minutes: 30,
                enabled: true,
            },

            // Storage Alerts
            AlertRule {
                name: "disk_space_low".to_string(),
                component: "*".to_string(),
                metric: "disk_usage".to_string(),
                condition: AlertCondition::GreaterThan(80.0),
                severity: AlertSeverity::Warning,
                message_template: "Disk space running low on {component}: {value}% used".to_string(),
                cooldown_minutes: 60,
                escalation_minutes: 180,
                enabled: true,
            },
        ]
    }

    fn load_default_channels() -> Vec<NotificationChannel> {
        vec![
            NotificationChannel {
                name: "email_alerts".to_string(),
                channel_type: ChannelType::Email,
                config: [
                    ("smtp_host".to_string(), "smtp.gmail.com".to_string()),
                    ("smtp_port".to_string(), "587".to_string()),
                    ("username".to_string(), "alerts@videorss.com".to_string()),
                    ("to_addresses".to_string(), "admin@videorss.com,ops@videorss.com".to_string()),
                ].into_iter().collect(),
                enabled: true,
            },
            NotificationChannel {
                name: "slack_alerts".to_string(),
                channel_type: ChannelType::Slack,
                config: [
                    ("webhook_url".to_string(), "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK".to_string()),
                    ("channel".to_string(), "#alerts".to_string()),
                ].into_iter().collect(),
                enabled: true,
            },
            NotificationChannel {
                name: "discord_alerts".to_string(),
                channel_type: ChannelType::Discord,
                config: [
                    ("webhook_url".to_string(), "https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK".to_string()),
                ].into_iter().collect(),
                enabled: false,
            },
        ]
    }

    fn load_default_suppressions() -> Vec<SuppressionRule> {
        vec![
            SuppressionRule {
                name: "maintenance_window".to_string(),
                component_pattern: "*".to_string(),
                metric_pattern: "*".to_string(),
                start_time: "02:00".to_string(),
                end_time: "04:00".to_string(),
                days_of_week: vec!["Sunday".to_string()],
                enabled: false,
            },
            SuppressionRule {
                name: "batch_processing".to_string(),
                component_pattern: "batch-*".to_string(),
                metric_pattern: "cpu_usage".to_string(),
                start_time: "00:00".to_string(),
                end_time: "06:00".to_string(),
                days_of_week: vec!["Monday".to_string(), "Wednesday".to_string(), "Friday".to_string()],
                enabled: true,
            },
        ]
    }

    pub async fn process_metrics(&mut self, metrics: &HashMap<String, HashMap<String, f64>>) -> Result<()> {
        for (component, component_metrics) in metrics {
            for (metric_name, &metric_value) in component_metrics {
                self.evaluate_alerts(component, metric_name, metric_value).await?;
            }
        }

        // Check for escalations
        self.check_escalations().await?;

        // Resolve alerts if conditions are no longer met
        self.resolve_alerts(metrics).await?;

        Ok(())
    }

    async fn evaluate_alerts(&mut self, component: &str, metric: &str, value: f64) -> Result<()> {
        for rule in &self.alert_rules {
            if !rule.enabled {
                continue;
            }

            if !self.matches_component(&rule.component, component) {
                continue;
            }

            if rule.metric != metric {
                continue;
            }

            if self.is_suppressed(component, metric).await {
                continue;
            }

            let alert_key = format!("{}_{}", component, rule.name);

            // Check cooldown
            if let Some(existing_alert) = self.active_alerts.get(&alert_key) {
                let cooldown = chrono::Duration::minutes(rule.cooldown_minutes as i64);
                if Utc::now() - existing_alert.timestamp < cooldown {
                    continue;
                }
            }

            if self.evaluate_condition(&rule.condition, value, component, metric).await? {
                self.trigger_alert(rule, component, metric, value).await?;
            }
        }

        Ok(())
    }

    fn matches_component(&self, pattern: &str, component: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        if pattern.contains('*') {
            let prefix = pattern.trim_end_matches('*');
            return component.starts_with(prefix);
        }

        pattern == component
    }

    async fn is_suppressed(&self, component: &str, metric: &str) -> bool {
        let now = Utc::now();
        let current_time = now.format("%H:%M").to_string();
        let current_day = now.format("%A").to_string();

        for rule in &self.suppression_rules {
            if !rule.enabled {
                continue;
            }

            if !self.matches_component(&rule.component_pattern, component) {
                continue;
            }

            if !self.matches_component(&rule.metric_pattern, metric) {
                continue;
            }

            if !rule.days_of_week.contains(&current_day) {
                continue;
            }

            if self.is_time_in_range(&current_time, &rule.start_time, &rule.end_time) {
                return true;
            }
        }

        false
    }

    fn is_time_in_range(&self, current: &str, start: &str, end: &str) -> bool {
        current >= start && current <= end
    }

    async fn evaluate_condition(
        &self,
        condition: &AlertCondition,
        value: f64,
        _component: &str,
        _metric: &str,
    ) -> Result<bool> {
        match condition {
            AlertCondition::GreaterThan(threshold) => Ok(value > *threshold),
            AlertCondition::LessThan(threshold) => Ok(value < *threshold),
            AlertCondition::PercentageIncrease { threshold, window_minutes: _ } => {
                // In a real implementation, this would compare with historical data
                // For now, simulate based on current value
                Ok(value > 100.0 + threshold)
            },
            AlertCondition::PercentageDecrease { threshold, window_minutes: _ } => {
                // In a real implementation, this would compare with historical data
                Ok(value < 100.0 - threshold)
            },
            AlertCondition::AbsenceOfData { minutes: _ } => {
                // In a real implementation, this would check timestamp of last data point
                Ok(false)
            },
            AlertCondition::ErrorRateSpike { threshold, window_minutes: _ } => {
                Ok(value > *threshold)
            },
        }
    }

    async fn trigger_alert(
        &mut self,
        rule: &AlertRule,
        component: &str,
        metric: &str,
        value: f64,
    ) -> Result<()> {
        let alert_id = uuid::Uuid::new_v4().to_string();
        let alert_key = format!("{}_{}", component, rule.name);

        let message = rule.message_template
            .replace("{component}", component)
            .replace("{metric}", metric)
            .replace("{value}", &format!("{:.2}", value));

        let threshold = match &rule.condition {
            AlertCondition::GreaterThan(t) | AlertCondition::LessThan(t) => *t,
            AlertCondition::PercentageIncrease { threshold, .. } |
            AlertCondition::PercentageDecrease { threshold, .. } |
            AlertCondition::ErrorRateSpike { threshold, .. } => *threshold,
            AlertCondition::AbsenceOfData { .. } => 0.0,
        };

        let alert = Alert {
            id: alert_id.clone(),
            timestamp: Utc::now(),
            severity: rule.severity.clone(),
            component: component.to_string(),
            metric: metric.to_string(),
            message: message.clone(),
            current_value: value,
            threshold,
            status: AlertStatus::Active,
            resolution_time: None,
            escalation_level: 0,
        };

        info!("Alert triggered: {} - {}", alert_id, message);

        // Store alert
        self.active_alerts.insert(alert_key, alert.clone());
        self.alert_history.push(alert.clone());

        // Send notifications
        self.send_notifications(&alert).await?;

        Ok(())
    }

    async fn send_notifications(&self, alert: &Alert) -> Result<()> {
        for channel in &self.notification_channels {
            if !channel.enabled {
                continue;
            }

            if let Err(e) = self.send_notification(channel, alert).await {
                error!("Failed to send notification via {}: {}", channel.name, e);
            }
        }

        Ok(())
    }

    async fn send_notification(&self, channel: &NotificationChannel, alert: &Alert) -> Result<()> {
        match &channel.channel_type {
            ChannelType::Email => self.send_email_notification(channel, alert).await,
            ChannelType::Slack => self.send_slack_notification(channel, alert).await,
            ChannelType::Discord => self.send_discord_notification(channel, alert).await,
            ChannelType::Webhook => self.send_webhook_notification(channel, alert).await,
            ChannelType::PagerDuty => self.send_pagerduty_notification(channel, alert).await,
        }
    }

    async fn send_email_notification(&self, channel: &NotificationChannel, alert: &Alert) -> Result<()> {
        // In a real implementation, this would use SMTP to send emails
        info!("Sending email notification for alert {} via {}", alert.id, channel.name);

        let subject = format!("[{}] {} - {}",
            format!("{:?}", alert.severity).to_uppercase(),
            alert.component,
            match alert.severity {
                AlertSeverity::Critical => "CRITICAL ALERT",
                AlertSeverity::Warning => "Warning Alert",
                AlertSeverity::Info => "Info Alert",
            }
        );

        let body = format!(
            "Alert Details:\n\
            ID: {}\n\
            Component: {}\n\
            Metric: {}\n\
            Current Value: {:.2}\n\
            Threshold: {:.2}\n\
            Message: {}\n\
            Timestamp: {}\n\
            \n\
            This is an automated alert from the Video RSS Aggregator monitoring system.",
            alert.id,
            alert.component,
            alert.metric,
            alert.current_value,
            alert.threshold,
            alert.message,
            alert.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        );

        info!("Email Alert - Subject: {}, Body: {}", subject, body);
        Ok(())
    }

    async fn send_slack_notification(&self, channel: &NotificationChannel, alert: &Alert) -> Result<()> {
        // In a real implementation, this would send to Slack webhook
        info!("Sending Slack notification for alert {} via {}", alert.id, channel.name);

        let color = match alert.severity {
            AlertSeverity::Critical => "#dc3545",
            AlertSeverity::Warning => "#ffc107",
            AlertSeverity::Info => "#17a2b8",
        };

        let emoji = match alert.severity {
            AlertSeverity::Critical => "🚨",
            AlertSeverity::Warning => "⚠️",
            AlertSeverity::Info => "ℹ️",
        };

        let payload = serde_json::json!({
            "attachments": [{
                "color": color,
                "title": format!("{} {} Alert", emoji, format!("{:?}", alert.severity)),
                "text": alert.message,
                "fields": [
                    {
                        "title": "Component",
                        "value": alert.component,
                        "short": true
                    },
                    {
                        "title": "Metric",
                        "value": alert.metric,
                        "short": true
                    },
                    {
                        "title": "Current Value",
                        "value": format!("{:.2}", alert.current_value),
                        "short": true
                    },
                    {
                        "title": "Threshold",
                        "value": format!("{:.2}", alert.threshold),
                        "short": true
                    }
                ],
                "footer": "Video RSS Aggregator Monitoring",
                "ts": alert.timestamp.timestamp()
            }]
        });

        info!("Slack payload: {}", payload);
        Ok(())
    }

    async fn send_discord_notification(&self, channel: &NotificationChannel, alert: &Alert) -> Result<()> {
        // In a real implementation, this would send to Discord webhook
        info!("Sending Discord notification for alert {} via {}", alert.id, channel.name);

        let color = match alert.severity {
            AlertSeverity::Critical => 0xdc3545,
            AlertSeverity::Warning => 0xffc107,
            AlertSeverity::Info => 0x17a2b8,
        };

        let payload = serde_json::json!({
            "embeds": [{
                "title": format!("{:?} Alert - {}", alert.severity, alert.component),
                "description": alert.message,
                "color": color,
                "fields": [
                    {
                        "name": "Metric",
                        "value": alert.metric,
                        "inline": true
                    },
                    {
                        "name": "Current Value",
                        "value": format!("{:.2}", alert.current_value),
                        "inline": true
                    },
                    {
                        "name": "Threshold",
                        "value": format!("{:.2}", alert.threshold),
                        "inline": true
                    }
                ],
                "timestamp": alert.timestamp.to_rfc3339(),
                "footer": {
                    "text": "Video RSS Aggregator Monitoring"
                }
            }]
        });

        info!("Discord payload: {}", payload);
        Ok(())
    }

    async fn send_webhook_notification(&self, channel: &NotificationChannel, alert: &Alert) -> Result<()> {
        // In a real implementation, this would send HTTP POST to webhook URL
        info!("Sending webhook notification for alert {} via {}", alert.id, channel.name);

        let payload = serde_json::json!({
            "alert_id": alert.id,
            "timestamp": alert.timestamp,
            "severity": format!("{:?}", alert.severity),
            "component": alert.component,
            "metric": alert.metric,
            "message": alert.message,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "status": format!("{:?}", alert.status)
        });

        info!("Webhook payload: {}", payload);
        Ok(())
    }

    async fn send_pagerduty_notification(&self, channel: &NotificationChannel, alert: &Alert) -> Result<()> {
        // In a real implementation, this would use PagerDuty Events API
        info!("Sending PagerDuty notification for alert {} via {}", alert.id, channel.name);

        let event_action = match alert.status {
            AlertStatus::Active => "trigger",
            AlertStatus::Resolved => "resolve",
            _ => "trigger",
        };

        let payload = serde_json::json!({
            "routing_key": channel.config.get("routing_key").unwrap_or(&"".to_string()),
            "event_action": event_action,
            "dedup_key": alert.id,
            "payload": {
                "summary": alert.message,
                "source": alert.component,
                "severity": match alert.severity {
                    AlertSeverity::Critical => "critical",
                    AlertSeverity::Warning => "warning",
                    AlertSeverity::Info => "info",
                },
                "component": alert.component,
                "group": "video-rss-aggregator",
                "class": alert.metric,
                "custom_details": {
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "metric": alert.metric
                }
            }
        });

        info!("PagerDuty payload: {}", payload);
        Ok(())
    }

    async fn check_escalations(&mut self) -> Result<()> {
        let now = Utc::now();

        for alert in self.active_alerts.values_mut() {
            if matches!(alert.status, AlertStatus::Active) {
                // Find the rule for this alert
                if let Some(rule) = self.alert_rules.iter().find(|r|
                    alert.component.contains(&r.component.replace("*", "")) &&
                    r.metric == alert.metric
                ) {
                    let escalation_time = chrono::Duration::minutes(rule.escalation_minutes as i64);

                    if now - alert.timestamp > escalation_time && alert.escalation_level == 0 {
                        alert.escalation_level = 1;

                        warn!("Escalating alert {} - no resolution after {} minutes",
                            alert.id, rule.escalation_minutes);

                        // Send escalation notification
                        let mut escalated_alert = alert.clone();
                        escalated_alert.message = format!("ESCALATED: {}", alert.message);

                        self.send_notifications(&escalated_alert).await?;
                    }
                }
            }
        }

        Ok(())
    }

    async fn resolve_alerts(&mut self, current_metrics: &HashMap<String, HashMap<String, f64>>) -> Result<()> {
        let mut alerts_to_resolve = Vec::new();

        for (alert_key, alert) in &self.active_alerts {
            if !matches!(alert.status, AlertStatus::Active) {
                continue;
            }

            // Check if the condition is no longer met
            if let Some(component_metrics) = current_metrics.get(&alert.component) {
                if let Some(&current_value) = component_metrics.get(&alert.metric) {
                    // Find the rule for this alert
                    if let Some(rule) = self.alert_rules.iter().find(|r|
                        r.name == alert_key.split('_').skip(1).collect::<Vec<_>>().join("_")
                    ) {
                        let condition_met = self.evaluate_condition(
                            &rule.condition,
                            current_value,
                            &alert.component,
                            &alert.metric
                        ).await?;

                        if !condition_met {
                            // Wait for a grace period before resolving
                            let grace_period = chrono::Duration::minutes(5);
                            if Utc::now() - alert.timestamp > grace_period {
                                alerts_to_resolve.push(alert_key.clone());
                            }
                        }
                    }
                }
            }
        }

        // Resolve alerts
        for alert_key in alerts_to_resolve {
            if let Some(mut alert) = self.active_alerts.remove(&alert_key) {
                alert.status = AlertStatus::Resolved;
                alert.resolution_time = Some(Utc::now());

                info!("Alert resolved: {} - {}", alert.id, alert.message);

                // Send resolution notification
                let mut resolved_alert = alert.clone();
                resolved_alert.message = format!("RESOLVED: {}", alert.message);

                self.send_notifications(&resolved_alert).await?;

                // Update history
                if let Some(history_alert) = self.alert_history.iter_mut().find(|a| a.id == alert.id) {
                    history_alert.status = AlertStatus::Resolved;
                    history_alert.resolution_time = alert.resolution_time;
                }
            }
        }

        Ok(())
    }

    pub async fn acknowledge_alert(&mut self, alert_id: &str) -> Result<()> {
        for alert in self.active_alerts.values_mut() {
            if alert.id == alert_id {
                alert.status = AlertStatus::Acknowledged;
                info!("Alert acknowledged: {}", alert_id);
                return Ok(());
            }
        }

        Err(anyhow::anyhow!("Alert not found: {}", alert_id))
    }

    pub async fn suppress_alert(&mut self, alert_id: &str, duration_minutes: i32) -> Result<()> {
        for alert in self.active_alerts.values_mut() {
            if alert.id == alert_id {
                alert.status = AlertStatus::Suppressed;
                info!("Alert suppressed for {} minutes: {}", duration_minutes, alert_id);

                // In a real implementation, would set up automatic unsuppression
                let alert_id_clone = alert_id.to_string();
                tokio::spawn(async move {
                    sleep(Duration::from_secs((duration_minutes * 60) as u64)).await;
                    info!("Suppression expired for alert: {}", alert_id_clone);
                    // Would automatically reactivate alert if condition still met
                });

                return Ok(());
            }
        }

        Err(anyhow::anyhow!("Alert not found: {}", alert_id))
    }

    pub fn get_active_alerts(&self) -> Vec<&Alert> {
        self.active_alerts.values().collect()
    }

    pub fn get_alert_history(&self, hours: i32) -> Vec<&Alert> {
        let cutoff = Utc::now() - chrono::Duration::hours(hours as i64);
        self.alert_history
            .iter()
            .filter(|alert| alert.timestamp > cutoff)
            .collect()
    }

    pub fn get_alert_statistics(&self) -> HashMap<String, i32> {
        let mut stats = HashMap::new();

        let active_count = self.active_alerts.len() as i32;
        stats.insert("active_alerts".to_string(), active_count);

        let critical_count = self.active_alerts.values()
            .filter(|a| matches!(a.severity, AlertSeverity::Critical))
            .count() as i32;
        stats.insert("critical_alerts".to_string(), critical_count);

        let warning_count = self.active_alerts.values()
            .filter(|a| matches!(a.severity, AlertSeverity::Warning))
            .count() as i32;
        stats.insert("warning_alerts".to_string(), warning_count);

        let last_24h = Utc::now() - chrono::Duration::hours(24);
        let recent_count = self.alert_history.iter()
            .filter(|a| a.timestamp > last_24h)
            .count() as i32;
        stats.insert("alerts_last_24h".to_string(), recent_count);

        stats
    }
}