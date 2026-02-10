"""
Alert Manager - Smart notification system

Sends alerts ONLY when you need to know:
- Circuit breaker triggered (action may be needed)
- Crisis mode activated (FYI - EA handling it)
- Manual restart required (action needed)
- Weekly summary (regular update)
- Significant milestones (celebrate wins!)

NOT spammy:
- Respects quiet hours (no alerts at 3 AM)
- Groups similar alerts (no alert storm)
- Priority-based (critical alerts get through immediately)
- Configurable (you control what you get)

Delivery methods:
- Email (primary)
- Telegram (optional - future)
- Log file (always)
"""

from enum import Enum
from typing import Optional, Dict, List
from datetime import datetime, time, timedelta
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from loguru import logger


class AlertLevel(Enum):
    """Alert priority levels"""
    INFO = "INFO"           # FYI, no action needed
    WARNING = "WARNING"     # Attention needed soon
    CRITICAL = "CRITICAL"   # Immediate action required


class AlertType(Enum):
    """Types of alerts"""
    # Protection system alerts
    CIRCUIT_BREAKER_DAILY = "circuit_breaker_daily"
    CIRCUIT_BREAKER_WEEKLY = "circuit_breaker_weekly"
    CIRCUIT_BREAKER_MONTHLY = "circuit_breaker_monthly"
    CRISIS_MODE_ACTIVATED = "crisis_mode_activated"
    CRISIS_MODE_RESOLVED = "crisis_mode_resolved"
    VOLATILITY_ELEVATED = "volatility_elevated"

    # Action required
    MANUAL_RESTART_REQUIRED = "manual_restart_required"
    RESUME_READY = "resume_ready"

    # Performance alerts
    STRATEGY_UNDERPERFORMING = "strategy_underperforming"
    LARGE_LOSS = "large_loss"
    LARGE_GAIN = "large_gain"

    # Regular updates
    WEEKLY_SUMMARY = "weekly_summary"
    MONTHLY_SUMMARY = "monthly_summary"

    # Milestones
    BALANCE_MILESTONE = "balance_milestone"
    PROFIT_MILESTONE = "profit_milestone"

    # System
    EA_STARTED = "ea_started"
    EA_STOPPED = "ea_stopped"
    ERROR = "error"


@dataclass
class Alert:
    """Single alert instance"""
    timestamp: datetime
    level: AlertLevel
    alert_type: AlertType
    subject: str
    message: str
    data: Optional[Dict] = None


class AlertManager:
    """
    Manages all notifications and alerts for the EA.

    Features:
    - Smart alerting (only important events)
    - Quiet hours (no 3 AM wake-ups)
    - Alert grouping (no spam)
    - Multiple delivery methods
    - Configurable thresholds
    """

    def __init__(
        self,
        email_enabled: bool = False,
        email_from: Optional[str] = None,
        email_to: Optional[str] = None,
        smtp_server: Optional[str] = None,
        smtp_port: int = 587,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,

        # Quiet hours (local time)
        quiet_start: time = time(22, 0),    # 10 PM
        quiet_end: time = time(7, 0),       # 7 AM

        # Alert settings
        log_file: Optional[Path] = None,
        alert_cooldown_minutes: int = 60,   # Min time between same alert type
    ):
        """
        Initialize alert manager.

        Args:
            email_enabled: Enable email alerts
            email_from: Sender email address
            email_to: Recipient email address
            smtp_server: SMTP server (e.g., smtp.gmail.com)
            smtp_port: SMTP port (usually 587)
            smtp_username: SMTP username
            smtp_password: SMTP password
            quiet_start: Start of quiet hours (no alerts)
            quiet_end: End of quiet hours
            log_file: File to log all alerts
            alert_cooldown_minutes: Cooldown between duplicate alerts
        """
        self.email_enabled = email_enabled
        self.email_from = email_from
        self.email_to = email_to
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password

        self.quiet_start = quiet_start
        self.quiet_end = quiet_end

        self.log_file = log_file
        self.alert_cooldown = timedelta(minutes=alert_cooldown_minutes)

        # Alert history
        self.alert_history: List[Alert] = []
        self.last_alert_time: Dict[AlertType, datetime] = {}

        # Statistics
        self.stats = {
            'total_alerts': 0,
            'alerts_sent': 0,
            'alerts_suppressed': 0,
            'emails_sent': 0,
            'emails_failed': 0,
        }

        logger.info(
            f"Initialized AlertManager: "
            f"Email={'enabled' if email_enabled else 'disabled'}, "
            f"Quiet={quiet_start.strftime('%H:%M')}-{quiet_end.strftime('%H:%M')}"
        )

    def send_alert(
        self,
        level: AlertLevel,
        alert_type: AlertType,
        subject: str,
        message: str,
        data: Optional[Dict] = None,
        force: bool = False
    ) -> bool:
        """
        Send an alert.

        Args:
            level: Alert priority level
            alert_type: Type of alert
            subject: Alert subject line
            message: Alert message body
            data: Additional data (for logging/debugging)
            force: Force send even in quiet hours / cooldown

        Returns:
            True if alert was sent, False if suppressed
        """
        now = datetime.now()

        # Create alert object
        alert = Alert(
            timestamp=now,
            level=level,
            alert_type=alert_type,
            subject=subject,
            message=message,
            data=data
        )

        # Add to history
        self.alert_history.append(alert)
        self.stats['total_alerts'] += 1

        # Log to file if configured
        if self.log_file:
            self._log_to_file(alert)

        # Check if we should suppress
        if not force:
            if self._should_suppress(alert):
                self.stats['alerts_suppressed'] += 1
                logger.debug(f"Alert suppressed: {alert_type.value} (cooldown/quiet hours)")
                return False

        # Log alert
        log_method = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.error,
        }[level]

        log_method(f"🔔 ALERT [{level.value}]: {subject}")
        log_method(f"   {message}")

        # Send via configured channels
        sent = False

        if self.email_enabled:
            sent = self._send_email(alert)

        if sent:
            self.stats['alerts_sent'] += 1
            self.last_alert_time[alert_type] = now

        return sent

    def _should_suppress(self, alert: Alert) -> bool:
        """Check if alert should be suppressed."""

        # Never suppress CRITICAL alerts
        if alert.level == AlertLevel.CRITICAL:
            return False

        # Check quiet hours (only for INFO/WARNING)
        if self._is_quiet_hours():
            return True

        # Check cooldown (prevent alert spam)
        if alert.alert_type in self.last_alert_time:
            time_since_last = alert.timestamp - self.last_alert_time[alert.alert_type]
            if time_since_last < self.alert_cooldown:
                return True

        return False

    def _is_quiet_hours(self) -> bool:
        """Check if current time is in quiet hours."""
        now = datetime.now().time()

        if self.quiet_start < self.quiet_end:
            # Normal case: 22:00 to 07:00
            return self.quiet_start <= now or now <= self.quiet_end
        else:
            # Crosses midnight: 23:00 to 01:00
            return self.quiet_start <= now <= self.quiet_end

        return False

    def _send_email(self, alert: Alert) -> bool:
        """Send alert via email."""

        if not all([self.email_from, self.email_to, self.smtp_server]):
            logger.debug("Email not configured, skipping")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.level.value}] {alert.subject}"
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Date'] = alert.timestamp.strftime('%a, %d %b %Y %H:%M:%S %z')

            # Create plain text version
            text_body = self._format_email_body(alert, html=False)
            text_part = MIMEText(text_body, 'plain')
            msg.attach(text_part)

            # Create HTML version
            html_body = self._format_email_body(alert, html=True)
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            self.stats['emails_sent'] += 1
            logger.info(f"✉️ Email sent: {alert.subject}")
            return True

        except Exception as e:
            self.stats['emails_failed'] += 1
            logger.error(f"Failed to send email: {e}")
            return False

    def _format_email_body(self, alert: Alert, html: bool = False) -> str:
        """Format email body."""

        if html:
            # HTML version with styling
            color = {
                AlertLevel.INFO: '#4A90E2',
                AlertLevel.WARNING: '#F5A623',
                AlertLevel.CRITICAL: '#D0021B',
            }[alert.level]

            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                <div style="background: {color}; color: white; padding: 20px; border-radius: 5px;">
                    <h2 style="margin: 0;">Smart Grid EA Alert</h2>
                    <p style="margin: 5px 0 0 0;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div style="padding: 20px; background: #f9f9f9; margin-top: 20px; border-radius: 5px;">
                    <h3 style="margin-top: 0; color: {color};">{alert.subject}</h3>
                    <p style="white-space: pre-wrap;">{alert.message}</p>
                </div>

                {self._format_data_html(alert.data) if alert.data else ''}

                <div style="margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 5px; font-size: 12px;">
                    <p style="margin: 0;"><strong>Alert Type:</strong> {alert.alert_type.value}</p>
                    <p style="margin: 5px 0 0 0;"><strong>Priority:</strong> {alert.level.value}</p>
                </div>

                <div style="margin-top: 30px; text-align: center; color: #999; font-size: 11px;">
                    <p>This is an automated alert from your Smart Grid EA trading system.</p>
                </div>
            </body>
            </html>
            """
        else:
            # Plain text version
            body = f"""
Smart Grid EA Alert
{'=' * 60}

Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Level: {alert.level.value}
Type: {alert.alert_type.value}

Subject: {alert.subject}

{alert.message}

{self._format_data_text(alert.data) if alert.data else ''}

---
This is an automated alert from your Smart Grid EA trading system.
            """

        return body

    def _format_data_html(self, data: Dict) -> str:
        """Format data dictionary as HTML."""
        if not data:
            return ""

        rows = []
        for key, value in data.items():
            rows.append(f"<tr><td><strong>{key}:</strong></td><td>{value}</td></tr>")

        return f"""
        <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 5px;">
            <h4 style="margin-top: 0;">Additional Details</h4>
            <table style="width: 100%; border-collapse: collapse;">
                {''.join(rows)}
            </table>
        </div>
        """

    def _format_data_text(self, data: Dict) -> str:
        """Format data dictionary as plain text."""
        if not data:
            return ""

        lines = ["\nAdditional Details:", "-" * 40]
        for key, value in data.items():
            lines.append(f"{key}: {value}")

        return "\n".join(lines)

    def _log_to_file(self, alert: Alert):
        """Log alert to file."""
        if not self.log_file:
            return

        try:
            # Ensure parent directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            # Append to log file
            with open(self.log_file, 'a') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"[{alert.timestamp.isoformat()}] {alert.level.value} - {alert.alert_type.value}\n")
                f.write(f"Subject: {alert.subject}\n")
                f.write(f"Message:\n{alert.message}\n")
                if alert.data:
                    f.write(f"Data: {alert.data}\n")
        except Exception as e:
            logger.error(f"Failed to log alert to file: {e}")

    # === Convenience Methods for Common Alerts ===

    def alert_circuit_breaker_triggered(
        self,
        period: str,
        loss_pct: float,
        balance: float,
        pause_duration: str
    ):
        """Alert that circuit breaker was triggered."""
        level = AlertLevel.CRITICAL if period == 'monthly' else AlertLevel.WARNING

        alert_type = {
            'daily': AlertType.CIRCUIT_BREAKER_DAILY,
            'weekly': AlertType.CIRCUIT_BREAKER_WEEKLY,
            'monthly': AlertType.CIRCUIT_BREAKER_MONTHLY,
        }[period]

        subject = f"Circuit Breaker Triggered ({period.upper()})"
        message = f"""
Your trading EA has automatically paused due to {period} loss limits.

Loss: {loss_pct:.1f}%
Current Balance: ${balance:.2f}
Pause Duration: {pause_duration}

{
"ACTION REQUIRED: Manual restart needed after reviewing performance." if period == 'monthly'
else "The EA will auto-resume after the pause period unless issues persist."
}

This is a protective measure to prevent further losses during unfavorable conditions.
        """.strip()

        self.send_alert(
            level=level,
            alert_type=alert_type,
            subject=subject,
            message=message,
            data={
                'period': period,
                'loss_percentage': f"{loss_pct:.2f}%",
                'current_balance': f"${balance:.2f}",
                'pause_duration': pause_duration,
            }
        )

    def alert_crisis_mode(self, atr_ratio: float, balance: float):
        """Alert that crisis mode was activated."""
        self.send_alert(
            level=AlertLevel.WARNING,
            alert_type=AlertType.CRISIS_MODE_ACTIVATED,
            subject="Crisis Mode Activated - Trading Paused",
            message=f"""
The EA has detected extreme market volatility and automatically entered crisis mode.

ATR Ratio: {atr_ratio:.2f}x normal
Current Balance: ${balance:.2f}

All positions have been closed and trading is paused for at least 7 days.

This is a protective measure during market crashes like COVID-19.
The EA will monitor volatility and resume when safe.

NO ACTION NEEDED - The system is handling this automatically.
            """.strip(),
            data={
                'atr_ratio': f"{atr_ratio:.2f}x",
                'balance': f"${balance:.2f}",
                'cooldown_days': 7,
            }
        )

    def alert_manual_restart_required(self, reason: str, balance: float):
        """Alert that manual restart is required."""
        self.send_alert(
            level=AlertLevel.CRITICAL,
            alert_type=AlertType.MANUAL_RESTART_REQUIRED,
            subject="Manual Restart Required",
            message=f"""
ACTION REQUIRED: Your EA needs manual restart.

Reason: {reason}
Current Balance: ${balance:.2f}

Steps to restart:
1. Review the dashboard and recent performance
2. Check logs for any errors
3. If comfortable, click "Resume Trading" in dashboard
4. Or contact support if unsure

The EA will NOT resume automatically - this requires your approval.
            """.strip(),
            data={
                'reason': reason,
                'balance': f"${balance:.2f}",
            },
            force=True  # Always send this one
        )

    def alert_ready_to_resume(self, balance: float):
        """Alert that EA is ready to resume after pause."""
        self.send_alert(
            level=AlertLevel.INFO,
            alert_type=AlertType.RESUME_READY,
            subject="Ready to Resume Trading",
            message=f"""
Your EA is ready to resume trading after the pause period.

Current Balance: ${balance:.2f}
Status: All systems normal

To resume:
1. Check the dashboard
2. Review recent performance
3. Click "Resume Trading" if ready

Or the EA will auto-resume in the next check cycle.
            """.strip(),
            data={'balance': f"${balance:.2f}"}
        )

    def alert_weekly_summary(
        self,
        balance: float,
        week_pnl: float,
        week_pnl_pct: float,
        trades_this_week: int,
        status: str
    ):
        """Send weekly performance summary."""
        emoji = "📈" if week_pnl > 0 else "📉" if week_pnl < 0 else "➖"

        self.send_alert(
            level=AlertLevel.INFO,
            alert_type=AlertType.WEEKLY_SUMMARY,
            subject=f"Weekly Summary: {emoji} {week_pnl_pct:+.1f}%",
            message=f"""
Here's your weekly trading summary:

Balance: ${balance:.2f}
Week P&L: ${week_pnl:+.2f} ({week_pnl_pct:+.1f}%)
Trades: {trades_this_week}
Status: {status}

{
"Great week! Keep it up! 🎉" if week_pnl_pct > 5
else "Solid performance! 💪" if week_pnl_pct > 0
else "Minor setback, staying the course. 📊" if week_pnl_pct > -5
else "Tough week, but systems are protecting capital. 🛡️"
}

View detailed stats in your dashboard.
            """.strip(),
            data={
                'balance': f"${balance:.2f}",
                'week_pnl': f"${week_pnl:+.2f}",
                'week_pnl_pct': f"{week_pnl_pct:+.1f}%",
                'trades': trades_this_week,
                'status': status,
            }
        )

    def alert_balance_milestone(self, milestone: int, balance: float, roi: float):
        """Alert when reaching a balance milestone."""
        self.send_alert(
            level=AlertLevel.INFO,
            alert_type=AlertType.BALANCE_MILESTONE,
            subject=f"🎉 Milestone Reached: ${milestone}!",
            message=f"""
Congratulations! You've reached a significant milestone!

New Balance: ${balance:.2f}
Total ROI: {roi:.1f}%

Keep up the great work! Your disciplined approach is paying off.

{
"Next milestone: $" + str(milestone * 2) if milestone < 10000
else "You're building serious wealth! 💰"
}
            """.strip(),
            data={
                'milestone': f"${milestone}",
                'balance': f"${balance:.2f}",
                'roi': f"{roi:.1f}%",
            }
        )

    def get_statistics(self) -> Dict:
        """Get alert statistics."""
        return {
            **self.stats,
            'alert_history_count': len(self.alert_history),
            'unique_alert_types': len(set(a.alert_type for a in self.alert_history)),
        }

    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        """Get recent alerts."""
        return self.alert_history[-count:]

    def clear_history(self):
        """Clear alert history (for testing)."""
        self.alert_history = []
        self.last_alert_time = {}
        logger.info("Alert history cleared")
