"""Topically-unrelated documents.

These never answer an eval question. They exist to make retrieval realistic:
unanswerable questions still retrieve *something*, so a naive baseline is
tempted to answer from weakly-related context.
"""
from __future__ import annotations

from typing import Dict, List

DOCUMENTS: List[Dict] = [
    {
        "doc_id": "wiki_office_wifi_2026_01",
        "title": "Office WiFi Setup",
        "source_type": "wiki",
        "created_at": "2026-01-08",
        "authority_score": 0.60,
        "status": "active",
        "text": "Connect to the CORP-WIFI network. The guest network is GUEST-WIFI and rotates its password weekly. Contact IT for the corporate certificate.",
        "metadata": {"topic": "office_it", "department": "it"},
    },
    {
        "doc_id": "email_holiday_schedule_2026_01",
        "title": "2026 Company Holidays",
        "source_type": "email",
        "created_at": "2026-01-02",
        "authority_score": 0.65,
        "status": "active",
        "text": "The company will be closed on New Year's Day, Memorial Day, Independence Day, Thanksgiving (two days), and the week between Christmas and New Year's.",
        "metadata": {"topic": "holidays", "department": "people_ops"},
    },
    {
        "doc_id": "product_doc_sso_2026_03",
        "title": "Single Sign-On Setup",
        "source_type": "product_doc",
        "created_at": "2026-03-05",
        "authority_score": 0.85,
        "status": "active",
        "text": "We support SAML and OIDC single sign-on. Enterprise customers can configure SSO from the admin console under Security > SSO.",
        "metadata": {"topic": "sso", "department": "product"},
    },
    {
        "doc_id": "product_doc_webhooks_2026_04",
        "title": "Webhooks Overview",
        "source_type": "product_doc",
        "created_at": "2026-04-02",
        "authority_score": 0.85,
        "status": "active",
        "text": "Webhooks deliver event notifications via HTTPS POST. Failed deliveries are retried with exponential backoff for up to 24 hours. Configure endpoints in the developer settings.",
        "metadata": {"topic": "webhooks", "department": "product"},
    },
    {
        "doc_id": "product_doc_export_2026_05",
        "title": "Data Export Feature",
        "source_type": "product_doc",
        "created_at": "2026-05-01",
        "authority_score": 0.85,
        "status": "active",
        "text": "Customers can export their data as CSV or JSON from Account > Export. Large exports are emailed as a download link when ready.",
        "metadata": {"topic": "data_export", "department": "product"},
    },
    {
        "doc_id": "slack_general_chatter_2026_04",
        "title": "Slack #general lunch thread",
        "source_type": "slack",
        "created_at": "2026-04-03",
        "authority_score": 0.40,
        "status": "active",
        "text": "Reminder the new espresso machine is in the third floor kitchen. Please rinse the portafilter after use.",
        "metadata": {"topic": "office", "department": "people_ops"},
    },
    {
        "doc_id": "product_doc_audit_logs_2026_04",
        "title": "Audit Logs",
        "source_type": "product_doc",
        "created_at": "2026-04-20",
        "authority_score": 0.85,
        "status": "active",
        "text": "Enterprise customers have access to audit logs that record admin actions, login events, and data access. Logs can be streamed to a SIEM via the audit log API.",
        "metadata": {"topic": "audit_logs", "department": "product"},
    },
    {
        "doc_id": "wiki_meeting_norms_2026_02",
        "title": "Meeting Norms",
        "source_type": "wiki",
        "created_at": "2026-02-22",
        "authority_score": 0.70,
        "status": "active",
        "text": "Every meeting should have an agenda and a note-taker. Default meeting length is 25 or 50 minutes to leave buffer time. No-meeting Wednesdays are encouraged.",
        "metadata": {"topic": "meetings", "department": "people_ops"},
    },
    {
        "doc_id": "product_doc_mobile_2026_05",
        "title": "Mobile App Overview",
        "source_type": "product_doc",
        "created_at": "2026-05-05",
        "authority_score": 0.85,
        "status": "active",
        "text": "The mobile app is available for iOS and Android. It supports push notifications, offline viewing of recent items, and biometric login.",
        "metadata": {"topic": "mobile", "department": "product"},
    },
]
