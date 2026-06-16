"""People-ops / sales / legal policy documents.

Carries the conflict pattern (remote work, data retention), the
informal-vs-official pattern (Slack vs official policy), and several
clear-answer / partial-evidence policies.
"""
from __future__ import annotations

from typing import Dict, List

DOCUMENTS: List[Dict] = [
    # ---- refund policy (CLEAR + PARTIAL + informal Slack) ----
    {
        "doc_id": "official_refund_policy_2026_02",
        "title": "Customer Refund Policy",
        "source_type": "official_docs",
        "created_at": "2026-02-15",
        "authority_score": 0.95,
        "status": "active",
        "text": "Customers may request a full refund within 30 days of purchase. Refunds are processed within 5 business days to the original payment method. Refunds are not available for usage-based overage charges.",
        "metadata": {"topic": "refunds", "department": "support"},
    },
    {
        "doc_id": "slack_refund_chatter_2026_03",
        "title": "Slack #support thread on refunds",
        "source_type": "slack",
        "created_at": "2026-03-02",
        "authority_score": 0.40,
        "status": "active",
        "text": "Someone asked about refunds for annual plans. Not sure how proration works for annual plans honestly, we should ask finance. The 30-day window definitely applies though.",
        "metadata": {"topic": "refunds", "department": "support"},
    },
    # ---- remote work policy (CONFLICT: contemporary email vs wiki) ----
    {
        "doc_id": "email_remote_policy_2026_01",
        "title": "All-hands email: return to office",
        "source_type": "email",
        "created_at": "2026-01-20",
        "authority_score": 0.65,
        "status": "active",
        "text": "Effective February, all employees are expected in the office 3 days per week (Tuesday, Wednesday, Thursday).",
        "metadata": {"topic": "remote_work", "department": "people_ops"},
    },
    {
        "doc_id": "wiki_remote_policy_2026_01",
        "title": "Remote Work Policy (Wiki)",
        "source_type": "wiki",
        "created_at": "2026-01-22",
        "authority_score": 0.60,
        "status": "active",
        "text": "Our remote work policy allows employees to work fully remote. Office attendance is optional.",
        "metadata": {"topic": "remote_work", "department": "people_ops"},
    },
    # ---- data retention (CLEAR + CONFLICT: official vs Slack guess) ----
    {
        "doc_id": "official_data_retention_2026_02",
        "title": "Data Retention Policy",
        "source_type": "official_docs",
        "created_at": "2026-02-01",
        "authority_score": 0.95,
        "status": "active",
        "text": "Customer data is retained for 90 days after account deletion, after which it is permanently purged. Backups are retained for an additional 30 days.",
        "metadata": {"topic": "data_retention", "department": "legal"},
    },
    {
        "doc_id": "slack_data_retention_2026_02",
        "title": "Slack #legal note on retention",
        "source_type": "slack",
        "created_at": "2026-02-10",
        "authority_score": 0.40,
        "status": "active",
        "text": "I thought we kept customer data for 180 days after deletion? Pretty sure that's what we tell enterprise customers.",
        "metadata": {"topic": "data_retention", "department": "legal"},
    },
    # ---- onboarding laptop policy (CLEAR + PARTIAL) ----
    {
        "doc_id": "wiki_laptop_policy_2026_01",
        "title": "New Hire Laptop Policy",
        "source_type": "wiki",
        "created_at": "2026-01-05",
        "authority_score": 0.70,
        "status": "active",
        "text": "New engineering hires receive a 16-inch MacBook Pro. Non-engineering hires receive a 14-inch MacBook Air. Laptops are shipped to arrive on the first day.",
        "metadata": {"topic": "onboarding_laptop", "department": "it"},
    },
    {
        "doc_id": "jira_onboarding_ticket_2026_02",
        "title": "JIRA OPS-201: laptop shipping delays",
        "source_type": "jira",
        "created_at": "2026-02-12",
        "authority_score": 0.55,
        "status": "active",
        "text": "OPS-201: Several new hires reported laptops arriving 2-3 days late. Investigating vendor shipping SLA.",
        "metadata": {"topic": "onboarding_laptop", "department": "it"},
    },
    # ---- pricing approval (CLEAR + PARTIAL + informal Slack) ----
    {
        "doc_id": "official_pricing_approval_2026_03",
        "title": "Pricing Approval Process",
        "source_type": "official_docs",
        "created_at": "2026-03-01",
        "authority_score": 0.95,
        "status": "active",
        "text": "Any discount above 15% requires VP of Sales approval. Discounts above 30% require CFO approval. All custom pricing must be recorded in the deal desk system.",
        "metadata": {"topic": "pricing_approval", "department": "sales"},
    },
    {
        "doc_id": "slack_pricing_chatter_2026_03",
        "title": "Slack #sales discount question",
        "source_type": "slack",
        "created_at": "2026-03-20",
        "authority_score": 0.40,
        "status": "active",
        "text": "Customer is asking for a discount. I think anything big needs sign-off from a VP but I don't remember the exact threshold.",
        "metadata": {"topic": "pricing_approval", "department": "sales"},
    },
    # ---- PTO / benefits (CLEAR) ----
    {
        "doc_id": "wiki_vacation_policy_2026_01",
        "title": "Vacation / PTO Policy",
        "source_type": "wiki",
        "created_at": "2026-01-15",
        "authority_score": 0.70,
        "status": "active",
        "text": "Full-time employees accrue 20 days of paid time off per year. PTO must be requested at least one week in advance for blocks longer than 3 days.",
        "metadata": {"topic": "pto", "department": "people_ops"},
    },
    {
        "doc_id": "email_benefits_2026_01",
        "title": "2026 Benefits Enrollment",
        "source_type": "email",
        "created_at": "2026-01-12",
        "authority_score": 0.65,
        "status": "active",
        "text": "Open enrollment runs through the end of January. The company covers 90% of medical premiums and offers a 401k match up to 4% of salary.",
        "metadata": {"topic": "benefits", "department": "people_ops"},
    },
    {
        "doc_id": "email_expense_policy_2026_02",
        "title": "Expense Reimbursement Policy",
        "source_type": "email",
        "created_at": "2026-02-05",
        "authority_score": 0.65,
        "status": "active",
        "text": "Submit expenses within 30 days. Meals while traveling are reimbursed up to $75 per day. Receipts are required for any expense over $25.",
        "metadata": {"topic": "expenses", "department": "finance"},
    },
]
