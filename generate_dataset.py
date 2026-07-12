"""
Dataset generation for AI-Driven Phishing Email Detection using NLP
Generates a realistic synthetic dataset of phishing and legitimate emails
with both text content and metadata features (URLs, sender domain patterns).
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------
# Building blocks for PHISHING emails
# ---------------------------------------------------------------
phishing_subjects = [
    "Urgent: Your Account Will Be Suspended",
    "Action Required: Verify Your Identity Now",
    "Security Alert: Unusual Sign-in Activity Detected",
    "Your Payment Failed - Update Billing Information",
    "You Have Won a Prize! Claim Now",
    "Final Notice: Your Account Has Been Locked",
    "Immediate Action Needed: Confirm Your Password",
    "Your Package Could Not Be Delivered",
    "Important: Your Refund Is Pending Approval",
    "Warning: Suspicious Login From New Device",
    "Your Subscription Is About to Expire",
    "Verify Your Bank Details to Avoid Suspension",
    "Congratulations! You Are Selected for a Reward",
    "Your Invoice Is Overdue - Pay Immediately",
    "Confirm Your Email to Restore Access",
]

phishing_bodies = [
    "Dear Customer, we detected unusual activity on your account. Click the link below immediately to verify your identity or your account will be permanently suspended within 24 hours. {url}",
    "Your account has been flagged for suspicious activity. To avoid permanent suspension, please confirm your login credentials at {url} within 12 hours.",
    "Congratulations! You have been selected to receive a cash reward of $1000. Claim your prize now by clicking {url} and entering your bank details.",
    "We were unable to process your recent payment. Please update your billing information immediately at {url} to avoid service interruption.",
    "This is an urgent security notice. Someone tried to access your account from an unrecognized device. Verify it was you at {url} or your account will be locked.",
    "Your package delivery has failed due to an incomplete address. Click {url} now and pay a small redelivery fee to reschedule.",
    "Act now! Your subscription expires today. Renew immediately at {url} to avoid losing access to your account permanently.",
    "Dear user, our system detected a policy violation on your account. Verify your information at {url} within 24 hours to prevent permanent closure.",
    "You have an unclaimed refund of $250.00 waiting. Submit your bank account details at {url} to receive your refund today.",
    "Final warning: your mailbox storage is full and incoming emails will be rejected. Click {url} immediately to upgrade for free.",
]

phishing_urls = [
    "http://secure-login-verify.com/account",
    "http://paypal-security-update.net/confirm",
    "http://bank0famerica-alert.com/verify",
    "http://appleid-confirm-account.xyz/login",
    "http://amaz0n-rewards-claim.info/prize",
    "http://microsoft-support-alert.co/reset",
    "http://192.168.44.21/verify-account",
    "http://update-your-info-now.tk/secure",
    "http://netflix-billing-update.ru/pay",
    "http://irs-tax-refund-claim.top/form",
]

phishing_senders = [
    "security@paypa1-support.com",
    "no-reply@appleid-verify.net",
    "alerts@bank0famerica.co",
    "support@amaz0n-rewards.info",
    "service@microsoft-alerts.xyz",
    "billing@netflix-update.ru",
    "team@irs-refunds.top",
    "admin@account-verify-center.com",
]

# ---------------------------------------------------------------
# Building blocks for LEGITIMATE emails
# ---------------------------------------------------------------
legit_subjects = [
    "Meeting Rescheduled to 3 PM Tomorrow",
    "Your Monthly Newsletter is Here",
    "Project Update: Sprint Review Notes",
    "Invoice #4521 Attached",
    "Weekly Team Sync Agenda",
    "Welcome to Our Service!",
    "Your Order Has Shipped",
    "Reminder: Submit Your Timesheet",
    "Notes from Today's Client Call",
    "Thank You for Your Recent Purchase",
    "Upcoming Webinar: AI in Healthcare",
    "Your Flight Itinerary Confirmation",
    "Quarterly Performance Report Attached",
    "Congratulations on Your Work Anniversary",
    "New Blog Post: Tips for Remote Work",
]

legit_bodies = [
    "Hi team, just a quick note that our meeting tomorrow has been moved to 3 PM in Conference Room B. Let me know if this conflicts with your schedule.",
    "Hello, here is our monthly newsletter with updates on new features, upcoming events, and customer stories. Read more on our blog at {url}.",
    "Hi all, sharing the notes from today's sprint review. We completed 8 out of 10 planned stories and identified two blockers for next sprint.",
    "Please find attached invoice #4521 for services rendered in June. Payment is due within 30 days as per our agreement.",
    "Hi everyone, attached is the agenda for our weekly team sync. Please review before the meeting and add any topics you'd like to discuss.",
    "Welcome aboard! We're excited to have you as part of our community. Here's a quick guide to help you get started with your new account.",
    "Good news! Your order #78421 has shipped and is on its way. You can track your shipment using the link {url}.",
    "Hi, this is a friendly reminder to submit your timesheet for this week by Friday 5 PM. Thanks for your cooperation.",
    "Hi team, sharing my notes from today's call with the client. They are happy with the current progress and requested a demo next week.",
    "Thank you for your recent purchase! Your receipt is attached. If you have any questions, feel free to reach out to our support team.",
]

legit_urls = [
    "https://company-blog.com/newsletter",
    "https://shipping.fedex.com/track",
    "https://docs.google.com/spreadsheets/d/abc123",
    "https://zoom.us/j/123456789",
    "https://github.com/company/project",
]

legit_senders = [
    "hr@company.com",
    "team-lead@company.com",
    "billing@vendor.com",
    "notifications@github.com",
    "no-reply@zoom.us",
    "support@fedex.com",
    "newsletter@company.com",
    "manager@company.com",
]


GREETINGS = ["Dear Customer,", "Hi there,", "Hello,", "Dear User,", "Hi,", ""]
CLOSINGS = ["Best regards,\nSupport Team", "Thanks,\nThe Team", "Sincerely,\nCustomer Service",
            "Regards,\nAccount Services", "Thank you,\nTeam", ""]
LEGIT_CLOSINGS = ["Best,\nSarah", "Thanks,\nRaj", "Cheers,\nPriya", "Regards,\nAdmin Team",
                  "Thank you,\nHR", "Best regards,\nProject Team"]

TYPO_MAP = {"account": "accont", "verify": "verifyy", "immediately": "immediatly",
            "your": "yur", "click": "clik", "security": "securrity"}


def add_typo_noise(text, prob=0.12):
    """Occasionally introduce a light typo to mimic real-world noisy text."""
    if random.random() < prob:
        for k, v in TYPO_MAP.items():
            if k in text and random.random() < 0.5:
                text = text.replace(k, v, 1)
                break
    return text


def make_phishing_email():
    subject = random.choice(phishing_subjects)
    body_template = random.choice(phishing_bodies)
    url = random.choice(phishing_urls)
    body = body_template.format(url=url)
    greeting = random.choice(GREETINGS)
    closing = random.choice(CLOSINGS)
    full_body = " ".join([p for p in [greeting, body, closing] if p])
    full_body = add_typo_noise(full_body)
    sender = random.choice(phishing_senders)

    # Some phishing emails are more subtle -- fewer urls, softer tone
    subtle = random.random() < 0.15
    n_urls = random.choice([0, 1]) if subtle else random.choice([1, 1, 2, 3])
    urgent_words = ["urgent", "immediately", "verify", "suspend", "click",
                     "confirm", "act now", "final notice", "alert", "locked"]
    urgency_count = sum(full_body.lower().count(w) for w in urgent_words)
    return {
        "subject": subject,
        "body": full_body,
        "sender": sender,
        "num_urls": n_urls,
        "has_suspicious_domain": 1,
        "urgency_word_count": urgency_count,
        "label": 1,  # phishing
    }


def make_legit_email():
    subject = random.choice(legit_subjects)
    body_template = random.choice(legit_bodies)
    url = random.choice(legit_urls)
    body = body_template.format(url=url) if "{url}" in body_template else body_template
    greeting = random.choice(GREETINGS)
    closing = random.choice(LEGIT_CLOSINGS)
    full_body = " ".join([p for p in [greeting, body, closing] if p])
    full_body = add_typo_noise(full_body, prob=0.05)
    sender = random.choice(legit_senders)

    # A minority of legitimate emails legitimately use urgency-adjacent words
    # (e.g. "please confirm your timesheet") to create realistic class
    # overlap rather than a perfectly separable synthetic set.
    if random.random() < 0.12:
        full_body += " Please confirm receipt of this email at your earliest convenience."
    if random.random() < 0.08:
        full_body = "Urgent: " + full_body

    n_urls = random.choice([0, 0, 1])
    urgent_words = ["urgent", "immediately", "verify", "suspend", "click",
                     "confirm", "act now", "final notice", "alert", "locked"]
    urgency_count = sum(full_body.lower().count(w) for w in urgent_words)
    return {
        "subject": subject,
        "body": full_body,
        "sender": sender,
        "num_urls": n_urls,
        "has_suspicious_domain": 0,
        "urgency_word_count": urgency_count,
        "label": 0,  # legitimate
    }


def generate_dataset(n_per_class=500):
    rows = []
    for _ in range(n_per_class):
        rows.append(make_phishing_email())
    for _ in range(n_per_class):
        rows.append(make_legit_email())
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "email_id", range(1, len(df) + 1))
    return df


if __name__ == "__main__":
    df = generate_dataset(n_per_class=650)
    df.to_csv("/home/claude/phishing_project/phishing_email_dataset_raw.csv", index=False)
    print(df.shape)
    print(df["label"].value_counts())
    print(df.head())
