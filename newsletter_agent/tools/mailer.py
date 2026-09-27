"""Email tool: simulate sending the newsletter.

Builds a real multipart MIME message (plain text + HTML alternative) exactly as it would
be handed to an SMTP server or an email API, then writes it to an outbox folder instead
of transmitting it. The `.eml` file opens in any mail client.
"""

from __future__ import annotations

import json
from email.message import EmailMessage
from email.utils import format_datetime, formataddr, make_msgid
from pathlib import Path

from pydantic import BaseModel

from newsletter_agent.config import Settings
from newsletter_agent.models import DeliveryReceipt, RenderedNewsletter
from newsletter_agent.tools.renderer import UNSUBSCRIBE_URL
from newsletter_agent.utils import slugify, utc_now


class Subscriber(BaseModel):
    name: str
    email: str


def load_subscribers(path: Path) -> list[Subscriber]:
    if not path.exists():
        return []
    return [Subscriber(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def build_email(rendered: RenderedNewsletter, settings: Settings, recipients: int) -> EmailMessage:
    domain = settings.sender_email.rsplit("@", 1)[-1]
    message = EmailMessage()
    message["Subject"] = rendered.subject
    message["From"] = formataddr((settings.sender_name, settings.sender_email))
    # Subscribers are BCC'd, as newsletter platforms do, so no addresses leak.
    message["To"] = "undisclosed-recipients:;"
    message["Date"] = format_datetime(utc_now())
    message["Message-ID"] = make_msgid(domain=domain)
    message["List-Unsubscribe"] = f"<{UNSUBSCRIBE_URL}>"
    message["X-Newsletter-Recipients"] = str(recipients)
    message.set_content(rendered.text)
    message.add_alternative(rendered.html, subtype="html")
    return message


def send_newsletter(rendered: RenderedNewsletter, settings: Settings) -> DeliveryReceipt:
    """'Send' the issue to every subscriber by writing it to the outbox."""
    subscribers = load_subscribers(settings.subscribers_file)
    sent_at = utc_now()
    out_dir = settings.outbox_dir / f"{sent_at:%Y%m%d-%H%M%S}-{slugify(settings.newsletter_name)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    message = build_email(rendered, settings, recipients=len(subscribers))
    files = {
        "html": out_dir / "newsletter.html",
        "markdown": out_dir / "newsletter.md",
        "eml": out_dir / "newsletter.eml",
        "receipt": out_dir / "delivery.json",
    }
    files["html"].write_text(rendered.html, encoding="utf-8")
    files["markdown"].write_text(rendered.markdown, encoding="utf-8")
    files["eml"].write_bytes(message.as_bytes())

    receipt = DeliveryReceipt(
        message_id=message["Message-ID"],
        subject=rendered.subject,
        sender=message["From"],
        recipients=[s.email for s in subscribers],
        sent_at=sent_at,
        output_dir=str(out_dir),
        files={kind: str(path) for kind, path in files.items()},
    )
    files["receipt"].write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    return receipt
