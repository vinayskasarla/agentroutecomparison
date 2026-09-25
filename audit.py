"""Audit trail: one JSON line per user action on stdout. App Runner ships stdout to CloudWatch Logs, where
`log_type = "audit"` separates these from access logs (see infra/aws/cloudwatch.tf for queries and a dashboard).

Identity comes from the SSO layer in front of the app, via request headers. Only trust these headers when the
app is reachable solely through that layer (otherwise anyone can set them).
"""
import base64
import datetime
import json
import os
import sys

from pii import redact

USER_HEADERS = [h.strip().lower() for h in os.environ.get(
    "AUDIT_USER_HEADERS",
    "x-amzn-oidc-data,x-amzn-oidc-identity,x-forwarded-email,x-auth-request-email,x-forwarded-user",
).split(",") if h.strip()]
REDACT_PII = os.environ.get("AUDIT_REDACT_PII", "true").lower() != "false"
MAX_TEXT = 2000
ALLOWED_CLIENT_EVENTS = {"adr_downloaded", "adjust_opened", "example_used", "knowledge_opened", "details_opened"}


def identity(headers) -> dict:
    """Who is making the request, from the SSO proxy's headers."""
    for h in USER_HEADERS:
        value = headers.get(h)
        if not value:
            continue
        if h.endswith("oidc-data") and value.count(".") == 2:  # ALB/Cognito OIDC: a JWT carrying the user's claims
            try:
                part = value.split(".")[1]
                claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
                return {"id": claims.get("sub") or claims.get("email"), "email": claims.get("email"),
                        "name": claims.get("name"), "via": h}
            except (ValueError, json.JSONDecodeError):
                continue
        return {"id": value, "email": value if "@" in value else None, "via": h}
    return {"id": "anonymous", "email": None, "via": None}


def _clean(value):
    if isinstance(value, str):
        text = redact(value)[0] if REDACT_PII else value
        return text[:MAX_TEXT]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value][:100]
    return value


def log(event: str, request=None, **data):
    rec = {"log_type": "audit", "event": event,
           "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")}
    if request is not None:
        rec["user"] = identity(request.headers)
        rec["session_id"] = request.cookies.get("plab_sid") or getattr(request.state, "session_id", None)
        rec["request_id"] = getattr(request.state, "request_id", None)
        rec["ip"] = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
                     or (request.client.host if request.client else None))
        rec["user_agent"] = request.headers.get("user-agent", "")[:200]
        rec["path"] = request.url.path
    rec["data"] = _clean(data)
    sys.stdout.write(json.dumps(rec, default=str) + "\n")
    sys.stdout.flush()
