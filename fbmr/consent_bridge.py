"""
Consent Verification Bridge for FBMR
====================================
Provides local, deterministic cryptographic consent verification
and audit ledger logging. Operates offline without requiring a live
Ethereum/Sepolia testnet connection or ETH gas.
"""

from enum import Enum
import hashlib
import json
import time
from typing import Dict, List, Optional


class ConsentStatus(str, Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


class ConsentVerificationBridge:
    """
    Simulates on-device consent management and records immutable audit events
    using SHA-256 hash chaining (mirroring SafeMind's ConsentRegistry.sol).
    """

    def __init__(self):
        # In-memory registry: client_id -> consent record
        self._registry: Dict[str, Dict] = {}
        # Audit log trail
        self._audit_trail: List[Dict] = []
        self._last_block_hash: str = "0" * 64

    def register_consent(
        self,
        client_id: str,
        college_id: str,
        status: ConsentStatus = ConsentStatus.ACTIVE,
        purpose: str = "academic_wellness_refinement",
    ) -> Dict:
        """Register or update a client's consent state."""
        timestamp = time.time()
        record = {
            "client_id": client_id,
            "college_id": college_id,
            "status": status.value,
            "purpose": purpose,
            "timestamp": timestamp,
        }
        self._registry[client_id] = record
        self.log_event(
            event_type="CONSENT_UPDATED",
            client_id=client_id,
            college_id=college_id,
            details={"status": status.value, "purpose": purpose},
        )
        return record

    def revoke_consent(self, client_id: str) -> bool:
        """Revoke a client's consent immediately."""
        if client_id in self._registry:
            self._registry[client_id]["status"] = ConsentStatus.REVOKED.value
            self.log_event(
                event_type="CONSENT_REVOKED",
                client_id=client_id,
                college_id=self._registry[client_id]["college_id"],
                details={"status": "REVOKED"},
            )
            return True
        return False

    def is_consent_valid(
        self,
        client_id: str,
        required_purpose: str = "academic_wellness_refinement",
    ) -> bool:
        """
        Verify whether the client currently has active consent for the given purpose.
        """
        record = self._registry.get(client_id)
        if not record:
            return False
        if record["status"] != ConsentStatus.ACTIVE.value:
            return False
        if record["purpose"] != required_purpose:
            return False
        return True

    def log_event(
        self,
        event_type: str,
        client_id: Optional[str] = None,
        college_id: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> str:
        """
        Append a deterministic event to the audit ledger with SHA-256 hash chaining.
        """
        timestamp = time.time()
        event = {
            "index": len(self._audit_trail),
            "timestamp": timestamp,
            "event_type": event_type,
            "client_id": client_id or "anonymous",
            "college_id": college_id or "federation",
            "details": details or {},
            "prev_hash": self._last_block_hash,
        }
        event_bytes = json.dumps(event, sort_keys=True).encode("utf-8")
        event_hash = hashlib.sha256(event_bytes).hexdigest()
        event["hash"] = event_hash
        self._last_block_hash = event_hash
        self._audit_trail.append(event)
        return event_hash

    def get_audit_trail(self) -> List[Dict]:
        """Return the immutable audit log list."""
        return list(self._audit_trail)
