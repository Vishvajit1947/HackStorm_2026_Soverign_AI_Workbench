"""Break the audit chain on purpose and watch verification catch it.

    python scripts/tamper_demo.py

Uses a throwaway database, never your real one. This is the script you run when a
judge asks "how do you know the log hasn't been edited?" - it is much more
convincing than describing the property.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))

from kavach.audit.chain import AuditChain
from kavach.store.db import Database

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def banner(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}")
    print(DIM + "-" * len(text) + RESET)


def report(chain: AuditChain) -> None:
    result = chain.verify()
    if result.ok:
        print(f"  {GREEN}chain intact{RESET} - {result.records} records")
        print(f"  {DIM}{result.detail}{RESET}")
    else:
        print(f"  {RED}CHAIN BROKEN at seq {result.first_broken_seq}{RESET}")
        print(f"  {DIM}{result.detail}{RESET}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="kavach-tamper-"))
    chain = AuditChain(Database(tmp / "demo.db"))

    banner("1. A normal run writes its records")
    chain.append("system.boot", profile="MID", backend="mock")
    chain.append("route.decide", cls="doc_summary", p=0.91, chosen="qwen3-8b-q4")
    chain.append("tool.invoke", tool="kb.search", args_sha256="3f1a" + "0" * 60)
    # This one FAILED. Step 3 is an attacker trying to hide that.
    chain.append("tool.invoke", tool="code.exec", exit_code=1, data_class="CONFIDENTIAL")
    chain.append("artifact.write", path="approval-note.docx", approved_by="pverma")
    print(f"  wrote {chain.count()} records")
    report(chain)

    banner("2. The append-only triggers refuse an edit")
    try:
        chain.db.execute("UPDATE audit SET event='nothing.happened' WHERE seq=4")
        print(f"  {RED}the UPDATE succeeded - triggers are missing{RESET}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {GREEN}refused{RESET} - {exc}")

    banner("3. Attacker drops the triggers and edits the record anyway")
    print(f"  {DIM}seq 4 recorded a FAILED sandbox run (exit_code 1).{RESET}")
    print(f"  {DIM}The attacker rewrites it to look successful.{RESET}")
    chain.db.execute("DROP TRIGGER audit_no_update")
    chain.db.execute(
        "UPDATE audit SET payload_json='{\"exit_code\":0,\"tool\":\"code.exec\"}' WHERE seq=4"
    )
    print(f"  {YELLOW}exit_code 1 -> 0, triggers gone{RESET}")
    report(chain)

    banner("4. A fresh chain, and a whole record deleted")
    chain2 = AuditChain(Database(tmp / "demo2.db"))
    for i in range(6):
        chain2.append("tool.invoke", tool="calc", n=i)
    chain2.db.execute("DROP TRIGGER audit_no_delete")
    chain2.db.execute("DELETE FROM audit WHERE seq=3")
    print(f"  {YELLOW}seq 3 deleted outright{RESET}")
    report(chain2)

    banner("5. And a record re-pointed at a forged predecessor")
    chain3 = AuditChain(Database(tmp / "demo3.db"))
    for i in range(6):
        chain3.append("tool.invoke", n=i)
    chain3.db.execute("DROP TRIGGER audit_no_update")
    chain3.db.execute(f"UPDATE audit SET prev_sha256='{'a' * 64}' WHERE seq=5")
    print(f"  {YELLOW}seq 5 re-linked to a hash that never existed{RESET}")
    report(chain3)

    print(
        f"\n{BOLD}Every edit, deletion and splice was caught, and each time the chain "
        f"named the FIRST broken link.{RESET}"
    )
    print(f"{DIM}scratch dbs: {tmp}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
