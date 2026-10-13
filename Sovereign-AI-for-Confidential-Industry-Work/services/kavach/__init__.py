"""Kavach Sovereign Workbench - air-gapped multi-model agentic AI runtime.

Layer map (see KVC-ARCH-001):
    L0  sovereignty/  audit/     isolation, tamper-evident log, policy
    L1  runtime/                 model registry, VRAM warden, backends
    L2  router/                  4-stage selection cascade
    L3  knowledge/               ingest + hybrid retrieval
    L4  tools/                   manifest-declared tools + sandbox
    L5  agent/                   plan -> execute -> observe -> critique
    L6  api/                     gateway the UI talks to
"""

__version__ = "0.1.0"
