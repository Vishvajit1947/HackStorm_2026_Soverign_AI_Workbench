"""Mock backend.

It has to be deterministic (so tests and rehearsals are reproducible), honest about
timing (so the warden's swap penalty is exercised), and imperfect on small models
(so Phase 5 is forced to handle malformed tool JSON rather than assuming it away).
"""

from __future__ import annotations

import pytest

from kavach.runtime.backends.base import ModelNotLoaded
from kavach.runtime.backends.mock import MockBackend
from kavach.runtime.registry import ModelSpec
from kavach.types import GenRequest


def spec(model_id="text", tier=2, dim=None, role="generation", vram=4.0):
    return ModelSpec(
        id=model_id,
        file=f"{model_id}.gguf",
        license="apache-2.0",
        tier=tier,
        capabilities=["text", "tools"],
        context=8192,
        vram_gb=vram,
        load_ms=100,
        tok_per_s=1000.0,
        role=role,
        dim=dim,
    )


@pytest.fixture()
def backend():
    return MockBackend(time_scale=0.0)


# ---------------------------------------------------------------- lifecycle

def test_generate_requires_a_loaded_model(backend):
    """If this ever stops raising, the residency table has become decorative."""
    with pytest.raises(ModelNotLoaded):
        backend.generate(spec(), GenRequest(prompt="hello"))


def test_load_then_generate(backend):
    s = spec()
    backend.load(s)
    result = backend.generate(s, GenRequest(prompt="hello"))
    assert result.model_id == "text"
    assert result.text


def test_unload_is_idempotent(backend):
    s = spec()
    backend.unload(s)
    backend.load(s)
    backend.unload(s)
    backend.unload(s)
    assert backend.loaded_models() == []


def test_double_load_is_a_noop(backend):
    s = spec()
    backend.load(s)
    backend.load(s)
    assert backend.load_calls == ["text"]


# ---------------------------------------------------------------- determinism

def test_same_prompt_same_output(backend):
    s = spec()
    backend.load(s)
    a = backend.generate(s, GenRequest(prompt="calculate NPSH"))
    b = backend.generate(s, GenRequest(prompt="calculate NPSH"))
    assert a.text == b.text


def test_different_models_differ(backend):
    a_spec, b_spec = spec("alpha"), spec("beta")
    backend.load(a_spec)
    backend.load(b_spec)
    a = backend.generate(a_spec, GenRequest(prompt="same prompt"))
    b = backend.generate(b_spec, GenRequest(prompt="same prompt"))
    assert a.text != b.text


def test_output_is_labelled_as_simulated(backend):
    """A screenshot of this must never be mistakable for a real model answer."""
    s = spec()
    backend.load(s)
    result = backend.generate(s, GenRequest(prompt="summarise the SOP"))
    assert "SIMULATED" in result.text


# ---------------------------------------------------------------- grammar

def test_grammar_produces_json_on_capable_models(backend):
    import json

    s = spec(tier=2)
    backend.load(s)
    grammar = 'root ::= "{" ws "\\"tool\\"" ws ":" ws string "}"'
    result = backend.generate(s, GenRequest(prompt="call a tool", grammar=grammar))
    json.loads(result.text)


def test_tiny_models_sometimes_emit_malformed_json(backend):
    """Deliberate. Phase 5 must survive this, so the mock must produce it."""
    s = spec(model_id="tiny", tier=0)
    backend.load(s)
    grammar = 'root ::= "{" ws "\\"tool\\"" ws ":" ws string "}"'

    outputs = [
        backend.generate(s, GenRequest(prompt=f"call tool {i}", grammar=grammar)).text
        for i in range(60)
    ]
    malformed = [o for o in outputs if not o.strip().startswith("{")]
    assert malformed, "tier-0 mock never misbehaved; Phase 5 would be tested on a lie"


# ---------------------------------------------------------------- embeddings

def test_embeddings_are_the_declared_dimension(backend):
    s = spec(model_id="emb", role="embedding", dim=64)
    vectors = backend.embed(s, ["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 64


def test_embeddings_are_normalised(backend):
    s = spec(model_id="emb", role="embedding", dim=64)
    (vec,) = backend.embed(s, ["pressure drop across the line"])
    assert sum(v * v for v in vec) == pytest.approx(1.0, abs=1e-6)


def test_similar_text_gets_similar_vectors(backend):
    """The router's kNN head has to be genuinely exercised offline, so the mock
    embedder needs real locality rather than random noise."""
    s = spec(model_id="emb", role="embedding", dim=256)
    near_a, near_b, far = backend.embed(s, [
        "write a python script to calculate pressure drop",
        "write python code that calculates pressure drop",
        "draft an approval note to the general manager",
    ])

    def dot(x, y):
        return sum(a * b for a, b in zip(x, y, strict=True))

    assert dot(near_a, near_b) > dot(near_a, far)


def test_embedding_is_deterministic(backend):
    s = spec(model_id="emb", role="embedding", dim=64)
    a = backend.embed(s, ["the same text"])
    b = backend.embed(s, ["the same text"])
    assert a == b


# ---------------------------------------------------------------- health

def test_health_declares_itself_simulated(backend):
    health = backend.health()
    assert health["simulated"] is True
    assert "MOCK" in health["warning"]
