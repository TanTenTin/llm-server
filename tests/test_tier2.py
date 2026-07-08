"""
Tier 2 고도화(E-04·E-05·E-08·E-09·E-11) 회귀 테스트.

- E-04: Ollama thinking → reasoning_content 노출
- E-05: keep_alive payload 주입
- E-08: 동적 num_ctx(작은 요청 하한, 큰 요청 상한 클램프)
- E-09: 샘플링 파라미터(top_p/top_k/stop/seed) → Ollama options 매핑
- E-11: 로컬 임베딩 체인에 SaaS 폴백 보장
"""

from app.config import settings
from app.models import ChatCompletionRequest, Message
from app.providers.ollama import (
    _MIN_NUM_CTX,
    OllamaProvider,
    _build_options,
)
from app.registry import MODELS, resolve_embedding

_SPEC = MODELS["ollama/qwen3:14b"]


def _provider() -> OllamaProvider:
    # 생성 시 I/O 없음(httpx client는 지연 연결). 세마포어는 기본 0이라 이벤트 루프 불필요.
    return OllamaProvider("http://localhost:11434")


def _req(content: str, **extra) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="ollama/qwen3:14b",
        messages=[Message(role="user", content=content)],
        **extra,
    )


# ── E-09 샘플링 파라미터 매핑 ──────────────────────────────────
def test_sampling_params_mapped_to_options() -> None:
    opts = _build_options(_req("hi", top_p=0.8, top_k=40, seed=7, temperature=0.3))
    assert opts["top_p"] == 0.8
    assert opts["top_k"] == 40
    assert opts["seed"] == 7
    assert opts["temperature"] == 0.3


def test_stop_normalized_to_list() -> None:
    assert _build_options(_req("hi", stop="END"))["stop"] == ["END"]
    assert _build_options(_req("hi", stop=["A", "B"]))["stop"] == ["A", "B"]


# ── E-05 keep_alive ───────────────────────────────────────────
def test_keep_alive_injected() -> None:
    payload = _provider()._build_native_payload(_req("hi"), _SPEC, None)
    assert payload["keep_alive"] == settings.ollama_keep_alive


# ── E-08 동적 num_ctx ─────────────────────────────────────────
def test_dynamic_num_ctx_small_request_hits_floor() -> None:
    """작은 요청은 하한(_MIN_NUM_CTX)까지만 창을 잡는다(KV 캐시 절약)."""
    ctx = _provider()._resolve_num_ctx(_req("hi"))
    assert ctx == _MIN_NUM_CTX
    assert ctx < settings.ollama_num_ctx  # 설정 상한보다 작음


def test_dynamic_num_ctx_large_request_clamps_to_configured() -> None:
    """큰 요청은 설정 상한(ollama_num_ctx)까지만 확장한다(초과 클램프)."""
    big = _req("x" * 300_000)  # ~100k 토큰 추정 → 상한 초과
    assert _provider()._resolve_num_ctx(big) == settings.ollama_num_ctx


# ── E-04 reasoning_content ────────────────────────────────────
def test_thinking_exposed_as_reasoning_content() -> None:
    native = {
        "message": {"role": "assistant", "content": "답", "thinking": "속으로 생각"},
        "prompt_eval_count": 5,
        "eval_count": 3,
    }
    out = _provider()._native_to_openai(native, "qwen3:14b")
    assert out["choices"][0]["message"]["reasoning_content"] == "속으로 생각"


# ── E-11 임베딩 SaaS 폴백 ─────────────────────────────────────
def test_embedding_local_chain_gets_saas_fallback() -> None:
    """ollama 임베딩 직접 지정 시에도 SaaS(Gemini) 폴백이 체인에 보장된다."""
    chain = resolve_embedding("ollama/nomic-embed-text").chain
    providers = [spec.provider for spec in chain]
    assert providers[0] == "ollama"
    assert "gemini" in providers  # 미설치(404) 시 넘어갈 클라우드 후보
