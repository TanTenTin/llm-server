# llm-server 고도화 백로그

로컬 LLM 서버 구축에 맞춰 게이트웨이를 점검한 결과. 각 항목은 **증상/영향 → 근거(파일:라인) → 수정 방향 → 검증**으로 정리했다.
우선순위: 🔴 Tier 1(정합성·비용 직결) / 🟠 Tier 2(로컬 활용) / 🟡 Tier 3(프로덕션 견고화).

> 각 항목은 GitHub 이슈로 등록됨 — **E-NN → 이슈 #N** (E-01 → [#1](https://github.com/TanTenTin/llm-server/issues/1) … E-17 → [#17](https://github.com/TanTenTin/llm-server/issues/17)). 라벨 `tier-1`/`tier-2`/`tier-3` + `enhancement`.

> 근거 라인 번호는 정리 시점(main @ 9fe7581) 기준. 착수 전 해당 파일에서 재확인할 것.

---

## ⛔ 보호 대상 — 이번 주 커밋(2026-07-02~07-08), 롤백 금지

아래 커밋들은 로컬 LLM 서버 전환 과정의 개선 시도다. **이 백로그의 어떤 항목도 이들을 되돌리는 방향으로 구현하지 않는다.**

| 커밋 | 내용 | 연관 항목 |
|------|------|-----------|
| `180b7da` | auto 라우팅 Phase 4 (long 티어·안전마진·overflow) | — |
| `d0eb6cd` | P0 관측 (동적 쿨다운·usage·심층 헬스·CI 게이트) | E-13, E-17 |
| `dba52c9` | P1 (vision·embeddings·캐시·예산가드·레이트리밋) | E-01, E-11, E-14, E-15 |
| `73e1002` | 기본 모델 → 로컬 Ollama, 폴백 Gemini | **E-02(주의)** |
| `49a115d` | auto 로컬 Ollama 우선 | **E-02(주의)** |
| `ef25e03` | /v1/models Ollama 실시간 조회 | E-03(기반) |
| `7922f63` | Ollama 네이티브 /api/chat·num_ctx·think | E-02, E-04~E-10 |
| `8f5acc7` | 컨텍스트 가드 + 로컬 SaaS 폴백 보장 | **E-02**, E-11 |

**핵심 주의 2건**
- **E-02**: `context_window`를 낮춰 num_ctx에 맞추면 로컬 usable 창이 줄어 로컬 우선 라우팅(49a115d/73e1002)이 후퇴한다 → **num_ctx를 올려 맞추는 방향으로만** 구현.
- **E-11**: 8f5acc7의 로컬 SaaS 폴백 보장은 chat 경로 전용이다 → embedding으로 **확장**(되돌리기 아님).

---

## 🔴 Tier 1 — 정합성 / 비용 직결

### E-01. 스트리밍 응답 토큰이 집계되지 않음 → 예산 가드·관측 실명
- **영향**: 에이전트 트래픽 대부분인 스트리밍이 `/v1/usage`에 토큰 0으로 잡힘. `PAID_DAILY_TOKEN_BUDGET` 가드를 스트리밍만으로 무한 우회 가능(과금 사고 직결).
- **근거**: `service.py:477` 스트리밍 성공을 `record_success(label, None, ...)`로 집계 → `usage.py:92` body=None이라 토큰 미적립.
- **수정 방향**: SSE 마지막 청크의 usage를 파싱해 집계. OpenAI 경로는 `stream_options.include_usage`로 usage 청크 유도, Ollama는 done 청크의 `prompt_eval_count`/`eval_count`, Anthropic은 `message_delta.usage`. 스트림 종료 시점에 누적 usage로 `record_success` 보정.
- **검증**: 스트리밍 요청 후 `/v1/usage`에 토큰이 잡히는지, 예산 초과 시 유료 후보가 스킵되는지 테스트.

### E-02. context_window(32k) vs num_ctx(16384) 불일치 → 조용한 입력 잘림
- **영향**: auto 라우터는 `ollama/qwen3:14b`를 "usable 25.6k(32000×0.8)까지 로컬 OK"로 판단하지만, 런타임엔 16384 창만 주입돼 그 사이 입력이 조용히 잘림. 라우팅 정합성 붕괴.
- **근거**: `registry.py:90` `context_window=32_000` 선언 · `_guard_context`(`registry.py:307`, `342`, `389`) vs `config.py:15` `OLLAMA_NUM_CTX=16384` · `ollama.py:153` 주입. 두 값이 서로를 모름.
- **수정 방향**: 단일 소스로 통일하되 **로컬 우선 라우팅을 후퇴시키지 않는 방향**으로. ⛔ `context_window`(32k)를 16384로 낮추는 방식은 금지 — usable 창이 25,600→13,107로 줄어 이번 주 로컬 우선 전환(49a115d/73e1002)과 8f5acc7 가드 판단이 후퇴한다. 대신 (a) `OLLAMA_NUM_CTX`를 32000(또는 context_window의 usable 값)까지 **올려** 실제 창을 선언에 맞추거나, (b) num_ctx를 레지스트리 spec에서 파생, (c) 요청 크기 기반 동적 산정(E-08 연계). 서버 RAM 한도 내에서 num_ctx 상향이 가능한지 확인 후 결정.
- **검증**: 20k 토큰 입력 auto 요청이 로컬로 갔을 때 잘리지 않는지, 또는 큰 컨텍스트 모델로 라우팅되는지. num_ctx 상향 후 OOM 없이 동작하는지.

### E-03. Ollama capability가 라우팅으로 환류되지 않음
- **영향**: 패스스루 Ollama 모델은 항상 `supports_tools=True`, `supports_vision=False`, `context_window=32_000` 고정. → vision 모델(llava 등)이 auto의 vision 필터에서 탈락해 이미지 요청이 로컬로 안 감. tools 미지원 모델엔 도구 요청이 갔다 실패.
- **근거**: `main.py:228` `_ollama_models`가 `/api/tags` capability를 `/v1/models` 표시엔 쓰지만, `registry.py:24`/`208`/`233` 패스스루 spec은 기본값 고정. capability→라우팅 다리 없음.
- **수정 방향**: 시작 시(또는 TTL 캐시) `/api/tags` capability를 읽어 패스스루 spec의 `supports_tools`/`supports_vision`/`context_window`에 반영. `_ollama_models`가 이미 조회하므로 그 결과를 registry가 참조하도록 연결.
- **검증**: llava류 설치 후 이미지 auto 요청이 로컬로 라우팅되는지, tools 미지원 모델이 도구 요청에서 제외되는지.

---

## 🟠 Tier 2 — 로컬 서버 활용 고도화

### E-04. Ollama thinking 출력 유실 (실질 버그)
- **영향**: qwen3/deepseek-r1 등에서 `think=True`를 명시해도 reasoning이 응답/스트림에서 조용히 사라짐.
- **근거**: Ollama 네이티브는 reasoning을 `message.thinking`으로 별도 반환하는데, `_native_to_openai`(`ollama.py:161`)·스트림(`ollama.py:282`) 모두 `content`만 읽음.
- **수정 방향**: `thinking` 필드를 OpenAI 응답의 `reasoning_content`(또는 message 확장 필드)로 노출. 스트림도 동일.
- **검증**: `think=true` 요청에서 reasoning이 노출되는지.

### E-05. keep_alive / 모델 워밍업 부재 → 콜드스타트 지연
- **영향**: payload에 `keep_alive`가 없어 Ollama 기본 5분 후 언로드. 큰 `num_ctx=16384`와 결합돼 재적재(모델+KV 캐시) 지연이 큼.
- **근거**: `ollama.py:132-157` payload에 `keep_alive` 없음. `config.py`에 관련 설정 없음.
- **수정 방향**: `OLLAMA_KEEP_ALIVE` 설정 추가해 payload에 주입(예: `"30m"` 또는 `-1` 상주). lifespan 기동 시 기본 모델 워밍업(빈/짧은 프롬프트 예열) 옵션.
- **검증**: 워밍업 후 첫 실요청 지연 감소 확인.

### E-06. 로컬 동시성 제어 없음 → 단일 GPU 스왑 스래싱
- **영향**: 단일 GPU 로컬 서버인데 게이트웨이 세마포어가 없어 동시 요청이 Ollama 큐에 쌓이거나 서로 다른 모델 요청이 섞여 모델 스왑 스래싱. 고정 `TIMEOUT=120s` 큐 대기 중 타임아웃 → 불필요한 폴백.
- **근거**: `service.py:137` 단일 인스턴스 + 단일 httpx 클라이언트 공유. 게이트웨이 레벨 큐/세마포어 없음. `ollama.py:13` `TIMEOUT=120.0` 고정.
- **수정 방향**: Ollama 호출에 `asyncio.Semaphore`(동시성 N, 설정화). read/connect 타임아웃 분리(E-16과 연계).
- **검증**: 동시 요청 부하에서 타임아웃/폴백 감소.

### E-07. 원격 이미지 URL 처리 버그
- **영향**: `http(s)://` 이미지 URL을 base64 자리에 그대로 넣어 Ollama가 400/무시.
- **근거**: `ollama.py:46-51` data URL이 아니면 `images.append(url)`로 URL 문자열을 그대로 실음. fetch→base64 로직 없음.
- **수정 방향**: 원격 URL이면 게이트웨이가 fetch해서 base64로 변환(크기 상한·타임아웃 포함), 실패 시 명확한 에러.
- **검증**: 원격 이미지 URL 멀티모달 요청 정상 동작.

### E-08. 컨텍스트가 정적·글로벌 (동적 num_ctx 미적용)
- **영향**: 모든 모델·요청에 동일 `num_ctx=16384`. 작은 요청에도 큰 창을 잡아 메모리/적재 낭비, 큰 요청엔 잘림(E-02).
- **근거**: `config.py:15` 글로벌 값 · `ollama.py:153` 무조건 주입. `registry.py:284` `_estimate_tokens`가 이미 있는데 num_ctx로 연결 안 됨.
- **수정 방향**: 요청 추정 토큰 + max_tokens 기반으로 num_ctx를 필요한 만큼만 산정(모델 상한 클램프). E-02/E-03과 함께 설계.
- **검증**: 작은 요청과 큰 요청의 num_ctx가 달라지는지.

### E-09. 샘플링 파라미터 매핑 누락
- **영향**: `top_p`/`top_k`/`stop`/`seed`/`presence_penalty`/`frequency_penalty`/`repeat_penalty`가 요청에 있어도 무시됨(OpenAI 호환 격차).
- **근거**: `ollama.py:147-156` `temperature`·`num_predict`만 매핑.
- **수정 방향**: OpenAI 파라미터 → Ollama `options` 매핑 테이블 추가.
- **검증**: `top_p`/`stop` 등이 실제 반영되는지.

### E-10. 스트리밍 tool_calls 델타 형식/id 취약
- **영향**: tool_calls 청크마다 새 uuid id 생성 → 멀티청크 시 id 불일치. arguments를 증분이 아닌 전체로 실어 누적 클라이언트와 어긋날 수 있음. done 청크에 tool_calls 없으면 finish_reason이 `stop`으로 오표기 여지.
- **근거**: `ollama.py:293-301`, id 생성 `ollama.py:102`.
- **수정 방향**: 한 스트림 내 tool_call id 안정화(인덱스 기반 유지), finish_reason 판정 보정. (현재 Ollama가 대개 단일 청크라 저빈도지만 취약.)
- **검증**: 스트리밍 도구 호출 왕복 회귀 테스트(현재 미검증 영역).

### E-11. embeddings 폴백 미보장
- **영향**: `ollama/nomic-embed-text` 직접 지정 시 미설치면 404 → 체인에 SaaS 폴백이 없어 그대로 에러.
- **근거**: `registry.py:419` `resolve_embedding`이 `_ensure_saas_fallback` 미호출. 레지스트리 항목(`registry.py:131`)에 `fallback` 비어 있음. (참고: 이번 주 8f5acc7이 추가한 `_ensure_saas_fallback`은 chat 경로 `resolve`/`_auto_route`에만 적용되고 embedding엔 미적용.)
- **수정 방향**: 8f5acc7의 로컬 SaaS 폴백 보장 정책을 **embedding으로 확장**(되돌리기 아님) — `resolve_embedding`도 `_ensure_saas_fallback` 상당의 SaaS 폴백을 붙이도록. chat의 resolve와 정합.
- **검증**: 미설치 로컬 임베딩 모델 요청이 Gemini embedding으로 폴백되는지.

---

## 🟡 Tier 3 — 프로덕션 견고화 (외부 노출 / 수평 확장 시)

### E-12. 모든 상태 인메모리 → 멀티워커/수평확장 붕괴 (근본 블로커)
- **영향**: `--workers N`에서 레이트리밋 실효 상한 N배 느슨, 예산 N배 과소집계(비용 폭주), 캐시 hit ratio ≈ 1/N, breaker 워커별 불일치. `/v1/usage`·breaker 상태가 한 워커 뷰만 반환해 관측 오도.
- **근거**: `main.py:47`(ratelimit)·`service.py:63,148`(breaker)·`service.py:150`/`usage.py:71`(usage)·`main.py:164`/`cache.py:49`(cache). "단일 스레드라 락 불필요" 주석은 워커 1개 내에서만 유효.
- **수정 방향**: 상태 백엔드 추상화 계층 도입 후 Redis 등으로 외부화. 당장 단일 워커면 무해 — 확장 로드맵 진입 전 선결 과제로 표시.
- **검증**: 2워커에서 레이트리밋/예산이 합산 집행되는지.

### E-13. 서킷 브레이커 설계 약점
- **영향**: 단일 실패로 즉시·provider 전체 개방(한 모델 404가 그 provider 전 모델을 뒤로 밈). half-open 만료 시 동시 요청이 전부 회복 탐침으로 몰림(썬더링 허드).
- **근거**: `service.py:88` 임계치 없이 1회로 개방·키가 provider 단위. `service.py:71-73` 만료 시 엔트리 삭제 후 전원 통과.
- **수정 방향**: N회/에러율 임계치 도입, (선택) 모델 단위 격리, half-open 단일 탐침 게이팅.
- **검증**: 1회 429로 전체 개방되지 않는지, 회복 시 탐침이 1건인지.

### E-14. 레이트리밋 — 프록시 뒤 IP 붕괴 + WS 무제한
- **영향**: X-Forwarded-For 미처리로 리버스 프록시(Caddy) 뒤에선 모든 클라이언트가 프록시 IP 하나로 묶임. WS `/v1/realtime`는 usage/budget/ratelimit 전부 우회.
- **근거**: `main.py:78` `client.host` 사용·XFF 미파싱. `main.py:111` WS 리밋 없음.
- **수정 방향**: 신뢰 프록시 XFF 파싱(신뢰 소스 화이트리스트), WS 연결 수/레이트 제한.
- **검증**: 프록시 뒤 실 클라이언트 IP로 리밋 걸리는지.

### E-15. 캐시 single-flight 부재 → 동시 miss 스탬피드
- **영향**: 동일 요청 N개 동시 진입 시 전부 miss로 업스트림 N번 호출 → 쿼터 절약 목표와 상충.
- **근거**: `main.py:344-368` get→await→put 사이 요청 합치기 없음.
- **수정 방향**: 진행 중 키에 대한 in-flight future 공유(single-flight).
- **검증**: 동일 요청 동시 다발 시 업스트림 1회만 호출되는지.

### E-16. connect 타임아웃 미분리 (죽은 호스트 폴백 지연)
- **영향**: connect/read/write/pool 전부 120초 단일값 → 죽은 호스트가 폴백까지 최대 120초를 잡아먹음.
- **근거**: `gemini.py:15,23,29`·`ollama.py`의 `TIMEOUT` 스칼라.
- **수정 방향**: `httpx.Timeout(connect=짧게, read=길게)` 분리. `httpx.Limits`도 명시. (선택) HTTP/2.
- **검증**: 도달 불가 업스트림에서 폴백 전환이 빠른지.

### E-17. 관측성 보강 (구조화 로그·메트릭·트레이싱)
- **영향**: 한글 free-text 로그·상관관계 ID 없음·레이턴시/큐/hit ratio 시계열 없음. 지연 회귀·요청 추적 불가.
- **근거**: `service.py:31,360-403` free-text·request_id 없음. Prometheus 엔드포인트 없음. `service.py:403` 응답 로그에 소요시간 없음.
- **수정 방향**: request_id 부여(응답 헤더 + 로그), 구조화(JSON) 로그 옵션, `/metrics`(provider별 레이턴시·에러·캐시 hit·in-flight), OTel 스팬(선택). Ollama `/api/ps`로 적재 상태 헬스 노출(선택).
- **검증**: 로그에서 단일 요청 추적, 메트릭 스크레이프 확인.

---

## 착수 제안 순서
1. **E-01, E-02, E-03** (Tier 1) — 비용·정합성 직결, 로컬 서버 실효 확보.
2. **E-04~E-09** (Tier 2) — 로컬 체감 성능/기능.
3. **E-12~E-17** (Tier 3) — 외부 노출/수평 확장 진입 전.
