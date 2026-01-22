import os
import re
import json
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

CATEGORIES = ["기관사칭", "투자사기", "채용빙자", "납치협박", "가족,지인사칭"]
KEYWORDS_MAX = 5

SYSTEM_PROMPT = """
너는 한국어 통화 STT 텍스트를 요약하는 어시스턴트다.

규칙:
- 원문에 없는 내용을 지어내지 않는다. 과한 추측 금지.
- 개인정보(이름/주민번호/계좌/주소/전화번호/이메일/인증번호/카드번호 등)가 있으면 마스킹한다.
- 마스킹은 보편적으로 쓰는 형태를 사용한다. (예: 홍*동, 010-****-5678)
- 키워드에는 개인정보/마스킹된 값/숫자열을 넣지 않는다.
- 말투는 부드럽고 이해하기 쉽게 한다.
- 출력은 반드시 JSON 하나만 반환한다. (추가 텍스트 금지)
""".strip()


STOPWORDS = {
    "저", "제가", "나는", "우리는", "그리고", "근데", "그런데", "그래서", "하지만",
    "네", "예", "아니요", "맞아요", "지금", "오늘", "내일", "어제", "감사합니다",
    "전화", "통화", "말씀", "확인", "가능", "때문", "정도", "같아요",
    # 흔한 조사/어미 느낌 토큰이 섞일 때 대비(완벽하진 않음)
    "은", "는", "이", "가", "을", "를", "에", "에서", "으로", "로", "와", "과", "도",
}

NAME_LABEL_PATTERN = re.compile(
    r"(?P<label>(이름|성명|고객명|수취인|예금주|계좌주|담당자|대표자)\s*(?:[:：]|은|는)?\s*)(?P<name>[가-힣]{2,4})"
)
NAME_SUFFIX_PATTERN = re.compile(
    r"(?<![가-힣])(?P<name>[가-힣]{2,4})(?=\s*(?:님|씨|과장|팀장|대리|부장|차장|주임|선생님|교수님|수사관|검사|사무관|조사관|드림|올림))"
)
RRN_PATTERN = re.compile(r"(?<!\d)(\d{6})[-\s]?([1-4]\d{6})(?!\d)")
PHONE_PATTERN = re.compile(r"(?<!\d)(0\d{1,2})[-\s]?(\d{3,4})[-\s]?(\d{4})(?!\d)")
CARD_PATTERN = re.compile(r"(?<!\d)(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})(?!\d)")
EMAIL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9._%+-])"
)
ACCOUNT_LABEL_PATTERN = re.compile(
    r"(?P<label>(계좌번호|계좌|입금계좌|송금계좌)\s*(?:[:：]|은|는)?\s*)(?P<num>\d{2,4}[-\s]?\d{2,6}[-\s]?\d{2,6})"
)
AUTH_CODE_PATTERN = re.compile(r"(?P<label>(인증번호|인증 코드|인증코드|OTP)\s*(?:[:：]|은|는)?\s*)(?P<code>\d{4,8})")

def _mask_korean_name(name: str) -> str:
    if not name:
        return ""
    if len(name) == 1:
        return "*"
    if len(name) == 2:
        return f"{name[0]}*"
    if len(name) == 3:
        return f"{name[0]}*{name[-1]}"
    return f"{name[0]}{'*' * (len(name) - 2)}{name[-1]}"

def _mask_digits_keep_edges(value: str, keep_start: int, keep_end: int) -> str:
    digits = [i for i, ch in enumerate(value) if ch.isdigit()]
    if len(digits) <= keep_start + keep_end:
        return value
    mask_indexes = digits[keep_start : len(digits) - keep_end]
    chars = list(value)
    for idx in mask_indexes:
        chars[idx] = "*"
    return "".join(chars)

def mask_sensitive_info(text: str) -> str:
    if not text:
        return text

    text = NAME_LABEL_PATTERN.sub(
        lambda m: f"{m.group('label')}{_mask_korean_name(m.group('name'))}", text
    )
    text = NAME_SUFFIX_PATTERN.sub(lambda m: _mask_korean_name(m.group("name")), text)
    text = RRN_PATTERN.sub(lambda m: f"{m.group(1)}-*******", text)
    text = PHONE_PATTERN.sub(
        lambda m: f"{m.group(1)}-{'*' * len(m.group(2))}-{m.group(3)}", text
    )
    text = CARD_PATTERN.sub(lambda m: f"{m.group(1)}-****-****-{m.group(4)}", text)
    text = EMAIL_PATTERN.sub(
        lambda m: f"{m.group(1)[:2]}{'*' * max(1, len(m.group(1)) - 2)}@{m.group(2)}",
        text,
    )
    text = ACCOUNT_LABEL_PATTERN.sub(
        lambda m: f"{m.group('label')}{_mask_digits_keep_edges(m.group('num'), 3, 2)}",
        text,
    )
    text = AUTH_CODE_PATTERN.sub(
        lambda m: f"{m.group('label')}{'*' * len(m.group('code'))}", text
    )
    return text

def _simple_keyword_fallback(text: str, max_k: int = 5) -> list[str]:
    """
    외부 라이브러리 없이 간단히 키워드 후보를 뽑는 fallback.
    - 숫자/마스킹/이메일/URL/전화번호 같은 건 제외
    - 공백 토큰 기반 빈도 상위 고유 토큰 반환
    """
    if not text:
        return []

    # URL/이메일 제거
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)

    # 마스킹 패턴/숫자열/전화번호 비슷한 패턴 제거
    text = re.sub(r"[*]{2,}", " ", text)
    text = re.sub(r"\b\d{2,}\b", " ", text)
    text = re.sub(r"\b\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4}\b", " ", text)

    # 특수문자 정리
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    tokens = [t.strip() for t in text.split() if t.strip()]

    cleaned = []
    for t in tokens:
        if len(t) < 2:
            continue
        if t in STOPWORDS:
            continue
        if any(ch.isdigit() for ch in t):
            continue
        if "*" in t:
            continue
        cleaned.append(t)

    if not cleaned:
        return []

    freq = Counter(cleaned)
    out = []
    for w, _ in freq.most_common(50):
        if w not in out:
            out.append(w)
        if len(out) >= max_k:
            break
    return out


def _simple_community_fallback(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return [f"- {text.strip()}"]


def postprocess_stt(
    text: str,
    is_voicephishing: bool,
    voicephishing_score: float,
) -> dict:
    """
    Android의 LlmResult 스키마에 맞춰 반환:
    {
      "isVoicephishing": bool,
      "voicephishingScore": float,
      "category": str|null,
      "summary": str,
      "keywords": string[],   # <= 5
      "community": string[]   # ["- ...", "- ...", "- ..."]
    }

    - is_voicephishing/score? ???? ?, ?? ??? STT ???? LLM? ??
    - LLM? category/summary/keywords/community ? isVoicephishing/score? ??
    """
    # 빈 텍스트 처리
    if not text or not text.strip():
        return {
            "isVoicephishing": bool(is_voicephishing),
            "voicephishingScore": float(voicephishing_score),
            "category": None,
            "summary": "",
            "keywords": [],
            "community": [],
        }

    prompt = f"""
아래는 통화 STT 원문이다.

중요:
- ?? ??? ??? ????? ??(isVoicephishing)? ??(voicephishingScore)? ?? ????.
- voicephishingScore? 0.0 ~ 1.0 ??? ?? ??? ????.

너의 작업:
- isVoicephishing이 true면: 아래 카테고리 중 하나로 분류하고, 핵심 내용을 자세히(3~6문장) 부드럽게 요약해라.
  카테고리 후보: {CATEGORIES}
- isVoicephishing이 false면: category는 null로 두고, 일반 통화 요약을 1~3문장으로 작성해라.
- 요약에서 이름/전화번호 등 민감정보가 보이면 보편적으로 쓰는 형태로 마스킹해라.
- 추가로 keywords를 5개 이하로 추출해라.
  - keywords는 핵심 주제/행동/요구사항 중심의 짧은 명사/구(2~12자 권장).
  - 개인정보(숫자열/계좌/전화/주소/인증번호/카드번호 등) 및 마스킹된 값(****)은 넣지 마라.
  - 중복은 제거해라.

반드시 아래 JSON 스키마만 출력:
{{
  "isVoicephishing": boolean,
  "voicephishingScore": number,
  "category": "기관사칭" | "투자사기" | "채용빙자" | "납치협박" | "가족,지인사칭" | null,
  "summary": string,
  "keywords": string[],
  "community": string[]
}}

STT 원문:
<<<
{text}
>>>
""".strip()
    prompt += "\n\nCOMMUNITY: Provide speaker-segmented utterances as an array. Prefix each item with \"- \", do not use \"a:\"/\"b:\" labels, and keep each item to a single speaker's utterance. Example: [\"- ...\", \"- ...\", \"- ...\"]. If unclear, return a single item like \"- <text>\"."

    resp = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-2024-11-20"),
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        text={"format": {"type": "json_object"}},
        temperature=0.2,
    )

    # 파싱
    try:
        data = json.loads(resp.output_text)
    except Exception:
        # 파싱 실패 시에도 Android 스키마 유지 + keywords fallback
        fallback_keywords = _simple_keyword_fallback(text, KEYWORDS_MAX)
        return {
            "isVoicephishing": bool(is_voicephishing),
            "voicephishingScore": float(voicephishing_score),
            "category": None,
            "summary": mask_sensitive_info(resp.output_text.strip()),
            "keywords": fallback_keywords,
            "community": [mask_sensitive_info(c) for c in _simple_community_fallback(text)],
        }

    # 외부 판정값 강제
    # LLM ?? ??/??? (?? ? ????? fallback)
    if "isVoicephishing" not in data or not isinstance(data["isVoicephishing"], (bool, int)):
        data["isVoicephishing"] = bool(is_voicephishing)
    else:
        data["isVoicephishing"] = bool(data["isVoicephishing"])

    score = data.get("voicephishingScore", None)
    try:
        score = float(score)
    except Exception:
        score = float(voicephishing_score)
    if score != score:  # NaN guard
        score = float(voicephishing_score)
    score = max(0.0, min(1.0, score))
    data["voicephishingScore"] = score

    # category 후처리
    if not data["isVoicephishing"]:
        data["category"] = None
    else:
        if data.get("category") not in CATEGORIES:
            data["category"] = None

    # summary 보장
    if "summary" not in data or not isinstance(data["summary"], str):
        data["summary"] = ""
    data["summary"] = mask_sensitive_info(data["summary"])

    # keywords 보장/정규화 (<=5, 문자열 리스트, 개인정보성 토큰 제거)
    kws = data.get("keywords", [])
    if not isinstance(kws, list):
        kws = []
    kws = [k for k in kws if isinstance(k, str)]
    # trim + 중복 제거(순서 유지)
    seen = set()
    norm = []
    for k in [k.strip() for k in kws]:
        if not k:
            continue
        # 개인정보/숫자/마스킹/URL/이메일 냄새 제거
        if "*" in k:
            continue
        if any(ch.isdigit() for ch in k):
            continue
        if "@" in k or "http" in k or "www" in k:
            continue
        if k in seen:
            continue
        seen.add(k)
        norm.append(k)
        if len(norm) >= KEYWORDS_MAX:
            break

    # 너무 비면 summary(또는 text)로 fallback
    if not norm:
        base = data["summary"] if data.get("summary") else text
        norm = _simple_keyword_fallback(base, KEYWORDS_MAX)

    data["keywords"] = norm

    # ensure community
    community = data.get("community", [])
    if not isinstance(community, list):
        community = []
    community = [c.strip() for c in community if isinstance(c, str)]
    community = [mask_sensitive_info(c) for c in community if c]
    if not community:
        community = [mask_sensitive_info(c) for c in _simple_community_fallback(text)]
    data["community"] = community

    return data
