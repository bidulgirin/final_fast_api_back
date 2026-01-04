import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

CATEGORIES = ["기관사칭", "투자사기", "채용빙자", "납치협박", "가족,지인사칭"]

SYSTEM_PROMPT = """
너는 한국어 통화 STT 텍스트를 요약하는 어시스턴트다.

규칙:
- 원문에 없는 내용을 지어내지 않는다. 과한 추측 금지.
- 개인정보(주민번호/계좌/주소/전화번호/인증번호/카드번호 등)가 있으면 마스킹한다.
- 말투는 부드럽고 이해하기 쉽게 한다.
- 출력은 반드시 JSON 하나만 반환한다. (추가 텍스트 금지)
""".strip()


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
      "summary": str
    }

    - is_voicephishing / score 는 외부에서 주어진 값을 그대로 사용(재판단 금지)
    - LLM은 category/summary 작성에 집중
    """
    # 빈 텍스트 처리
    if not text or not text.strip():
        return {
            "isVoicephishing": bool(is_voicephishing),
            "voicephishingScore": float(voicephishing_score),
            "category": None,
            "summary": "",
        }

    prompt = f"""
아래는 통화 STT 원문이다.

중요:
- 보이스피싱 여부는 외부 시스템이 이미 판단했고, isVoicephishing={str(is_voicephishing).lower()} 로 확정이다.
- 보이스피싱 점수도 외부에서 정해졌고, voicephishingScore={voicephishing_score} 로 확정이다.
- 너는 이 두 값을 재판단/수정하지 말고 그대로 출력 JSON에 넣어라.

너의 작업:
- isVoicephishing이 true면: 아래 카테고리 중 하나로 분류하고, 핵심 내용을 자세히(3~6문장) 부드럽게 요약해라.
  카테고리 후보: {CATEGORIES}
- isVoicephishing이 false면: category는 null로 두고, 일반 통화 요약을 1~3문장으로 작성해라.
- 요약에서 개인정보가 보이면 마스킹해라.

반드시 아래 JSON 스키마만 출력:
{{
  "isVoicephishing": boolean,
  "voicephishingScore": number,
  "category": "기관사칭" | "투자사기" | "채용빙자" | "납치협박" | "가족,지인사칭" | null,
  "summary": string
}}

STT 원문:
<<<
{text}
>>>
""".strip()

    resp = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18"),
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
        # 파싱 실패 시에도 Android 스키마 유지
        return {
            "isVoicephishing": bool(is_voicephishing),
            "voicephishingScore": float(voicephishing_score),
            "category": None,
            "summary": resp.output_text.strip(),
        }

    # 외부 판정값 강제
    data["isVoicephishing"] = bool(is_voicephishing)
    data["voicephishingScore"] = float(voicephishing_score)

    # category 후처리
    if not data["isVoicephishing"]:
        data["category"] = None
    else:
        if data.get("category") not in CATEGORIES:
            data["category"] = None

    # summary 보장
    if "summary" not in data or not isinstance(data["summary"], str):
        data["summary"] = ""

    return data
