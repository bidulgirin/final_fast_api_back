import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
너는 한국어 통화 STT 텍스트를 후처리하는 어시스턴트다.
- 의미를 바꾸지 말고, 오탈자/띄어쓰기/구어체 정리만 한다.
- 개인정보(주민번호/계좌/주소 등)가 포함되면 마스킹한다.
- 출력은 JSON만 반환한다.
"""

def postprocess_stt(text: str) -> dict:
    """
    STT 결과 텍스트를 LLM으로 정리/구조화해 반환.
    """
    if not text or not text.strip():
        return {
            "clean_text": "",
            "summary": "",
            "keywords": [],
            "redacted": False
        }

    prompt = f"""
다음은 통화 STT 원문이다. 아래 요구사항에 맞춰 JSON으로 출력해라.

요구사항:
1) clean_text: 맞춤법/띄어쓰기/중복어/군더더기 제거(의미 유지)
2) summary: 한글 1~2문장 요약
3) keywords: 핵심 키워드 3~8개 배열
4) redacted: 민감정보 마스킹이 발생했으면 true 아니면 false

STT 원문:
{text}
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",  # 비용/속도 균형 좋음. 필요시 gpt-4.1로 올리면 품질↑
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    # Responses API는 output_text로 바로 꺼낼 수 있음
    import json
    return json.loads(resp.output_text)
