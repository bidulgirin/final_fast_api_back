from openai import OpenAI
from dotenv import load_dotenv
from app.core.config import settings
from app.schemas.chat import SendChatRequest

load_dotenv()
client = OpenAI()

LEGAL_ASSISTANT_DEVELOPER_PROMPT = """
당신은 보이스피싱/금융사기 피해 대응을 돕는 '법률상담사 스타일' 안내 챗봇입니다.

규칙:
1) 단정적인 법률 자문이 아니라 '일반 정보'임을 짧게 고지하세요.
2) 사용자의 상황을 2~4개의 짧은 질문으로 먼저 확인한 뒤, 단계별 조치(우선순위/긴급도)를 제시하세요.
3) 기관 연락처(112, 1332, 거래은행 고객센터 등)와 증거 보존(통화기록/문자/계좌정보)을 강조하세요.
4) 민감정보(주민번호 전체, 계좌 비밀번호 등)는 절대 요구하지 마세요.
5) 답변은 한국어로, 문단/불릿으로 읽기 쉽게 작성하세요.
"""

def _build_case_context(req: SendChatRequest) -> str | None:
    summary_text = (req.summary_text or "").strip()
    call_text = (req.call_text or "").strip()

    if not req.call_id and not summary_text and not call_text:
        return None

    call_text_snippet = call_text[:1200] if call_text else ""

    parts = []
    if req.call_id:
        parts.append(f"- call_id: {req.call_id}")
    if summary_text:
        parts.append(f"- 요약:\n{summary_text}")
    if call_text_snippet:
        parts.append(f"- 통화 텍스트(발췌):\n{call_text_snippet}")

    return (
        "아래는 사용자가 챗봇에 진입한 '통화 상세 화면'의 맥락 정보입니다.\n"
        "이 정보를 전제로 답하되, 사용자가 묻지 않은 내부 식별자(call_id)는 굳이 언급하지 마세요.\n\n"
        + "\n".join(parts)
    )

def build_input_items(history: list[dict], req: SendChatRequest) -> list[dict]:
    items = [{"role": "system", "content": LEGAL_ASSISTANT_DEVELOPER_PROMPT}]

    case_context = _build_case_context(req)
    if case_context:
        items.append({"role": "user", "content": case_context})

    items.extend(history)
    items.append({"role": "user", "content": req.user_text})
    return items

# openai 가 아니라 이제 llm 모델에 물어봐야함 
def ask_openai(history: list[dict], req: SendChatRequest) -> str:
    response = client.responses.create(
        model=settings.OPENAI_MODEL,
        input=build_input_items(history, req),
        max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS,
        store=False,
        # debugging 중이면 아래는 잠깐 빼도 됨
        # reasoning={"effort": "low"},
    )
    return (response.output_text or "").strip()
