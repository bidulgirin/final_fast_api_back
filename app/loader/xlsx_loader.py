from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import re  

# 마스킹 함수
def advanced_deidentify(text: str) -> str:
    if not isinstance(text, str): 
        return ""
    titles = r"님|씨|과장|팀장|대리|부장|차장|주임|선생님|교수님"
    text = re.sub(rf'([가-힣]{{2,4}})({titles})', r'[NAME]\2', text)
    text = re.sub(r'([가-힣]{{2,4}})\s*(수사관|검사|사무관|조사관|드림|올림)', r'[NAME] \2', text)
    text = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '[TEL]', text)
    text = re.sub(r'\d{10,14}', '[ACC]', text)
    text = re.sub(r'http[s]?://\S+', '[URL]', text)
    text = re.sub(r'\d{4,}', '[NUM]', text)
    return text


@dataclass
class GroupDoc:
    file_id: int
    interval: int
    case_name: str
    text: str


def load_grouped_docs_from_xlsx(path: str) -> list[GroupDoc]:
    df = pd.read_excel(path).fillna("")

    required = {"text", "file_id", "interval", "파일"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"엑셀에 필요한 컬럼이 없습니다: {sorted(missing)}")

    df["file_id"] = df["file_id"].astype(int)
    df["interval"] = df["interval"].astype(int)
    df["파일"] = df["파일"].astype(str)
    df["text"] = df["text"].astype(str)

    # 여기서 text 컬럼 전체 마스킹 적용
    df["text"] = df["text"].map(advanced_deidentify)

    grouped_docs: list[GroupDoc] = []

    for (file_id, interval, case_name), g in df.groupby(["file_id", "interval", "파일"], sort=True):
        lines = [t.strip() for t in g["text"].tolist() if str(t).strip()]
        if not lines:
            continue

        merged = "\n".join(lines)

        merged_text = (
            f"[보이스피싱 사례]\n"
            f"file_id: {file_id}\n"
            f"interval: {interval}\n"
            f"case_name: {case_name}\n"
            f"내용:\n{merged}"
        )

        grouped_docs.append(
            GroupDoc(
                file_id=int(file_id),
                interval=int(interval),
                case_name=str(case_name),
                text=merged_text,
            )
        )

    return grouped_docs
