from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGET_MIN_CHARS = 500
TARGET_MAX_CHARS = 1000
OVERLAP_CHARS = 120
SHORT_CHUNK_MERGE_THRESHOLD = 300

PAGE_MARK_RE = re.compile(r"^\s*=+\s*第\s*(\d+)\s*页\s*=+\s*$", re.MULTILINE)
HAS_CONTENT_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
CHAPTER_RE = re.compile(r"^第\s*([一二三四五六七八九十百千万\d]+)\s*[章节篇部]\s*[:：、.\s]*(.*)$")
NUMBERED_SECTION_RE = re.compile(
    r"^(\d{1,2})[\.．](\d{1,2})(?:[\.．]\d{1,2})?\s*(\S.*)$"
)
CN_NUMBERED_SECTION_RE = re.compile(r"^[（(]?[一二三四五六七八九十]{1,4}[）)]?[、.．]\s*(\S.*)$")


@dataclass(frozen=True)
class TextLine:
    text: str
    page: int | None


@dataclass(frozen=True)
class Heading:
    title: str
    kind: str
    root_number: int | None = None


@dataclass
class Section:
    chapter_id: str
    chapter_title: str
    source_scope: str
    lines: list[TextLine]


def build_chunks_from_parsed_text(parsed_text: str, parsed_name: str) -> dict[str, Any]:
    """
    Build the first-version single-document chunk artifact from parsed PDF text.

    The function is deliberately local and deterministic: no embedding, vector
    index, retrieval, or answer generation happens here.
    """
    document_id = Path(parsed_name).stem
    lines = _extract_clean_lines(parsed_text)
    sections = _build_sections(lines)

    draft_chunks: list[dict[str, Any]] = []
    for section in sections:
        draft_chunks.extend(_split_section_into_drafts(section))

    draft_chunks = _merge_short_chunks(draft_chunks)

    chunks = []
    for index, draft in enumerate(draft_chunks, start=1):
        chunk_text = draft["chunk_text"].strip()
        chunks.append(
            {
                "document_id": document_id,
                "parsed_name": parsed_name,
                "chunk_id": f"c_{index:03d}",
                "chunk_order": index,
                "chunk_text": chunk_text,
                "chapter_id": draft["chapter_id"],
                "chapter_title": draft["chapter_title"],
                "source_scope": draft["source_scope"],
                "page_range": draft["page_range"],
                "chunk_type": draft["chunk_type"],
                "summary_hint": _build_summary_hint(chunk_text, draft["source_scope"]),
            }
        )

    return {
        "document_id": document_id,
        "parsed_name": parsed_name,
        "chunks": chunks,
    }


def save_chunks(chunks_payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _split_text_into_pages(parsed_text: str) -> list[tuple[int | None, str]]:
    if not parsed_text or not parsed_text.strip():
        return []

    matches = list(PAGE_MARK_RE.finditer(parsed_text))
    if not matches:
        return [(None, parsed_text)]

    pages: list[tuple[int | None, str]] = []
    prefix = parsed_text[: matches[0].start()].strip()
    if prefix:
        pages.append((None, prefix))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(parsed_text)
        page_text = parsed_text[start:end]
        pages.append((int(match.group(1)), page_text))

    return pages


def _extract_clean_lines(parsed_text: str) -> list[TextLine]:
    pages = _split_text_into_pages(parsed_text)
    if not pages:
        return []

    cleaned_by_page: list[tuple[int | None, list[str]]] = []
    for page, page_text in pages:
        page_lines = []
        for raw_line in page_text.splitlines():
            line = _normalize_line(raw_line)
            if not line or _is_noise_line(line):
                continue
            page_lines.append(line)
        cleaned_by_page.append((page, page_lines))

    repeated_lines = _find_repeated_header_footer_lines(cleaned_by_page)

    lines: list[TextLine] = []
    for page, page_lines in cleaned_by_page:
        for line in page_lines:
            if line in repeated_lines:
                continue
            lines.append(TextLine(text=line, page=page))

    return lines


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _is_noise_line(line: str) -> bool:
    if not HAS_CONTENT_RE.search(line):
        return True
    if re.fullmatch(r"\d{1,4}", line):
        return True
    if re.fullmatch(r"[\W_]{3,}", line) and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", line):
        return True
    if len(line) >= 3 and len(set(line)) == 1:
        return True
    return False


def _find_repeated_header_footer_lines(cleaned_by_page: list[tuple[int | None, list[str]]]) -> set[str]:
    page_count = sum(1 for page, _ in cleaned_by_page if page is not None)
    if page_count < 6:
        return set()

    line_pages: dict[str, set[int]] = {}
    for page, lines in cleaned_by_page:
        if page is None:
            continue
        for line in set(lines):
            if 2 <= len(line) <= 50:
                line_pages.setdefault(line, set()).add(page)

    min_repeats = max(3, math.ceil(page_count * 0.12))
    return {line for line, pages in line_pages.items() if len(pages) >= min_repeats}


def _build_sections(lines: list[TextLine]) -> list[Section]:
    if not lines:
        return []

    sections: list[Section] = []
    current_lines: list[TextLine] = []
    chapter_counter = 0
    current_chapter_id = "ch_00"
    current_chapter_title = "未识别章节"
    current_scope = "全文"
    title_by_root: dict[int, str] = {}

    def flush_current() -> None:
        nonlocal current_lines
        if not current_lines:
            return
        sections.append(
            Section(
                chapter_id=current_chapter_id,
                chapter_title=current_chapter_title,
                source_scope=current_scope,
                lines=current_lines,
            )
        )
        current_lines = []

    for line in lines:
        heading = _detect_heading(line.text)
        if heading:
            flush_current()

            if heading.kind == "chapter":
                chapter_counter += 1
                current_chapter_id = f"ch_{chapter_counter:02d}"
                current_chapter_title = heading.title
                current_scope = heading.title
                if heading.root_number is not None:
                    title_by_root[heading.root_number] = heading.title
                    current_chapter_id = f"ch_{heading.root_number:02d}"
            else:
                if heading.root_number is not None:
                    next_chapter_id = f"ch_{heading.root_number:02d}"
                    if (
                        current_chapter_id != next_chapter_id
                        or current_chapter_title in {"全文", "目录", "未识别章节"}
                        or title_by_root.get(heading.root_number)
                    ):
                        current_chapter_title = title_by_root.get(
                            heading.root_number,
                            f"第{heading.root_number}章",
                        )
                    current_chapter_id = next_chapter_id
                current_scope = _join_scope(current_chapter_title, heading.title)

            current_lines = [line]
            continue

        current_lines.append(line)

    flush_current()
    return sections


def _detect_heading(line: str) -> Heading | None:
    text = line.strip()
    if not text or len(text) > 80:
        return None

    if text in {"目录", "前言", "序", "自序", "后记", "附录"}:
        return Heading(title=text, kind="chapter")

    chapter_match = CHAPTER_RE.match(text)
    if chapter_match:
        title_tail = chapter_match.group(2).strip()
        title = text if not title_tail else title_tail
        return Heading(title=title, kind="chapter")

    numbered_match = NUMBERED_SECTION_RE.match(text)
    if numbered_match:
        section_title = numbered_match.group(3).strip()
        if not HAS_CONTENT_RE.search(section_title) or section_title.startswith("%"):
            return None
        root_number = int(numbered_match.group(1))
        return Heading(title=text, kind="section", root_number=root_number)

    if len(text) <= 45 and CN_NUMBERED_SECTION_RE.match(text) and not _looks_like_body_sentence(text):
        return Heading(title=text, kind="section")

    return None


def _looks_like_body_sentence(text: str) -> bool:
    if len(text) > 45:
        return True
    return text.endswith(("。", "，", "；", "：", "！", "？", ",", ";"))


def _join_scope(chapter_title: str, title: str) -> str:
    if not chapter_title or chapter_title == title:
        return title
    return f"{chapter_title} / {title}"


def _split_section_into_drafts(section: Section) -> list[dict[str, Any]]:
    expanded_lines = _expand_long_lines(section.lines)
    if not expanded_lines:
        return []

    drafts: list[dict[str, Any]] = []
    current: list[TextLine] = []
    current_len = 0

    for line in expanded_lines:
        projected_len = current_len + len(line.text) + (1 if current else 0)
        if current and projected_len > TARGET_MAX_CHARS:
            drafts.append(_make_draft(section, current))
            current = _tail_for_overlap(current)
            current_len = _lines_len(current)

        current.append(line)
        current_len += len(line.text) + (1 if current_len else 0)

    if current:
        drafts.append(_make_draft(section, current))

    return [draft for draft in drafts if draft["chunk_text"].strip()]


def _expand_long_lines(lines: list[TextLine]) -> list[TextLine]:
    expanded: list[TextLine] = []
    for line in lines:
        if len(line.text) <= TARGET_MAX_CHARS:
            expanded.append(line)
            continue

        start = 0
        text = line.text
        while start < len(text):
            end = min(start + TARGET_MAX_CHARS, len(text))
            if end < len(text):
                end = _find_split_point(text, start, end)
            part = text[start:end].strip()
            if part:
                expanded.append(TextLine(text=part, page=line.page))
            if end >= len(text):
                break
            start = max(end - OVERLAP_CHARS, start + 1)
    return expanded


def _find_split_point(text: str, start: int, end: int) -> int:
    lower_bound = min(start + TARGET_MIN_CHARS, end)
    for index in range(end - 1, lower_bound, -1):
        if text[index] in "。！？；.!?;":
            return index + 1
    return end


def _tail_for_overlap(lines: list[TextLine]) -> list[TextLine]:
    tail: list[TextLine] = []
    total = 0
    for line in reversed(lines):
        tail.insert(0, line)
        total += len(line.text) + 1
        if total >= OVERLAP_CHARS:
            break
    return tail


def _lines_len(lines: list[TextLine]) -> int:
    return sum(len(line.text) for line in lines) + max(0, len(lines) - 1)


def _make_draft(section: Section, lines: list[TextLine]) -> dict[str, Any]:
    chunk_text = "\n".join(line.text for line in lines).strip()
    return {
        "chunk_text": chunk_text,
        "chapter_id": section.chapter_id,
        "chapter_title": section.chapter_title,
        "source_scope": section.source_scope,
        "page_range": _format_page_range([line.page for line in lines]),
        "chunk_type": "正文",
    }


def _merge_short_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not chunks:
        return []

    result: list[dict[str, Any]] = []
    index = 0
    while index < len(chunks):
        current = chunks[index]
        while (
            len(current["chunk_text"]) < SHORT_CHUNK_MERGE_THRESHOLD
            and index + 1 < len(chunks)
            and _can_merge_chunks(current, chunks[index + 1])
        ):
            current = _merge_two_chunks(current, chunks[index + 1])
            index += 1

        if (
            result
            and len(current["chunk_text"]) < SHORT_CHUNK_MERGE_THRESHOLD
            and _can_merge_chunks(result[-1], current)
        ):
            result[-1] = _merge_two_chunks(result[-1], current)
        else:
            result.append(current)

        index += 1

    return result


def _can_merge_chunks(first: dict[str, Any], second: dict[str, Any]) -> bool:
    both_are_short = (
        len(first["chunk_text"]) < SHORT_CHUNK_MERGE_THRESHOLD
        and len(second["chunk_text"]) < SHORT_CHUNK_MERGE_THRESHOLD
    )
    if first["chapter_id"] != second["chapter_id"] and not both_are_short:
        return False
    merged_len = len(first["chunk_text"]) + len(second["chunk_text"]) + 2
    return merged_len <= TARGET_MAX_CHARS


def _merge_two_chunks(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_text": f"{first['chunk_text'].strip()}\n\n{second['chunk_text'].strip()}",
        "chapter_id": first["chapter_id"],
        "chapter_title": first["chapter_title"],
        "source_scope": _merge_scope(first["source_scope"], second["source_scope"]),
        "page_range": _merge_page_ranges(first["page_range"], second["page_range"]),
        "chunk_type": first["chunk_type"],
    }


def _merge_scope(first: str, second: str) -> str:
    if first == second:
        return first
    parts = []
    for value in (first, second):
        if value and value not in parts:
            parts.append(value)
    return _truncate(" / ".join(parts[:2]), 120)


def _format_page_range(pages: list[int | None]) -> str:
    known_pages = [page for page in pages if page is not None]
    if not known_pages:
        return "unknown"
    start = min(known_pages)
    end = max(known_pages)
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _merge_page_ranges(first: str, second: str) -> str:
    pages = []
    for page_range in (first, second):
        pages.extend(_pages_from_range(page_range))
    return _format_page_range(pages)


def _pages_from_range(page_range: str) -> list[int | None]:
    if not page_range or page_range == "unknown":
        return [None]
    if "-" not in page_range:
        try:
            return [int(page_range)]
        except ValueError:
            return [None]
    start_text, end_text = page_range.split("-", 1)
    try:
        return [int(start_text), int(end_text)]
    except ValueError:
        return [None]


def _build_summary_hint(chunk_text: str, source_scope: str) -> str:
    for line in chunk_text.splitlines():
        line = line.strip()
        if not line or _detect_heading(line):
            continue
        if len(line) >= 12:
            return _truncate(line, 70)
    if source_scope and source_scope != "全文":
        return f"围绕{_truncate(source_scope, 60)}的正文内容"
    return "该片段为文档正文内容"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
