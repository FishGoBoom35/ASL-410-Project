from pathlib import Path

INFO_DIR = Path(__file__).parent / "information"


def load_topics():
    topics = []
    for path in sorted(INFO_DIR.glob("*.txt")):
        first_line, _, body = path.read_text(encoding="utf-8").partition("\n")
        keywords = {k.strip().lower() for k in first_line.removeprefix("keywords:").split(",")}
        topics.append((keywords, body.strip()))
    return topics


TOPICS = load_topics()


def find_info(words):
    words = {w.lower() for w in words}
    return "\n\n".join(body for keywords, body in TOPICS if keywords & words)