import json
from datetime import datetime
from pathlib import Path

from ollama import chat
from retriever import find_info

MIN_CONFIDENCE = 60

PROMPTS_DIR = Path(__file__).parent / "prompts"
INSTRUCTIONS = (PROMPTS_DIR / "instructions.txt").read_text(encoding="utf-8")
EXAMPLES = json.loads((PROMPTS_DIR / "examples.json").read_text(encoding="utf-8"))


def respond(signs):
    if any(conf < MIN_CONFIDENCE for _, conf in signs):
        return "Sorry, I didn't catch that. Please sign again."

    words = [word for word, _ in signs]
    info = find_info(words) or "No information available for this question."
    now = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")

    prompt = f"Current time: {now}\n\nCampus info:\n{info}\n\nStudent signed: {' '.join(words)}"

    reply = chat(
        model="gemma4:e2b",
        messages=[
            {"role": "system", "content": INSTRUCTIONS},
            *EXAMPLES,
            {"role": "user", "content": prompt},
        ],
        think=False,
        options={"temperature": 0.2, "num_predict": 150},
    )
    return reply["message"]["content"]


if __name__ == "__main__":
    while True:
        text = input("Signs: ")
        if not text:
            break
        print(respond([(w, 100) for w in text.split()]), "\n")