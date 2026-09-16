import json, os

transcript_path = r"C:\Users\I Tech\.gemini\antigravity-ide\brain\ac8fb2c7-0697-4160-9745-5fdff0f5d00b\.system_generated\logs\transcript_full.jsonl"
with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        c = d.get("content", "")
        if "File Path: `file:///d:/WhiteBox/SIH2026_IF/dashboard/index.html`" in c:
            step = d.get("step_index")
            print("FOUND VIEW OF index.html at step:", step)
            with open(f"scratch_chunk_{step}.txt", "w", encoding="utf-8") as out:
                out.write(c)
