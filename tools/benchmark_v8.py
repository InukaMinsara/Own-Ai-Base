from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.chat_engine import OwnAIEngine


PROMPTS = [
    "hi",
    "What is a transformer?",
    "RAG කියන්නේ මොකක්ද?",
    "Can you explain Python simply?",
    "Calculate 125 * 48",
    "How do I create an HTML button?",
    "What can you do?",
]


def main():
    engine = OwnAIEngine()

    print("=" * 64)
    print("OWN AI LOCAL BENCHMARK")
    print("=" * 64)
    print("Stage:", engine.stage)
    print("Parameters:", f"{sum(p.numel() for p in engine.model.parameters()):,}")
    print("Context:", engine.model.block_size)
    print("Vocabulary:", engine.tokenizer.vocab_size)
    print("=" * 64)

    for prompt in PROMPTS:
        started = time.perf_counter()

        try:
            answer = engine.generate(
                prompt,
                use_rag=True,
                max_new_tokens=120,
            )
            elapsed = time.perf_counter() - started

            print()
            print("USER:", prompt)
            print("OWN AI:", answer)
            print("TIME:", f"{elapsed:.2f}s")

        except Exception as exc:
            print()
            print("USER:", prompt)
            print("ERROR:", exc)

    print()
    print("=" * 64)
    print("Benchmark finished.")
    print("=" * 64)


if __name__ == "__main__":
    main()
