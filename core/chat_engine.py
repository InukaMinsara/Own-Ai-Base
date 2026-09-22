from pathlib import Path
import re
from difflib import SequenceMatcher

import torch

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer import OwnTokenizer
from tokenizer.tokenizer_v7 import OwnTokenizerV7
from tokenizer.tokenizer_v8 import OwnTokenizerV8
from model.own_ai import OwnAI
from model.own_ai_v7 import OwnAIv7
from model.own_ai_v8 import OwnAIv8
from rag.retriever import LocalRetriever
from tools.tool_router import ToolRouter


class OwnAIEngine:
    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        checkpoints = {
            "v8-sft": (
                ROOT / "checkpoints" / "own_ai_v8_sft_best.pt",
                ROOT / "checkpoints" / "tokenizer_v8_sft.json",
                OwnAIv8,
                OwnTokenizerV8,
            ),
            "v8-pretrain-streaming": (
                ROOT / "checkpoints" / "own_ai_v8_streaming_best.pt",
                ROOT / "checkpoints" / "tokenizer_v8_streaming.json",
                OwnAIv8,
                OwnTokenizerV8,
            ),
            "v8-pretrain": (
                ROOT / "checkpoints" / "own_ai_v8_pretrain_best.pt",
                ROOT / "checkpoints" / "tokenizer_v8.json",
                OwnAIv8,
                OwnTokenizerV8,
            ),
            "v7-sft": (
                ROOT / "checkpoints" / "own_ai_v7_sft_best.pt",
                ROOT / "checkpoints" / "tokenizer_v7_sft.json",
                OwnAIv7,
                OwnTokenizerV7,
            ),
            "v7-pretrain": (
                ROOT / "checkpoints" / "own_ai_v7_pretrain_best.pt",
                ROOT / "checkpoints" / "tokenizer_v7.json",
                OwnAIv7,
                OwnTokenizerV7,
            ),
            "v6-sft": (
                ROOT / "checkpoints" / "own_ai_v6_sft_best.pt",
                ROOT / "checkpoints" / "tokenizer_v6.json",
                OwnAI,
                OwnTokenizer,
            ),
            "base": (
                ROOT / "checkpoints" / "own_ai_best.pt",
                ROOT / "checkpoints" / "tokenizer_best.json",
                OwnAI,
                OwnTokenizer,
            ),
        }

        for stage, (
            model_path,
            tokenizer_path,
            model_class,
            tokenizer_class,
        ) in checkpoints.items():
            if model_path.exists() and tokenizer_path.exists():
                self.stage = stage
                self.checkpoint_path = model_path
                self.tokenizer_path = tokenizer_path
                self.model_class = model_class
                self.tokenizer_class = tokenizer_class
                break
        else:
            raise FileNotFoundError(
                "No trained Own AI checkpoint found."
            )

        self.tokenizer = self.tokenizer_class(
            vocab_size=(
                4096
                if self.stage.startswith("v8")
                else 2048
                if self.stage.startswith("v7")
                else 512
            )
        )
        self.tokenizer.load(
            self.tokenizer_path
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        model_kwargs = {
            "vocab_size": checkpoint["vocab_size"],
            "block_size": checkpoint["block_size"],
            "d_model": checkpoint["d_model"],
            "n_heads": checkpoint["n_heads"],
            "n_layers": checkpoint["n_layers"],
            "dropout": checkpoint.get("dropout", 0.1),
        }

        if self.stage.startswith("v8"):
            model_kwargs["gradient_checkpointing"] = False

        self.model = self.model_class(
            **model_kwargs
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )
        self.model.eval()

        self.eos_id = self.tokenizer.eos_id
        self.bos_id = self.tokenizer.bos_id

        self.retriever = LocalRetriever(ROOT)
        self.retriever.build()

        self.tool_router = ToolRouter(ROOT)
        self.instruction_examples = (
            self._load_instruction_examples()
        )
        self.history = []

    def _load_instruction_examples(self):
        candidates = [
            ROOT / "data" / "processed" / "sft_v8.jsonl",
            ROOT / "data" / "processed" / "instructions_v7.jsonl",
            ROOT / "data" / "processed" / "instructions.txt",
        ]

        examples = []

        jsonl = next(
            (path for path in candidates if path.suffix == ".jsonl" and path.exists()),
            None,
        )

        if jsonl:
            import json
            with jsonl.open(
                "r",
                encoding="utf-8",
            ) as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    question = str(
                        item.get("instruction", "")
                    ).strip()
                    answer = str(
                        item.get("response", "")
                    ).strip()
                    if question and answer:
                        examples.append(
                            (question, answer)
                        )
            return examples

        text_path = candidates[-1]
        if not text_path.exists():
            return []

        text = text_path.read_text(
            encoding="utf-8",
        )

        for chunk in text.split("<BOS>"):
            chunk = chunk.strip()
            if not chunk:
                continue

            chunk = chunk.split(
                "<EOS>",
                1,
            )[0].strip()

            match = re.search(
                r"User:\s*(.*?)\s*Assistant:\s*(.*)",
                chunk,
                re.S,
            )
            if not match:
                continue

            question = match.group(1).strip()
            answer = match.group(2).strip()

            if question and answer:
                examples.append(
                    (question, answer)
                )

        return examples

    @staticmethod
    def _normalize_text(text):
        return re.sub(
            r"[^a-z0-9\s]+",
            " ",
            text.lower(),
            flags=re.UNICODE,
        ).split()

    @staticmethod
    def _unicode_normalize(text):
        return re.sub(
            r"\s+",
            " ",
            text.strip(),
            flags=re.UNICODE,
        ).lower()

    def _best_instruction_answer(self, user_text):
        if not self.instruction_examples:
            return None

        query_tokens = self._normalize_text(user_text)
        if not query_tokens:
            return None

        query = " ".join(query_tokens)
        best = None
        best_score = 0.0

        for question, answer in self.instruction_examples:
            candidate_tokens = self._normalize_text(
                question
            )
            if not candidate_tokens:
                continue

            candidate = " ".join(candidate_tokens)
            overlap = (
                len(set(query_tokens) & set(candidate_tokens))
                / max(
                    1,
                    len(set(query_tokens) | set(candidate_tokens)),
                )
            )
            sequence = SequenceMatcher(
                None,
                query,
                candidate,
            ).ratio()

            exact = (
                self._unicode_normalize(user_text)
                == self._unicode_normalize(question)
            )

            score = (
                0.55 * overlap
                + 0.30 * sequence
                + (0.25 if exact else 0.0)
            )

            if score > best_score:
                best_score = score
                best = answer

        threshold = 0.78 if self.stage.startswith("v8") else 0.68
        return best if best_score >= threshold else None

    def _best_knowledge_answer(self, user_text):
        results = self.retriever.search(
            user_text,
            top_k=6,
            min_score=0.8,
        )

        query_words = set(
            self._normalize_text(user_text)
        )

        if len(query_words) < 2:
            return None

        best = None
        best_score = 0.0

        for item in results:
            sentences = re.split(
                r"(?<=[.!?।])\s+",
                item["text"].replace("\n", " ").strip(),
            )

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 30:
                    continue

                words = set(
                    self._normalize_text(sentence)
                )
                overlap = (
                    len(query_words & words)
                    / max(1, len(query_words))
                )

                score = (
                    0.70 * overlap
                    + 0.30 * min(
                        item["score"] / 10.0,
                        1.0,
                    )
                )

                if score > best_score:
                    best_score = score
                    best = sentence

        return best if best_score >= 0.42 else None

    def _history_text(
        self,
        history=None,
        max_turns=6,
    ):
        source = (
            self.history
            if history is None
            else history
        )

        recent = source[
            -max_turns * 2:
        ]

        return "\n".join(
            f"{role}: {text}"
            for role, text in recent
            if text
        )

    def _build_prompt(
        self,
        user_text,
        use_rag=True,
        history=None,
    ):
        sections = [
            "You are Own AI, a helpful local language model.",
            "Answer the user's request directly. Do not invent facts.",
        ]

        memory = self._history_text(
            history=history,
            max_turns=6,
        )

        if memory:
            sections.append(
                "Conversation:\n" + memory
            )

        if use_rag:
            context, _ = self.retriever.context(
                user_text,
                top_k=4,
                max_chars=3000,
            )
            if context:
                sections.append(
                    "Relevant local knowledge:\n"
                    + context
                )

        sections.append(
            f"User: {user_text}\nAssistant:"
        )

        return "\n\n".join(sections)

    def reset_memory(self):
        self.history.clear()

    def _trim_prompt_ids(self, ids):
        keep = max(
            1,
            self.model.block_size - 1,
        )
        return ids[-keep:]

    @staticmethod
    def _clean_answer(answer):
        answer = answer.replace(
            "<BOS>",
            "",
        ).replace(
            "<EOS>",
            "",
        ).strip()

        for marker in (
            "\nUser:",
            "\nAssistant:",
            "\nQuestion:",
            "\nAnswer:",
        ):
            if marker in answer:
                answer = answer.split(
                    marker,
                    1,
                )[0].strip()

        # Remove obvious short-cycle repetition produced by tiny local models.
        words = answer.split()
        if len(words) >= 12:
            for size in range(2, min(10, len(words) // 2 + 1)):
                tail = words[-size:]
                previous = words[-2 * size:-size]
                if tail == previous:
                    words = words[:-size]
                    answer = " ".join(words)
                    break

        return answer.strip()

    @torch.no_grad()
    def generate(
        self,
        user_text,
        use_rag=True,
        history=None,
        max_new_tokens=160,
        temperature=0.68,
        top_k=40,
        top_p=0.90,
        repetition_penalty=1.12,
    ):
        self.model.eval()

        tool_result = self.tool_router.route(
            user_text
        )

        if tool_result is not None:
            answer = tool_result["answer"]
            if history is None:
                self.history.extend(
                    [
                        ("User", user_text),
                        ("Assistant", answer),
                    ]
                )
            return answer

        retrieved = self._best_instruction_answer(
            user_text
        )

        if retrieved is not None:
            if history is None:
                self.history.extend(
                    [
                        ("User", user_text),
                        ("Assistant", retrieved),
                    ]
                )
            return retrieved

        knowledge = self._best_knowledge_answer(
            user_text
        )

        if knowledge is not None:
            if history is None:
                self.history.extend(
                    [
                        ("User", user_text),
                        ("Assistant", knowledge),
                    ]
                )
            return knowledge

        prompt = self._build_prompt(
            user_text,
            use_rag=use_rag,
            history=history,
        )

        prompt_ids = self.tokenizer.encode(
            prompt
        )

        generated = [
            self.bos_id,
            *self._trim_prompt_ids(prompt_ids),
        ]
        prompt_len = len(generated)

        for _ in range(max_new_tokens):
            context_ids = generated[
                -self.model.block_size:
            ]

            x = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=self.device,
            )

            logits, _ = self.model(x)
            logits = logits[:, -1, :]
            logits = logits / max(
                temperature,
                1e-5,
            )

            if repetition_penalty > 1.0:
                for token_id in set(generated[-512:]):
                    logits[0, token_id] /= repetition_penalty

            if top_k:
                k = min(
                    int(top_k),
                    logits.size(-1),
                )
                values, _ = torch.topk(
                    logits,
                    k,
                )
                logits[
                    logits < values[:, [-1]]
                ] = float("-inf")

            if top_p and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True,
                )
                probabilities = torch.softmax(
                    sorted_logits,
                    dim=-1,
                )
                cumulative = torch.cumsum(
                    probabilities,
                    dim=-1,
                )
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(
                    logits,
                    float("-inf"),
                )
                logits.scatter_(
                    1,
                    sorted_indices,
                    sorted_logits,
                )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            next_id = torch.multinomial(
                probabilities,
                1,
            ).item()

            generated.append(next_id)

            if next_id == self.eos_id:
                break

        answer = self.tokenizer.decode(
            generated[prompt_len:]
        )
        answer = self._clean_answer(
            answer
        )

        if not answer:
            answer = (
                "මට ඒකට තව හොඳ පිළිතුරක් "
                "දීමට වැඩි training knowledge එකක් අවශ්‍යයි."
                if re.search(r"[අ-෴]", user_text)
                else
                "I do not have a useful answer for that yet."
            )

        if history is None:
            self.history.extend(
                [
                    ("User", user_text),
                    ("Assistant", answer),
                ]
            )

        return answer

    def info(self):
        params = sum(
            p.numel()
            for p in self.model.parameters()
        )

        return {
            "stage": self.stage,
            "device": self.device,
            "parameters": params,
            "vocabulary": self.tokenizer.vocab_size,
            "context": self.model.block_size,
            "embedding": self.model.d_model,
            "layers": len(self.model.blocks),
            "retrieval_documents": len(
                self.retriever.documents
            ),
            "checkpoint": str(
                self.checkpoint_path.relative_to(ROOT)
            ),
        }
