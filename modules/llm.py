"""Local generative source classification; weights load only for nonempty batches."""
import json
import re
from dataclasses import dataclass

from .contracts import MetadataBatch, SourceResult, SourceVocabulary, empty_source_table
from .generative import allowed_source_names, invalid_response_evidence, system_prompt as llm_system_prompt

DEFAULT_LLM_MODEL = "mistralai/Ministral-3-8B-Instruct-2512"


@dataclass(frozen=True)
class LLMConfig:
    model: str = DEFAULT_LLM_MODEL
    revision: str | None = None
    device: int = 0
    batch_size: int = 1
    max_new_tokens: int = 16

    def __post_init__(self):
        if self.device < -1 or self.batch_size < 1 or self.max_new_tokens < 1:
            raise ValueError("device must be >= -1; batch_size and max_new_tokens must be positive")


def llm_messages(system_prompt, record):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "Classify the biological host or environmental source "
            "of this sequencing sample.\n\nMETADATA\n"
            f"{record}\n\nReturn exactly one controlled source label."
        )},
    ]


def parse_llm_response(response, source_names):
    """Accept only a complete label (or a simple JSON object), never prose."""
    candidate = response.strip()
    if candidate.startswith("{"):
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            parsed = None
        candidate = parsed["source"] if isinstance(parsed, dict) and set(parsed) == {"source"} else None
    elif len(candidate) >= 2 and candidate[0] in "\"'`" and candidate[-1] == candidate[0]:
        candidate = candidate[1:-1].strip()

    allowed = allowed_source_names(source_names)
    if isinstance(candidate, str):
        if candidate in allowed:
            return candidate, ""
        normalized = lambda label: re.sub(r"[\s-]+", "_", label.strip().casefold())
        matches = [label for label in allowed if normalized(label) == normalized(candidate)]
        if len(matches) == 1:
            return matches[0], ""
    return "unknown", invalid_response_evidence(response, "invalid_llm_output")


def load_local_llm(config: LLMConfig):
    """Load Ministral 3 only when unresolved source rows need inference."""
    import torch

    if config.device < -1:
        raise ValueError("--device must be >= -1")
    if config.device >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use --device -1 for CPU.")
        if config.device >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {config.device} does not exist ({torch.cuda.device_count()} available).")
    device = torch.device("cpu" if config.device == -1 else f"cuda:{config.device}")
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

    revision = {"revision": config.revision} if config.revision is not None else {}
    tokenizer = MistralCommonBackend.from_pretrained(config.model, **revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("LLM tokenizer needs a pad token or EOS token for batched generation.")
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"dtype": "auto", "low_cpu_mem_usage": True, **revision}
    if config.device >= 0:
        # Keep the FP8 checkpoint on the specifically requested GPU.
        model_kwargs["device_map"] = config.device
    model = Mistral3ForConditionalGeneration.from_pretrained(config.model, **model_kwargs)
    if config.device == -1:
        model.to(device)
    model.eval()
    return tokenizer, model, device


def run(batch: MetadataBatch, sources: SourceVocabulary, config: LLMConfig = LLMConfig()) -> SourceResult:
    """Classify every supplied row; malformed responses are explicit unknown calls."""
    records = batch.records.tolist()
    source_labels, source_names = list(sources.labels), list(sources.names)
    output = empty_source_table(sources, batch.metadata.index)
    output["source_method"] = "llm"
    if not records:
        return SourceResult(output)

    import torch
    tokenizer, model, device = load_local_llm(config)
    system_prompt = llm_system_prompt(source_labels, source_names)
    for start in range(0, len(records), config.batch_size):
        conversations = [llm_messages(system_prompt, record)
                         for record in records[start:start + config.batch_size]]
        encoded = tokenizer.apply_chat_template(
            conversations, tokenize=True, padding=True, return_tensors="pt", return_dict=True,
        )
        inputs = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
        if "input_ids" not in inputs:
            raise RuntimeError("Mistral tokenizer did not return input_ids.")
        generation_kwargs = {"do_sample": False, "max_new_tokens": config.max_new_tokens}
        if tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
        # Every left-padded row has the same encoded prompt width.
        generated_only = generated[:, inputs["input_ids"].shape[1]:]
        responses = [tokenizer.decode(row.tolist(), skip_special_tokens=True).strip()
                     for row in generated_only]
        if len(responses) != len(conversations):
            raise RuntimeError("LLM returned an unexpected number of responses.")
        for offset, response in enumerate(responses):
            label, evidence = parse_llm_response(response, source_names)
            output.loc[batch.metadata.index[start + offset], ["best_hit", "source_evidence"]] = [label, evidence]
        del encoded, inputs, generated, generated_only, responses, conversations
        print(f"Batch {start // config.batch_size + 1}: local Ministral source classification done", flush=True)
    result = SourceResult(output)
    result.validate(batch, sources)
    return result
