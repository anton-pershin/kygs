import json
from dataclasses import dataclass

from kygs.message_provider import MessageCollection
from kygs.split_strategy import Metadata

from rally.interaction import request_based_on_prompts  # type: ignore # pylint: disable=import-error # isort: skip
from rally.llm import Llm  # type: ignore # pylint: disable=import-error # isort: skip
from rally.thinking import THINKING_REMOVERS  # type: ignore # pylint: disable=import-error # isort: skip


@dataclass
class Summary:
    text: str
    metadata: Metadata


def summarize_posts(
    message_splits: list[MessageCollection],
    llm: Llm,
    system_prompt: str,
    user_prompt_template: str,
    verbose: bool = False,
) -> list[Summary]:
    user_prompts = []
    metadatas: list[Metadata] = []
    for message_collection in message_splits:
        if len(message_collection.messages) == 0:
            continue

        messages_for_time_interval = [
            {
                "author": m.author,
                "time": m.time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": m.text,
            }
            for m in message_collection.messages
        ]
        messages_as_json = json.dumps(messages_for_time_interval)
        messages_as_json = json.JSONDecoder().decode(messages_as_json)
        user_prompt = user_prompt_template.format(
            messages_as_json=messages_as_json,
        )
        user_prompts.append(user_prompt)
        metadatas.append(message_collection.metadata)

    return _run_summarization_via_llm(
        llm=llm,
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        metadatas=metadatas,
        progress_title=(
            f"Summarizing {len(user_prompts)} message collections" if verbose else None
        ),
    )


def summarize_summaries(
    summary_collection: list[Summary],
    llm: Llm,
    system_prompt: str,
    user_prompt_template: str,
    max_characters_in_prompt: int,
    verbose: bool = False,
) -> list[Summary]:
    user_prompts = []
    metadatas: list[Metadata] = []
    n_characters = 0
    cur_summaries_buffer: list[Summary] = []
    for summary in summary_collection:
        if n_characters + len(summary.text) > max_characters_in_prompt:
            user_prompt = _compose_user_prompt_for_summary_collection(
                summary_collection=cur_summaries_buffer,
                user_prompt_template=user_prompt_template,
            )
            user_prompts.append(user_prompt)
            metadatas.append(_merge_summary_metadatas(cur_summaries_buffer))

            n_characters = 0
            cur_summaries_buffer = []

        cur_summaries_buffer.append(summary)
        n_characters += len(summary.text)

    if not cur_summaries_buffer:
        raise SummarizationException(
            "Got empty summary buffer after splitting the summary collection"
        )

    user_prompt = _compose_user_prompt_for_summary_collection(
        summary_collection=cur_summaries_buffer,
        user_prompt_template=user_prompt_template,
    )
    user_prompts.append(user_prompt)
    metadatas.append(_merge_summary_metadatas(cur_summaries_buffer))

    return _run_summarization_via_llm(
        llm=llm,
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        metadatas=metadatas,
        progress_title=(
            f"Summarizing {len(user_prompts)} summary collections" if verbose else None
        ),
    )


def _merge_summary_metadatas(summaries: list[Summary]) -> Metadata:
    merged = summaries[0].metadata
    for s in summaries[1:]:
        merged = merged.merge(s.metadata)
    return merged


def _compose_user_prompt_for_summary_collection(
    summary_collection: list[Summary],
    user_prompt_template: str,
) -> str:
    summaries_for_prompt = []
    for s in summary_collection:
        entry: dict = {"summary": s.text}
        entry.update(s.metadata.to_json_dict())
        summaries_for_prompt.append(entry)
    summaries_as_json = json.dumps(summaries_for_prompt)
    summaries_as_json = json.JSONDecoder().decode(summaries_as_json)
    user_prompt = user_prompt_template.format(
        summaries_as_json=summaries_as_json,
    )
    return user_prompt


def _run_summarization_via_llm(
    llm: Llm,
    system_prompt: str,
    user_prompts: list[str],
    metadatas: list[Metadata],
    progress_title: str | None = None,
) -> list[Summary]:
    responses: list[str] = request_based_on_prompts(
        llm_server_url=llm.url,
        max_concurrent_requests=llm.max_concurrent_requests,
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        authorization=llm.authorization,
        model=llm.model,
        progress_title=progress_title,
    )
    responses_wo_thinking = [THINKING_REMOVERS[llm.model_family](p) for p in responses]

    summary_collection = [
        Summary(text, metadata)
        for text, metadata in zip(responses_wo_thinking, metadatas)
    ]
    return summary_collection


class SummarizationException(Exception):
    pass
