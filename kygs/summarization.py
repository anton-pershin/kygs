import datetime
from dataclasses import dataclass
import json

from rally.llm import Llm
from rally.interaction import request_based_on_prompts
from rally.thinking import THINKING_REMOVERS

from kygs.message_provider import MessageProvider
from kygs.utils.typing import TimeUnit


@dataclass
class Summary:
    text: str
    start_dt: datetime.datetime
    end_dt: datetime.datetime


def summarize_posts(
    mp: MessageProvider,
    summarization_time_interval: TimeUnit,
    llm: Llm,
    system_prompt: str,
    user_prompt_template: str,
) -> list[Summary]:
    message_splits = mp.split_by(summarization_time_interval)
    user_prompts = []
    start_dts = []
    end_dts = []
    for i, message_collection in enumerate(message_splits):
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
        start_dts.append(message_collection.start_dt)
        end_dts.append(message_collection.end_dt)

    user_prompts = user_prompts[:5]
    start_dts = start_dts[:5]
    end_dts = end_dts[:5]

    return _run_summarization_via_llm(
        llm=llm,
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        start_dts=start_dts,
        end_dts=end_dts,
    )


def summarize_summaries(
    summary_collection: list[Summary],
    llm: Llm,
    system_prompt: str,
    user_prompt_template: str,
    max_characters_in_prompt: int,
):
    # TODO: stopped here. This function is done. Just need a script to run it and check
    user_prompts = []
    start_dts = []
    end_dts = []
    n_characters = 0
    cur_summaries_buffer = []
    cur_start_dt = None
    cur_end_dt = None
    for summary in summary_collection:
        n_characters += len(summary.text)
        if n_characters > max_characters_in_prompt:
            user_prompt = _compose_user_prompt_for_summary_collection(
                summary_collection=cur_summaries_buffer,
                user_prompt_template=user_prompt_template,
            )
            user_prompts.append(user_prompt)
            start_dts.append(cur_start_dt)
            end_dts.append(cur_end_dt)
            cur_start_dt = None
            cur_end_dt = None
        else:
            cur_summaries_buffer.append(summary)
            if cur_start_dt is None:
                cur_start_dt = summary.start_dt
            cur_end_dt = summary.end_dt

    return _run_summarization_via_llm(
        llm=llm,
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        start_dts=start_dts,
        end_dts=end_dts,
    )


def _compose_user_prompt_for_summary_collection(
    summary_collection: list[Summary],
    user_prompt_template: str,
) -> str:
    summaries_for_time_interval = [
        {
            "start_time": s.start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": s.end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": s.text,
        }
        for s in summary_collection
    ]
    summaries_as_json = json.dumps(summaries_for_time_interval)
    summaries_as_json = json.JSONDecoder().decode(summaries_as_json)
    user_prompt = user_prompt_template.format(
        summaries_as_json=summaries_as_json,
    )
    return user_prompt


def _run_summarization_via_llm(
    llm: Llm,
    system_prompt: str,
    user_prompts: list[str],
    start_dts: list[datetime.datetime],
    end_dts: list[datetime.datetime],
) -> list[Summary]:
    responses: list[str] = request_based_on_prompts(
        llm_server_url=llm.url,
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        authorization=llm.authorization,
        model=llm.model,
    )
    responses_wo_thinking = [THINKING_REMOVERS[llm.model_family](p) for p in responses]

    summary_collection = [
        Summary(text, start_dt, end_dt) 
        for text, start_dt, end_dt in zip(responses_wo_thinking, start_dts, end_dts)
    ]
    return summary_collection
    

