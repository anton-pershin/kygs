from rally.llm import Llm

from kygs.message_provider import MessageCollection
from kygs.summarization.base import BaseSummarization, Summary, to_message_collection
from kygs.summarization.direct import (
    BaseSummarizationPrompt,
    BaseSummaryBuilder,
    DirectSummarization,
)


class RecursiveSummarization(BaseSummarization):
    def __init__(
        self,
        llm: Llm,
        original_message_summarization_prompt: BaseSummarizationPrompt,
        partial_summary_summarization_prompt: BaseSummarizationPrompt,
        original_message_summary_builder: BaseSummaryBuilder,
        partial_summary_builder: BaseSummaryBuilder,
        max_characters_in_prompt: int,
        verbose: bool = False,
    ):
        self.original_message_summarization: DirectSummarization = DirectSummarization(
            llm=llm,
            summarization_prompt=original_message_summarization_prompt,
            summary_builder=original_message_summary_builder,
            verbose=verbose,
        )
        self.partial_summary_summarization: DirectSummarization = DirectSummarization(
            llm=llm,
            summarization_prompt=partial_summary_summarization_prompt,
            summary_builder=partial_summary_builder,
            verbose=verbose,
        )
        self.verbose = verbose
        self.max_characters_in_prompt = max_characters_in_prompt

    def __call__(self, message_collections: list[MessageCollection]) -> list[Summary]:
        # Build a summary for each message collection
        summaries = []
        for mc in message_collections:
            summary: Summary = self._summarize_recursively(mc)
            summaries.append(summary)

        return summaries

    def _summarize_recursively(self, message_collection: MessageCollection) -> Summary:
        cur_message_collection = message_collection
        original_message_mode = True
        while len(cur_message_collection.messages) > 1:
            # Partition a collection into smaller pieces fit into
            # the prompt size restriction
            partitioned_mc: list[MessageCollection] = _partition_collection(
                cur_message_collection, self.max_characters_in_prompt
            )

            # Ensure the recursion converges
            if len(partitioned_mc) >= len(cur_message_collection.messages):
                raise LackOfConvergenceException()

            # Summarize each partition separately
            # Distinguish between two cases:
            # - original message summarization (original_message_mode = True)
            # - partial summary summarization (original_message_mode = False)
            if original_message_mode:
                summaries: list[Summary] = self.original_message_summarization(
                    partitioned_mc
                )
                original_message_mode = False
            else:
                summaries = self.partial_summary_summarization(partitioned_mc)

            # Transform summaries to a message collection
            cur_message_collection = to_message_collection(summaries)

        # Transform message collection back to summary
        # At this point, we should have only one message/summary
        summary = Summary(
            text=cur_message_collection.messages[0].text,
            metadata=cur_message_collection.metadata,
        )

        return summary


def _partition_collection(
    mc: MessageCollection,
    max_chars_in_prompt: int,
) -> list[MessageCollection]:
    if not mc.messages:
        return []
    mc_partitions = []
    cur_chars = 0
    cur_begin_i = 0
    for i, m in enumerate(mc.messages):
        m_chars = len(m.text)
        if m_chars > max_chars_in_prompt:
            raise OutOfContextLengthException(
                f"Message is too long ({m_chars} while the limit is "
                f"{max_chars_in_prompt})"
            )
        if cur_chars + m_chars > max_chars_in_prompt:
            mc_partitions.append(
                MessageCollection(
                    messages=mc.messages[cur_begin_i:i], metadata=mc.metadata
                )
            )
            cur_chars = 0
            cur_begin_i = i

        cur_chars += m_chars

    # Add the last (possibly unfinished) piece
    mc_partitions.append(
        MessageCollection(messages=mc.messages[cur_begin_i:], metadata=mc.metadata)
    )

    return mc_partitions


class OutOfContextLengthException(Exception):
    pass


class LackOfConvergenceException(Exception):
    pass
