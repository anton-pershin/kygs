import json
from abc import ABC, abstractmethod

from rally.interaction import request_based_on_prompts
from rally.llm import Llm
from rally.thinking import THINKING_REMOVERS

from kygs.message_provider import Message, MessageCollection
from kygs.metadata import Metadata, MetadataFieldCollision
from kygs.summarization.base import BaseSummarization, Summary


class BaseSummarizationPrompt(ABC):
    def __init__(
        self,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def __call__(self, message_collection: MessageCollection) -> str:
        """Build a user prompt for a given collection of messages

        Args:
            message_collection: Collection of messages to summarize

        Returns:
            User prompt as a string
        """
        messages_for_split = [
            self.turn_message_to_dict(m) for m in message_collection.messages
        ]
        messages_as_json = json.dumps(messages_for_split, ensure_ascii=False)
        user_prompt = self.user_prompt_template.format(
            messages_as_json=messages_as_json,
        )
        return user_prompt

    @abstractmethod
    def turn_message_to_dict(self, message: Message) -> dict[str, str]:
        """Transform a Message object to a string-valued dict suitable for jsonsifying

        Args:
            message: Message object

        Returns:
            Dict representation of message
        """


class AnnotatedSummarizationPrompt(BaseSummarizationPrompt):
    def __init__(
        self,
        system_prompt: str,
        user_prompt_template: str,
        labels: dict[str, str],
    ) -> None:
        super().__init__(system_prompt, user_prompt_template)
        self.labels = labels

    def __call__(self, message_collection: MessageCollection) -> str:
        messages_for_split = [
            self.turn_message_to_dict(m) for m in message_collection.messages
        ]
        messages_as_json = json.dumps(messages_for_split, ensure_ascii=False)
        labels_formatted = "\n".join(
            [f"- **{name}**. {descr}" for name, descr in self.labels.items()]
        )
        user_prompt = self.user_prompt_template.format(
            messages_as_json=messages_as_json,
            labels=labels_formatted,
        )
        return user_prompt

    def turn_message_to_dict(self, message: Message) -> dict[str, str]:
        return {
            "message": message.text,
        }


class OnlyMessageSummarizationPrompt(BaseSummarizationPrompt):
    def turn_message_to_dict(self, message: Message) -> dict[str, str]:
        return {
            "message": message.text,
        }


class TimeBasedSummarizationPrompt(BaseSummarizationPrompt):
    def turn_message_to_dict(self, message: Message) -> dict[str, str]:
        return {
            "time": message.time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": message.text,
        }


class BaseSummaryBuilder(ABC):
    @abstractmethod
    def __call__(self, text: str, metadata: Metadata) -> Summary:
        """Build a summary based on the text which is typically an LLM response
        and message collection metadata

        Args:
            text: Raw summary text
            metadata: Metadata of the collection of messages which has been summarized

        Returns:
            Summary with parsed summary text and filled metadata
        """


class AnnotatedSummaryBuilder(BaseSummaryBuilder):
    def __init__(
        self,
        metadata_key: str = "annotation_labels",
        labels_key: str = "labels",
    ) -> None:
        self.metadata_key = metadata_key
        self.labels_key = labels_key

    def __call__(self, text: str, metadata: Metadata) -> Summary:
        parsed = json.loads(text)

        if self.metadata_key in metadata:
            raise MetadataFieldCollision(
                f"Annotation labels collision: "
                f"'{self.metadata_key}' field already exists."
            )

        original_class = type(metadata)
        class_name = f"Annotated{original_class.__name__}"
        parent_strategies = getattr(original_class, "_merge_strategies", {})
        new_strategies = {**parent_strategies, self.metadata_key: "union"}

        AnnotatedMetadataClass = type(
            class_name, (original_class,), {"_merge_strategies": new_strategies}
        )

        enriched_dict = {**metadata, self.metadata_key: parsed[self.labels_key]}
        enriched_metadata = AnnotatedMetadataClass.__new__(
            AnnotatedMetadataClass
        )  # type: ignore[call-overload]
        dict.__init__(enriched_metadata, enriched_dict)

        return Summary(text=parsed["summary"], metadata=enriched_metadata)


class PlainSummaryBuilder(BaseSummaryBuilder):
    def __call__(self, text: str, metadata: Metadata) -> Summary:
        return Summary(text=text, metadata=metadata)


class DirectSummarization(BaseSummarization):
    def __init__(
        self,
        llm: Llm,
        summarization_prompt: BaseSummarizationPrompt,
        summary_builder: BaseSummaryBuilder,
        verbose: bool = False,
    ):
        self.llm: Llm = llm
        self.summarization_prompt: BaseSummarizationPrompt = summarization_prompt
        self.summary_builder = summary_builder
        self.verbose = verbose

    def __call__(self, message_collections: list[MessageCollection]) -> list[Summary]:
        user_prompts = []
        metadatas: list[Metadata] = []
        for message_collection in message_collections:
            if len(message_collection.messages) == 0:
                continue

            user_prompt = self.summarization_prompt(message_collection)
            user_prompts.append(user_prompt)
            metadatas.append(message_collection.metadata)

        text_summaries: list[str] = self._run_summarization_via_llm(
            user_prompts=user_prompts,
            progress_title=(
                f"Summarizing {len(user_prompts)} message collections"
                if self.verbose
                else None
            ),
        )
        summaries = [
            self.summary_builder(text=t, metadata=md)
            for t, md in zip(text_summaries, metadatas)
        ]
        return summaries

    def _run_summarization_via_llm(
        self,
        user_prompts: list[str],
        progress_title: str | None = None,
    ) -> list[str]:
        responses: list[str] = request_based_on_prompts(
            llm_server_url=self.llm.url,
            max_concurrent_requests=self.llm.max_concurrent_requests,
            system_prompt=self.summarization_prompt.system_prompt,
            user_prompts=user_prompts,
            authorization=self.llm.authorization,
            model=self.llm.model,
            progress_title=progress_title,
        )
        responses_wo_thinking = [
            THINKING_REMOVERS[self.llm.model_family](p) for p in responses
        ]

        return responses_wo_thinking
