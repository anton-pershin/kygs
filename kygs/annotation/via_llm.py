from typing import List

from rally.llm import Llm
from rally.interaction import request_based_on_prompts
from kygs.message_provider import Message
from kygs.annotation.base import MessageAnnotation


class AnnotationViaLlm(MessageAnnotation):
    def __init__(self, llm: Llm, system_prompt: str, user_prompt_template: str, verbose=False):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.verbose = verbose

    def __call__(self, messages: list[Message], labels: dict[str, str]) -> list[str]:
        # Prepare prompts for all messages
        user_prompts = []
        for message in messages:
            prompt = self.user_prompt_template.format(
                content=message.text,
                labels="\n".join(
                    [f"- **{name}**. {descr}" for name, descr in labels.items()]
                ),
            )
            user_prompts.append(prompt)

        # Get responses for all prompts at once
        responses = request_based_on_prompts(
            llm_server_url=self.llm.url,
            system_prompt=self.system_prompt,
            user_prompts=user_prompts,
            authorization=self.llm.authorization,
            model=self.llm.model,
            progress_title="Annotating messages" if self.verbose else None,
        )
        
        # Parse responses into labels
        annotations = []
        for label in map(lambda s: s.strip(" \n\t\r*"), responses):
            if label not in labels:
                raise ValueError(f"LLM returned invalid label: {label}. Valid labels are: {labels}")
            annotations.append(label)
        
        return annotations
