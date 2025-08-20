from typing import List, Callable, Optional

from rally.llm import Llm
from rally.interaction import request_based_on_prompts
from kygs.message_provider import Message
from kygs.annotation.base import MessageAnnotation


LlmResponseParser = Callable[[str, dict[str, str]], Optional[str]]


class SimpleLabelParser:
    def __call__(self, llm_response: str, labels: dict[str, str]) -> Optional[str]:
        label = llm_response.strip(" \n\t\r*")
        return label if label in labels else None


class AnnotationViaLlm(MessageAnnotation):
    def __init__(
        self,
        llm: Llm,
        system_prompt: str, 
        user_prompt_template: str,
        response_parser: LlmResponseParser,
        verbose: bool = False
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.response_parser = response_parser
        self.verbose = verbose

    def __call__(self, messages: list[Message], labels: dict[str, str]) -> list[Optional[str]]:
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
            max_concurrent_requests=self.llm.max_concurrent_requests,
            system_prompt=self.system_prompt,
            user_prompts=user_prompts,
            authorization=self.llm.authorization,
            model=self.llm.model,
            progress_title="Annotating messages" if self.verbose else None,
        )
        
        # Parse responses into labels using the provided parser
        annotations = [self.response_parser(response, labels) for response in responses]
        
        return annotations
