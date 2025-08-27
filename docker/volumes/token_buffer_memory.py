import logging
import re
from collections.abc import Sequence
from typing import Optional

from sqlalchemy import select

from core.app.app_config.features.file_upload.manager import FileUploadConfigManager
from core.file import file_manager
from core.model_manager import ModelInstance
from core.model_runtime.entities import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageRole,
    TextPromptMessageContent,
    UserPromptMessage,
)
from core.model_runtime.entities.message_entities import PromptMessageContentUnionTypes
from core.prompt.utils.extract_thread_messages import extract_thread_messages
from extensions.ext_database import db
from factories import file_factory
from models.model import AppMode, Conversation, Message, MessageFile
from models.workflow import Workflow, WorkflowRun

logger = logging.getLogger(__name__)


class TokenBufferMemory:
    def __init__(
        self,
        conversation: Conversation,
        model_instance: ModelInstance,
    ) -> None:
        self.conversation = conversation
        self.model_instance = model_instance

    def get_history_prompt_messages(
        self, max_token_limit: int = 2000, message_limit: Optional[int] = None
    ) -> Sequence[PromptMessage]:
        """
        Get history prompt messages.
        :param max_token_limit: max token limit
        :param message_limit: message limit
        """
        app_record = self.conversation.app

        # fetch limited messages, and return reversed
        stmt = (
            select(Message).where(Message.conversation_id == self.conversation.id).order_by(Message.created_at.desc())
        )

        if message_limit and message_limit > 0:
            message_limit = min(message_limit, 500)
        else:
            message_limit = 500

        stmt = stmt.limit(message_limit)

        messages = db.session.scalars(stmt).all()

        # instead of all messages from the conversation, we only need to extract messages
        # that belong to the thread of last message
        thread_messages = extract_thread_messages(messages)

        # for newly created message, its answer is temporarily empty, we don't need to add it to memory
        if thread_messages and not thread_messages[0].answer and thread_messages[0].answer_tokens == 0:
            thread_messages.pop(0)

        messages = list(reversed(thread_messages))

        prompt_messages: list[PromptMessage] = []
        for message in messages:
            files = db.session.query(MessageFile).where(MessageFile.message_id == message.id).all()
            if files:
                file_extra_config = None
                if self.conversation.mode in {AppMode.AGENT_CHAT, AppMode.COMPLETION, AppMode.CHAT}:
                    file_extra_config = FileUploadConfigManager.convert(self.conversation.model_config)
                elif self.conversation.mode in {AppMode.ADVANCED_CHAT, AppMode.WORKFLOW}:
                    workflow_run = db.session.scalar(
                        select(WorkflowRun).where(WorkflowRun.id == message.workflow_run_id)
                    )
                    if not workflow_run:
                        raise ValueError(f"Workflow run not found: {message.workflow_run_id}")
                    workflow = db.session.scalar(select(Workflow).where(Workflow.id == workflow_run.workflow_id))
                    if not workflow:
                        raise ValueError(f"Workflow not found: {workflow_run.workflow_id}")
                    file_extra_config = FileUploadConfigManager.convert(workflow.features_dict, is_vision=False)
                else:
                    raise AssertionError(f"Invalid app mode: {self.conversation.mode}")

                detail = ImagePromptMessageContent.DETAIL.LOW
                if file_extra_config and app_record:
                    file_objs = file_factory.build_from_message_files(
                        message_files=files, tenant_id=app_record.tenant_id, config=file_extra_config
                    )
                    if file_extra_config.image_config and file_extra_config.image_config.detail:
                        detail = file_extra_config.image_config.detail
                else:
                    file_objs = []

                if not file_objs:
                    prompt_messages.append(UserPromptMessage(content=message.query))
                else:
                    prompt_message_contents: list[PromptMessageContentUnionTypes] = []
                    for file in file_objs:
                        prompt_message = file_manager.to_prompt_message_content(
                            file,
                            image_detail_config=detail,
                        )
                        prompt_message_contents.append(prompt_message)
                    prompt_message_contents.append(TextPromptMessageContent(data=message.query))

                    prompt_messages.append(UserPromptMessage(content=prompt_message_contents))

            else:
                prompt_messages.append(UserPromptMessage(content=message.query))

            # Check if assistant message has files
            assistant_files = db.session.query(MessageFile).filter(
                MessageFile.message_id == message.id,
                MessageFile.belongs_to == "assistant"
            ).all()

            if assistant_files:
                assistant_prompt_message_contents: list[PromptMessageContentUnionTypes] = []

                # Build File objects from assistant MessageFile records
                for message_file in assistant_files:
                    if message_file.upload_file_id:
                        # Build File object using file_factory approach
                        mapping = {
                            "transfer_method": message_file.transfer_method,
                            "url": message_file.url,
                            "id": message_file.id,
                            "type": message_file.type,
                            "tool_file_id": message_file.upload_file_id,  # For tool files
                        }
                        try:
                            from factories.file_factory import build_from_mapping
                            file = build_from_mapping(
                                mapping=mapping,
                                tenant_id=app_record.tenant_id if app_record else "",
                            )

                            # Convert to prompt message content using file_manager
                            prompt_message = file_manager.to_prompt_message_content(
                                file,
                                image_detail_config=ImagePromptMessageContent.DETAIL.LOW,
                            )
                            assistant_prompt_message_contents.append(prompt_message)

                        except Exception as e:
                            logger.warning(f"Failed to build file from assistant message_file {message_file.id}: {e}")
                            # Fallback to text description if file building fails
                            assistant_prompt_message_contents.append(
                                TextPromptMessageContent(data=f"[Assistant file: {message_file.id}]")
                            )

                # Add text content if exists
                if message.answer:
                    # Remove markdown image links since we have proper image content
                    text_content = re.sub(r'!\[.*?\]\(.*?\)', '', message.answer).strip()
                    if text_content:
                        assistant_prompt_message_contents.append(
                            TextPromptMessageContent(data=text_content)
                        )

                prompt_messages.append(AssistantPromptMessage(content=assistant_prompt_message_contents))
            else:
                prompt_messages.append(AssistantPromptMessage(content=message.answer))

        if not prompt_messages:
            return []

        # prune the chat message if it exceeds the max token limit
        curr_message_tokens = self.model_instance.get_llm_num_tokens(prompt_messages)

        if curr_message_tokens > max_token_limit:
            while curr_message_tokens > max_token_limit and len(prompt_messages) > 1:
                prompt_messages.pop(0)
                curr_message_tokens = self.model_instance.get_llm_num_tokens(prompt_messages)

        return prompt_messages

    def get_history_prompt_text(
        self,
        human_prefix: str = "Human",
        ai_prefix: str = "Assistant",
        max_token_limit: int = 2000,
        message_limit: Optional[int] = None,
    ) -> str:
        """
        Get history prompt text.
        :param human_prefix: human prefix
        :param ai_prefix: ai prefix
        :param max_token_limit: max token limit
        :param message_limit: message limit
        :return:
        """
        prompt_messages = self.get_history_prompt_messages(max_token_limit=max_token_limit, message_limit=message_limit)

        string_messages = []
        for m in prompt_messages:
            if m.role == PromptMessageRole.USER:
                role = human_prefix
            elif m.role == PromptMessageRole.ASSISTANT:
                role = ai_prefix
            else:
                continue

            if isinstance(m.content, list):
                inner_msg = ""
                for content in m.content:
                    if isinstance(content, TextPromptMessageContent):
                        inner_msg += f"{content.data}\n"
                    elif isinstance(content, ImagePromptMessageContent):
                        inner_msg += "[image]\n"

                string_messages.append(f"{role}: {inner_msg.strip()}")
            else:
                message = f"{role}: {m.content}"
                string_messages.append(message)

        return "\n".join(string_messages)
