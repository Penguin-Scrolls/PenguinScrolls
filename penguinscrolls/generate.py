import math
from multiprocessing.dummy import Pool as ThreadPool
from typing import (
    Any,
    Callable,
    List,
    NewType,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)

import fire
import numpy as np
import openai
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, field_validator
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerFast,
)

from .defs import INPUT_COL, KEY_COL
from .util import get_penguin_dataset

T = TypeVar("T")
R = TypeVar("R")


class ModelConfig(BaseModel):
    """Configuration for model settings."""

    model_config = ConfigDict(protected_namespaces=())
    model_name_or_path: str
    framework: str  # 'hf', 'vllm', or 'openai'
    temperature: float = 0.7
    top_p: float = 1.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    tensor_parallel_size: int = 1
    system_prompt: Optional[str] = None
    torch_dtype: str = "auto"
    do_sample: bool = True

    @field_validator("framework")
    @classmethod
    def validate_framework(cls, v):
        if v not in ["hf", "vllm", "openai"]:
            raise ValueError(f"Unsupported framework: {v}")
        return v

    @field_validator("torch_dtype")
    @classmethod
    def validate_dtype(cls, v):
        if v != "auto" and not hasattr(torch, v):
            raise ValueError(f"Invalid torch dtype: {v}")
        return v


class GenerateConfig(BaseModel):
    """Configuration for generation settings."""

    model_config = ConfigDict(protected_namespaces=())
    output_file: str
    model: ModelConfig  # type: ignore
    batch_size: int = 1


class Message(TypedDict):
    role: str
    content: str


MessageList = NewType("MessageList", List[Message])
Prompt = Union[MessageList, str]


class Response(BaseModel):
    """Response class for model inference."""

    response: Optional[str] = None
    error: Optional[str] = None
    tokens: Optional[List[int]] = None
    truncated: bool = False


def vectorize(
    func: Callable[[Any, T], R]
) -> Callable[[Any, Union[T, List[T]]], Union[R, List[R]]]:
    """Decorator to vectorize input/output for the InputProcesser."""

    def wrapper(*args, **kwargs) -> Union[R, List[R]]:
        self, prompt = args
        if isinstance(prompt, list):
            return [func(self, p) for p in prompt]
        return func(self, prompt)

    return wrapper


class InputProcesser:
    """Class for processing input for tokenization and formatting."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

    @vectorize
    def __call__(self, prompt: Prompt) -> str:
        """Preprocess input for tokenization and formatting."""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt  # Assume it's already a list of messages

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted_prompt


class ResponseVector:
    """Class for storing result vectors."""

    def __init__(self, batch: BatchEncoding, max_length: int, padding_side: str):
        self.batch = batch
        self.max_length = max_length
        self.batch_size = self.batch["input_ids"].shape[0]
        self.length_vec = batch["attention_mask"].sum(axis=1)
        self.mask = self.length_vec < self.max_length
        self.result_vec = [Response()] * self.batch_size
        self.padding_side = padding_side
        self.valid_items = self.mask.sum().item()
        for idx in torch.nonzero(self.mask == 0).flatten().tolist():
            self.result_vec[idx].error = "longer than max_length"

    def get_result(self) -> List[Response]:
        return self.result_vec

    def get_effective_batch(self) -> Optional[BatchEncoding]:
        if self.valid_items == self.batch_size:
            return self.batch
        if self.valid_items == 0:
            return None
        input_ids = self.batch["input_ids"][self.mask, :]  # type: ignore
        attention_mask = self.batch["attention_mask"][self.mask, :]  # type: ignore
        longest_size = int(attention_mask.sum(axis=1).max())
        if self.padding_side == "right":
            return {
                "input_ids": input_ids[:, :longest_size],
                "attention_mask": attention_mask[:, :longest_size],
            }  # type: ignore
        else:
            return {
                "input_ids": input_ids[:, -longest_size:],
                "attention_mask": attention_mask[:, -longest_size:],
            }  # type: ignore

    def set_effective_result(self, result: List[Response]):
        indices: List[int] = torch.nonzero(self.mask).flatten().tolist()
        for i, r in enumerate(result):
            index = indices[i]
            self.result_vec[index] = r


def processes_input_to_tensor(
    tokenizer: PreTrainedTokenizerFast,
    prompt: List[Prompt],
    max_length: int,
) -> ResponseVector:
    input_processor = InputProcesser(tokenizer)
    formatted_prompt = input_processor(prompt)
    batch = tokenizer(
        formatted_prompt, return_tensors="pt", padding=True, truncation=False
    )
    return ResponseVector(batch, max_length, tokenizer.padding_side)


class BaseInference:
    """Base class for model inference."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def generate(self, prompt: List[Prompt]) -> List[Response]:
        raise NotImplementedError


class HFInference(BaseInference):
    """Inference class for HuggingFace Transformers."""

    tokenizer: PreTrainedTokenizerFast

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        dtype = self.config.torch_dtype
        if dtype == "auto":
            dtype = torch.bfloat16
        else:
            dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path, torch_dtype=dtype, device_map="auto"
        ).eval()
        self.max_position_embeddings = self.model.config.max_position_embeddings

    def generate(self, prompt: List[Prompt]) -> List[Response]:
        response_vec = processes_input_to_tensor(
            self.tokenizer, prompt, self.max_position_embeddings
        )
        effective_batch = response_vec.get_effective_batch()
        if effective_batch is None:
            return response_vec.get_result()

        input_ids = effective_batch["input_ids"]
        attention_mask = effective_batch["attention_mask"]
        length_vec: List[int] = attention_mask.sum(axis=1).tolist()
        outputs = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_position_embeddings,
        )
        effective_result: List[Response] = []
        for output, length in zip(outputs, length_vec):
            output_tokens = output[length:]
            response = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            output_tokens = output_tokens.tolist()
            resp = Response(
                response=response,
                error=None,
                tokens=output_tokens,
                truncated=output_tokens[-1] != self.tokenizer.eos_token_id,
            )
            effective_result.append(resp)
        response_vec.set_effective_result(effective_result)
        return response_vec.get_result()


class VLLMInference(BaseInference):
    """Inference class for VLLM."""

    def __init__(self, config: ModelConfig):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Could not import vllm.")
        self.LLM = LLM
        self.SamplingParams = SamplingParams
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        self.model = self.LLM(
            model=self.config.model_name_or_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            enforce_eager=True,
            dtype=self.config.torch_dtype,
        )

    def generate(self, prompt: List[Prompt]) -> List[Response]:
        response_vec = processes_input_to_tensor(
            self.tokenizer, prompt, self.model.llm_engine.model_config.max_model_len
        )
        effective_batch = response_vec.get_effective_batch()
        if effective_batch is None:
            return response_vec.get_result()

        sampling_params = self.SamplingParams(
            temperature=self.config.temperature if self.config.do_sample else 0.0,
            top_p=self.config.top_p,
            max_tokens=None,
        )
        prompt_token_ids: List[List[int]] = []
        input_ids, attention_mask = (
            effective_batch["input_ids"],
            effective_batch["attention_mask"],
        )
        for input_id, attn_mask in zip(input_ids, attention_mask):  # type: ignore
            size = attn_mask.sum().item()
            input_id: List[int] = input_id[:size].tolist()  # type: ignore
            prompt_token_ids.append(input_id)
        outputs = self.model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)  # type: ignore
        effective_result: List[Response] = [
            Response(
                response=i.outputs[0].text,
                error=None,
                tokens=list(i.outputs[0].token_ids),
                truncated=i.outputs[0].finish_reason != "stop",
            )
            for i in outputs
        ]
        response_vec.set_effective_result(effective_result)
        return response_vec.get_result()


class OpenAIInference(BaseInference):
    """
    Inference class for OpenAI API.

    you may spawn an OpenAI campatible server to use this class.
    ```bash
    python3 -m vllm.entrypoints.openai.api_server --model MODEL_PATH  --api-key token-abc --dtype auto --served-model-name model --enforce-eager
    ```
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        self.client = openai.OpenAI(
            api_key=self.config.api_key, base_url=self.config.base_url
        )

    def generate_one(self, prompt: Prompt) -> Response:
        try:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt  # Assume it's already a list of messages

            response = self.client.chat.completions.create(
                model=self.config.model_name_or_path,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            return Response(response=response.choices[0].message.content)
        except Exception as e:
            return Response(response=None, error=str(e))

    def generate(self, prompt: List[Prompt]) -> List[Response]:  # type: ignore
        with ThreadPool(self.config.tensor_parallel_size) as pool:
            return list(pool.map(self.generate_one, prompt))


def get_model(config: ModelConfig) -> BaseInference:
    """Factory function to get the correct model."""
    if config.framework == "hf":
        return HFInference(config)
    elif config.framework == "vllm":
        return VLLMInference(config)
    elif config.framework == "openai":
        return OpenAIInference(config)
    else:
        raise ValueError(f"Unsupported framework: {config.framework}")


def process_dataset(
    config: GenerateConfig,
) -> None:
    """Process the evaluation dataset and save results."""
    model = get_model(config.model)  # Use factory function to get model

    df = get_penguin_dataset().select_columns([INPUT_COL, KEY_COL]).to_pandas()
    total_batchs = int(math.ceil(len(df) // config.batch_size))
    df_list: List[pd.DataFrame] = np.array_split(df, total_batchs)  # type: ignore
    output_df_list = []
    for block_df in tqdm(df_list, desc="Processing batches"):
        input_list = block_df[INPUT_COL].tolist()
        response_list = model.generate(input_list)  # type: ignore
        output_df = pd.DataFrame([i.model_dump(exclude={"tokens"}) for i in response_list])  # type: ignore
        output_df[KEY_COL] = block_df[KEY_COL].tolist()
        output_df_list.append(output_df)
    output_df = pd.concat(output_df_list).reset_index(drop=True)
    output_df.to_json(config.output_file, orient="records", lines=True)


if __name__ == "__main__":

    def main(config_file: str):
        config = GenerateConfig.model_validate_json(open(config_file).read())
        process_dataset(config)

    fire.Fire(main)
