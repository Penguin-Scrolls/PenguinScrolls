import hashlib
import json
import os
from dataclasses import dataclass
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

import openai
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerFast,
)

T = TypeVar('T')
R = TypeVar('R')



@dataclass
class ModelConfig:
    """Configuration for model settings."""
    model_name_or_path: str
    framework: str  # 'hf', 'vllm', or 'openai'
    temperature: float = 0.7
    top_p: float = 1.0
    api_key: Optional[str] = None
    tensor_parallel_size: int = 1
    system_prompt: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = torch.float16
    do_sample: bool = True
    max_new_tokens: int = 128


class Message(TypedDict):
    role: str
    content: str

MessageList = NewType('MessageList', List[Message])
Prompt = Union[MessageList, str]

@dataclass
class Response:
    """Response class for model inference."""
    response: Optional[str] = None
    error: Optional[str] = None
    tokens: Optional[List[int]] = None
    truncated: bool = False


def vectorize(func: Callable[[Any, T], R]) -> Callable[[Any, Union[T, List[T]]], Union[R, List[R]]]:
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
        
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return formatted_prompt


class ResponseVector:
    """Class for storing result vectors."""
    def __init__(self, batch: BatchEncoding, max_length: int, padding_side: str):
        self.batch = batch
        self.max_length = max_length
        self.batch_size = self.batch['input_ids'].shape[0]
        self.length_vec = batch['attention_mask'].sum(axis=1)
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
        input_ids = self.batch['input_ids'][self.mask, :] # type: ignore
        attention_mask = self.batch['attention_mask'][self.mask, :] # type: ignore
        longest_size = int(attention_mask.sum(axis=1).max())
        if self.padding_side == 'right':
            return {
             'input_ids': input_ids[:, :longest_size],
                'attention_mask': attention_mask[:, :longest_size]
            } # type: ignore
        else:
            return {
                'input_ids': input_ids[:, -longest_size:],
                'attention_mask': attention_mask[:, -longest_size:]
            } # type: ignore

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
    batch = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=False)
    return ResponseVector(batch, max_length, tokenizer.padding_side)

class BaseInference:
    """Base class for model inference."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def generate(self, prompt: Prompt) -> Response:
        raise NotImplementedError


class HFInference(BaseInference):
    """Inference class for HuggingFace Transformers."""

    tokenizer: PreTrainedTokenizerFast

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        dtype = self.config.torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=dtype,
            device_map="auto"
        ).eval()
        self.max_position_embeddings = self.model.config.max_position_embeddings

    def generate(self, prompt: List[Prompt]) -> List[Response]:
        response_vec = processes_input_to_tensor(self.tokenizer, prompt, self.max_position_embeddings)
        effective_batch = response_vec.get_effective_batch()
        if effective_batch is None:
            return response_vec.get_result()
        
        input_ids = effective_batch["input_ids"]
        attention_mask = effective_batch['attention_mask']
        length_vec: List[int] = attention_mask.sum(axis=1).tolist()
        outputs = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        effective_result: List[Response] = []
        for output, length in zip(outputs, length_vec):
            output_tokens = output[length:]
            response = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            output_tokens = output_tokens.tolist()
            resp =  Response(response=response, error=None, tokens=output_tokens, truncated=output_tokens[-1] != self.tokenizer.eos_token_id)
            effective_result.append(resp)
        response_vec.set_effective_result(effective_result)
        return response_vec.get_result()


class VLLMInference(BaseInference):
    """Inference class for VLLM."""

    def __init__(self, config: ModelConfig):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Could not import vllm or transformers.")
        self.LLM = LLM
        self.SamplingParams = SamplingParams
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        self.model = self.LLM(
            model=self.config.model_name_or_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
        )

    def generate(self, prompt: Prompt) -> Response:
        try:
            inputs = preprocess_input(self.tokenizer, prompt)
            sampling_params = self.SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_input_tokens
            )
            outputs = self.model.generate(inputs, sampling_params)
            response_text = outputs[0].outputs[0].text
            return Response(response=response_text, error=None)
        except Exception as e:
            return Response(response=None, error=str(e))


class OpenAIInference(BaseInference):
    """Inference class for OpenAI API."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        self.client = openai.OpenAI(api_key=self.config.api_key)

    def generate(self, prompt: Prompt) -> Response:
        try:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt  # Assume it's already a list of messages

            response = self.client.chat.completions.create(
                model=self.config.model_name_or_path,
                messages=messages,
                max_tokens=self.config.max_input_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            return Response(response=response.choices[0].message.content, error=None)
        except Exception as e:
            return Response(response=None, error=str(e))


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


def compute_md5(text: str) -> str:
    """Compute MD5 hash of input text."""
    return hashlib.md5(text.encode()).hexdigest()


def process_dataset(
    input_file: str,
    output_file: str,
    model_config: ModelConfig,
    batch_size: int = 1
) -> None:
    """Process the evaluation dataset and save results."""
    model = get_model(model_config) # Use factory function to get model
    
    # Create output file if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            pass

    # Track processed inputs to avoid duplicates
    processed_inputs = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_inputs.add(data['input_md5'])
                except:
                    continue

    # Process input file
    with open(input_file, 'r') as f_in, open(output_file, 'a') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                input_md5 = data['input_md5']
                
                # Skip if already processed
                if input_md5 in processed_inputs:
                    continue
                
                # Generate response
                response_obj = model.generate(data['input'])
                
                # Check for errors
                if response_obj.error:
                    print(f"Error generating response: {response_obj.error}")
                    continue
                
                output = response_obj.response
                
                # Save result
                result = {
                    'input_md5': input_md5,
                    'output': output
                }
                f_out.write(json.dumps(result) + '\n')
                processed_inputs.add(input_md5)
                
            except Exception as e:
                print(f"Error processing input: {e}")
                continue
