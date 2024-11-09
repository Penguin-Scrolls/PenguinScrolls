import json
import hashlib
from typing import Optional, TypedDict, NewType, List, Union
import os
from dataclasses import dataclass
import torch
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, BatchEncoding



@dataclass
class ModelConfig:
    """Configuration for model settings."""
    model_name_or_path: str
    framework: str  # 'hf', 'vllm', or 'openai'
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    api_key: Optional[str] = None
    tensor_parallel_size: int = 1
    system_prompt: Optional[str] = None


class Message(TypedDict):
    role: str
    content: str

MessageList = NewType('MessageList', List[Message])
Prompt = Union[MessageList, str]

@dataclass
class Response:
    """Response class for model inference."""
    response: Optional[str]
    error: Optional[str]


def preprocess_input(tokenizer: PreTrainedTokenizerFast, prompt: Prompt):
    """Preprocess input for tokenization and formatting."""
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt  # Assume it's already a list of messages
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return formatted_prompt

def processes_input_to_tensor(tokenizer: PreTrainedTokenizerFast, prompt: Prompt) -> BatchEncoding:
    formatted_prompt = preprocess_input(tokenizer, prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    return inputs

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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: Prompt) -> Response:
        try:
            inputs = processes_input_to_tensor(self.tokenizer, prompt)
            # Rest of generate method remains same
            outputs = self.model.generate(
                input_ids=inputs["input_ids"].to(self.model.device),
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(inputs['input_ids']):]
            return Response(response=response, error=None)
        except Exception as e:
            return Response(response=None, error=str(e))


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
                max_tokens=self.config.max_tokens
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
                max_tokens=self.config.max_tokens,
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
\