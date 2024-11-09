import json
import hashlib
from typing import Optional
import os
from dataclasses import dataclass
import torch
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vllm not found, VLLM inference will not be available")
    LLM = None
    SamplingParams = None

@dataclass
class ModelConfig:
    """Configuration for model settings."""
    model_name_or_path: str
    framework: str  # 'hf', 'vllm', or 'openai'
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    api_key: Optional[str] = None

class BaseInference:
    """Base class for model inference."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class HFInference(BaseInference):
    """Inference class for HuggingFace Transformers."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]


class VLLMInference(BaseInference):
    """Inference class for VLLM."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if LLM is None or SamplingParams is None:
            raise ImportError("Could not import vllm")
        self.model = LLM(
            model=self.config.model_name_or_path,
            tensor_parallel_size=1  # Adjust based on GPU availability
        )

    def generate(self, prompt: str) -> str:
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens
        )
        outputs = self.model.generate(prompt, sampling_params)
        return outputs[0].outputs[0].text


class OpenAIInference(BaseInference):
    """Inference class for OpenAI API."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        openai.api_key = self.config.api_key

    def generate(self, prompt: str) -> str:
        response = openai.Completion.create(
            model=self.config.model_name_or_path,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )
        return response.choices[0].text


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
                output = model.generate(data['input'])
                
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