import os
from contextlib import contextmanager
from logging import ERROR
from multiprocessing.dummy import Pool
from typing import Any, Dict, Optional, Union

import fire
import pandas as pd
from datasets import load_dataset
from jinja2 import Template
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from tqdm.auto import tqdm

from .defs import ERROR_PREFIX

openai = OpenAI()

template_no_evidence = Template(
    r"""Task Overview:
You are tasked with evaluating user answers based on a given question, reference answer. Your goal is to assess the correctness of the user answer using a specific metric.

Evaluation Criteria:
1. Yes/No Questions: Verify if the user's answer aligns with the reference answer in terms of a "yes" or "no" response.
2. Short Answers/Directives: Ensure key details such as numbers, specific nouns/verbs, and dates match those in the reference answer.
3. Abstractive/Long Answers: The user's answer can differ in wording but must convey the same meaning and contain the same key information as the reference answer to be considered correct.

Evaluation Process:
1. Identify the type of question presented.
2. Apply the relevant criteria from the Evaluation Criteria.
3. Compare the user's answer against the reference answer accordingly.
4. Score the answer with a binary label 0 or 1, where 0 denotes wrong and 1 denotes correct.
NOTE that if the user answer is 0 or an empty string, it should get a 0 score.

Question: {{question}}
User Answer: {{sys_ans}}
Reference Answer: {{ref_ans}}

Evaluation Form (score ONLY):
- Correctness:"""
)


class EvalResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: Optional[Union[float, int]] = None  # 0到1之间的分数，便于取平均
    result: Optional[str] = None  # 原始结果


def ask(prompt: str) -> str:
    try:
        resp = openai.chat.completions.create(
            model=os.environ["PENGUIN_EVAL_MODEl"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            top_p=0.9,
        )
        ret = resp.choices[0].message.content  # type: ignore
        if ret is None:
            return f'{ERROR_PREFIX}: "None" returned'
        return ret
    except Exception as ex:
        return f"{ERROR_PREFIX}: {str(ex)}"


class Row(BaseModel):
    dataset: str
    split: str
    payload: Dict[str, Any]
    input_md5: str
    input: str
    output: str
    prompt: str
    token_count: int
    doc_start: int
    doc_end: int
    response: Optional[str] = None

    @property
    def doc(self) -> str:
        return self.input[self.doc_start : self.doc_end]


def get_penguin_dataset(limit: Optional[int] = None):
    dataset = load_dataset(os.environ["PENGUIN_SCROLLS"])
    x = dataset["test"]
    if limit is not None:
        x = x.select(range(limit))
    return map(lambda x: Row.model_validate(x), x)


def generate_response_from_file(dataset, result_filename: str):
    df = pd.read_json(result_filename, lines=True)[["input_md5", "output"]].set_index(
        "input_md5"
    )

    def mapper(row: Row) -> Row:
        key = row.input_md5
        try:
            response = df.at[key, "output"]
        except KeyError:
            response = None
        row.response = response
        return row

    x = map(mapper, dataset)
    x = filter(lambda x: x.response is not None, x)
    return x


class DocBenchResult(BaseModel):
    result: str
    score: Optional[int] = None


def eval_row(row: Row) -> EvalResult:
    prompt = template_no_evidence.render(
        question=row.prompt, sys_ans=row.response, ref_ans=row.output
    )
    eval_response = ask(prompt).strip()
    if eval_response.startswith(ERROR_PREFIX):
        return EvalResult(score=None, result=eval_response)
    last_line = eval_response.split("\n")[-1]

    if "1" in last_line and "0" not in last_line:
        score = 1
    elif "0" in last_line and "1" not in last_line:
        score = 0
    else:
        score = None

    eval_result = EvalResult(result=eval_response, score=score)
    return eval_result


@contextmanager
def get_mapper(concurrency: int):
    assert concurrency >= 1
    if concurrency == 1:
        yield map
    else:
        with Pool(concurrency) as pool:
            yield pool.imap


def main(
    input_filename: str,  # must have 'input_md5' and 'output' columns
    output_filename: str,
    concurrency: int = 1,
    limit: Optional[int] = None,
):
    x = get_penguin_dataset(limit=limit)
    x = list(generate_response_from_file(x, input_filename))
    with get_mapper(concurrency) as mapper:
        eval_result = list(tqdm(mapper(eval_row, x), total=len(x)))
    result_df = pd.DataFrame(
        {
            "input_md5": [i.input_md5 for i in x],
            "response": [i.response for i in x],
            "score": [i.score for i in eval_result],
            "result": [i.result for i in eval_result],
        }
    )
    result_df.to_json(output_filename, lines=True, orient="records", force_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
