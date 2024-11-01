# PenguinScrolls: A Comprehensive Benchmark for Long-Text Understanding

PenguinScrolls (企鹅卷轴) is a comprehensive benchmark designed to evaluate and enhance the long-text processing capabilities of large language models (LLMs).  It addresses the shortcomings of existing long-text benchmarks by providing diverse document types, granular task categories, multi-turn dialogue data, and multilingual support, closely mirroring real-world user needs.

- [PenguinScrolls: A Comprehensive Benchmark for Long-Text Understanding](#penguinscrolls-a-comprehensive-benchmark-for-long-text-understanding)
  - [Setup](#setup)
  - [Data](#data)
  - [Running Evaluation](#running-evaluation)
  - [Adding New Tasks](#adding-new-tasks)
  - [Adding New Models](#adding-new-models)
  - [Dataset Correlation Analysis](#dataset-correlation-analysis)
  - [Others](#others)
  - [Contacts](#contacts)
  - [Citation](#citation)


## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suxue/PenguinScrolls.git
   cd PenguinScrolls
2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Data

The dataset is structured in the following format:

```json
[
  {
    "doc_id": "doc_001",
    "doc_type": "financial_report",
    "doc_text": "Long text content...",
    "tasks": [
      {
        "task_id": "task_001",
        "task_type": "information_extraction",
        "task_description": "Extract key financial figures.",
        "ground_truth": ["...", "..."],
        "multi_turn_dialogue": [
          {"user": "What is the net profit?", "bot": "..."},
          {"user": "And the revenue?", "bot": "..."}
        ]
      },
      {
        "task_id": "task_002",
        "task_type": "summarization",
        "task_description": "Summarize the key findings of the report.",
        "ground_truth": "...",
        "multi_turn_dialogue": []
      }
    ]
  },
  ...
]
```


The dataset includes a variety of document types (e.g., financial reports, legal documents, academic papers) ranging from 1K to 128K characters in length. Task types include information extraction, summarization, content analysis, reasoning, etc., with varying levels of difficulty. Multi-turn dialogue data simulates real-world interactions. Both Chinese and English data are provided.

## Running Evaluation

To evaluate a model on a specific task:

```bash
python evaluate.py --model_name your_model_name --task_type information_extraction
```

Replace your_model_name with the name of your model. See evaluate.py for more options and details.

## Adding New Tasks

To add a new task, create a new JSON file following the data format described above. Ensure the task_type field reflects the new task category.

## Adding New Models

To add a new model, implement the model interface in models.py. You can then evaluate the new model using the evaluate.py script.

## Dataset Correlation Analysis

Scripts and tools for analyzing dataset characteristics and correlations between different tasks are provided in the analysis directory.

## Others

Detailed information about the dataset statistics, task definitions, and evaluation metrics can be found in the docs directory.

## Contacts
For any questions or issues, please contact andyfei@tencent.com.

## Citation

If you use PenguinScrolls in your research, please cite it as follows:


```
TODO
```

