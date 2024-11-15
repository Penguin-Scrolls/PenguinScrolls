![](1.gif)

# PenguinScrolls: A Comprehensive Benchmark for Long-Text Understanding

PenguinScrolls (***企鹅卷轴***) is a comprehensive benchmark designed to evaluate and enhance the long-text processing capabilities of large language models (LLMs).

Existing long-text evaluation datasets suffer from several limitations: the scarcity of authentic long-document content, limited diversity in document types, insufficient coverage of real user needs, scarcity of Chinese-language data, and a deficiency of multi-turn dialogue data. These shortcomings weaken the correlation between evaluation scores on these test sets and the actual user experience. To address these challenges, we conducted an in-depth investigation into the needs of user groups requiring long-text processing, thoroughly understanding their demands. Based on the findings, We established a multi-level task classification framework oriented toward real user needs. Centered around this classification framework, we meticulously constructed a comprehensive long-text dataset named PenguinScrolls, covering various length ranges, document types, single-turn and multi-turn interactions, and multiple question types.

Overall, the PenguinScrolls dataset encompasses four major categories of tasks—Information Extraction, Information Localization, Qualitative Analysis, and Numerical Reasoning—amounting to a total of 2,880 single-turn data instances. Additionally, we have constructed 800 long-text multi-turn dialogue instances, which, to our knowledge, constitute the first dataset of its kind.



## Key Characteristics

* **Fine-grained Task Types**: Features multi-level tasks of varying difficulty, constructing a comprehensive task classification system rooted in long-context processing abilities.
* **Multi-turn Dialogue Data**: Incorporates human-simulated questioning to create authentic long-context multi-turn dialogue scenarios.
* **Document Diversity**: Includes a wide range of natural long-form texts, including financial reports, legal documents, and academic papers, with contexts extending up to 128K tokens.
* **Multilingual Support**: Provides data in both Chinese and English to meet the needs of multilingual applications.

## News
[2024-11-15] A detailed paper introducing the PenguinScrolls dataset is being diligently prepared and will be released within the next two to three weeks., please feel free to contact me at andyfei@tencent.com.

## Leaderboard
Here is the average scores (%) on the four major categories including XX commercial LLMs an YY open-source LLMs.


#### English
|           Model Name        | Avg  | Information Extraction | Information Localization | Qualitative Analysis | Numerical Reasoning |
| ----------------- | :--: | :-----------: | :----------: | :-----------: | :---------------: | 
| GPT-4o | XX | XX | XX | XX | XX |
| Hunyuan-Large | XX | XX | XX | XX | XX |
| Claude3.5-Sonnet | XX | XX | XX | XX | XX |
| Llama-3.1-70B | XX | XX | XX | XX | XX |
| GPT-4o-mini | XX | XX | XX | XX | XX |
| gemini-1.5-pro | XX | XX | XX | XX | XX | 
| qwen2.5-70B | XX | XX | XX | XX | XX | 


## Invitation to Collaborate on Enhancing the PenguinScrolls Long-Text Evaluation Dataset

In developing PenguinScrolls, we have made every effort to encompass the needs of real-world users. However, due to limitations in resources and other factors, there is still significant room for improvement in PenguinScrolls.
We sincerely invite organizations and individuals from all sectors with long-text requirements to collaborate in building this evaluation dataset. By pooling our collective wisdom, we aim to create a high-quality long-text dataset that will support and guide the advancement of long-context models.
You can contribute your long-text requirements in two ways:

* **Demand Registration**: Submit your specific long-text needs on [Demand Registration Page](https://huggingface.co/spaces/long-context/https://huggingface.co/spaces/long-context/数据集登记). We will refine your requirements, construct the dataset, and perform manual annotations based on your input.
* **Dataset Registration**: Upload specific long-text evaluation samples on [Dataset Registration Page](https://huggingface.co/spaces/long-context/数据集登记). We will manually annotate, classify, and integrate them into PenginusCross.




## Evaluate Your LLMs on **PenguinScrolls**

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


### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suxue/PenguinScrolls.git
   cd PenguinScrolls
2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### Data

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

### Running Evaluation

To evaluate a model on a specific task:

```bash
python evaluate.py --model_name your_model_name --task_type information_extraction
```

Replace your_model_name with the name of your model. See evaluate.py for more options and details.

### Adding New Tasks

To add a new task, create a new JSON file following the data format described above. Ensure the task_type field reflects the new task category.

### Adding New Models

To add a new model, implement the model interface in models.py. You can then evaluate the new model using the evaluate.py script.

### Dataset Correlation Analysis
****
Scripts and tools for analyzing dataset characteristics and correlations between different tasks are provided in the analysis directory.

### Others

Detailed information about the dataset statistics, task definitions, and evaluation metrics can be found in the docs directory.

## Contacts
For any questions or issues, please contact andyfei@tencent.com.

## Citation

If you use PenguinScrolls in your research, please cite it as follows:


```
TODO
```


