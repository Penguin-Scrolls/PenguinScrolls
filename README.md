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
| Model Name       |  Avg  | Information Extraction | Information Localization | Qualitative Analysis | Numerical Reasoning |
| ---------------- | :---: | :--------------------: | :----------------------: | :------------------: | :-----------------: |
| GPT-4o           |  **82.73**   |           92.78           |            76.97            |          **85.20**          |         **68.79**          |
| Llama-3.1-70B    |  67.70   |           83.62           |            63.67            |          64.24          |         45.75          |
| Qwen2.5-70B |  82.58   |           **92.95**           |            81.65            |          84.16          |         64.54          |
| DeepSeek-V2.5-236B    |  74.76   |           82.39           |            74.46            |          76.87          |         60.60          |
| Hunyuan-Large      |  82.28   |           91.37           |            **84.89**            |          85.10          |         62.72          |


## Invitation to Collaborate on Enhancing the PenguinScrolls Long-Text Evaluation Dataset

In developing PenguinScrolls, we have made every effort to encompass the needs of real-world users. However, due to limitations in resources and other factors, there is still significant room for improvement in PenguinScrolls.
We sincerely invite organizations and individuals from all sectors with long-text requirements to collaborate in building this evaluation dataset. By pooling our collective wisdom, we aim to create a high-quality long-text dataset that will support and guide the advancement of long-context models.
You can contribute your long-text requirements in two ways:

* **Demand Registration**: Submit your specific long-text needs on [Demand Registration Page](https://huggingface.co/spaces/long-context/https://huggingface.co/spaces/long-context/数据集登记). We will refine your requirements, construct the dataset, and perform manual annotations based on your input.
* **Dataset Registration**: Upload specific long-text evaluation samples on [Dataset Registration Page](https://huggingface.co/spaces/long-context/数据集登记). We will manually annotate, classify, and integrate them into PenginusCross.




## Evaluate Your LLMs on **PenguinScrolls**

- [PenguinScrolls: A Comprehensive Benchmark for Long-Text Understanding](#penguinscrolls-a-comprehensive-benchmark-for-long-text-understanding)
  - [Key Characteristics](#key-characteristics)
  - [News](#news)
  - [Leaderboard](#leaderboard)
      - [English](#english)
  - [Invitation to Collaborate on Enhancing the PenguinScrolls Long-Text Evaluation Dataset](#invitation-to-collaborate-on-enhancing-the-penguinscrolls-long-text-evaluation-dataset)
  - [Evaluate Your LLMs on **PenguinScrolls**](#evaluate-your-llms-on-penguinscrolls)
    - [Setup](#setup)


### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suxue/PenguinScrolls.git
   cd PenguinScrolls

### Data

The dataset is located at huggingface datasethub: [TODO]

The dataset includes a variety of document types (e.g., financial reports, legal documents, academic papers) ranging from 1K to 128K characters in length. Task types include information extraction, summarization, content analysis, reasoning, etc., with varying levels of difficulty. Multi-turn dialogue data simulates real-world interactions. Both Chinese and English data are provided.

### Running Evaluation

see [USAGE](./USAGE.md)


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


