![](1.gif)
<p align="center">
    🤗 <a href="https://huggingface.co/datasets/" target="_blank">HF Repo</a> 
</p>

# PenguinScrolls: A User-Aligned Fine-Grained Benchmark for Long-Context Language Model Evaluation

PenguinScrolls (***企鹅卷轴***) is a comprehensive benchmark designed to evaluate and enhance the long-text processing capabilities of large language models (LLMs).

Current benchmarks for evaluating long-context language models often rely on synthetic tasks that fail to  adequately reflect real user needs, leading to a weak correlation between benchmark scores and actual user perceptions of model performance. To bridge this gap,  we conducted an in-depth investigation into the requirements of user groups that rely on long-text processing, gaining a thorough understanding of their demands. 
Building on these insights, we established a multi-level task classification framework oriented toward real user needs. Centered around this classification framework, we created PenguinScrolls, a comprehensive long-text dataset that encompasses a broad spectrum of document lengths, types, and interaction modes, including both single-turn and multi-turn exchanges.

Overall, the PenguinScrolls dataset encompasses four top-level categories of tasks—Information Extraction (420 items), Information Localization (278 items), Qualitative Analysis (305 items), and Numerical Reasoning (313 items)—amounting to a total of 1,316 single-turn data instances. 



## Key Characteristics

* **Fine-grained Task Types**: Features multi-level tasks of varying difficulty, constructing a comprehensive task classification system rooted in long-context processing abilities;
* **Multi-turn Dialogue Data**: Incorporates human-simulated questioning to create authentic long-context multi-turn dialogue scenarios;
* **Document Diversity**: Includes a wide range of natural long-form texts, including books, financial reports, legal documents, and academic papers, with contexts extending up to 128K tokens.

## News
**[2024-11-20]** The multi-turn instances are on the way !

**[2024-11-20]** A detailed paper introducing the PenguinScrolls dataset is being diligently prepared and will be released within the next two to three weeks. please feel free to contact me at 3965243706@qq.com.

## Leaderboard
Here is the average scores (%) on the four major categories including 1 commercial LLMs an 4 open-source LLMs.


#### Overall Results on  PenguinScrolls
| Model Name       |  Micro Avg  | Information Extraction(420 items) | Information Localization(278 items) | Qualitative Analysis(305 items) | Numerical Reasoning(313 items) |
| ---------------- | :---: | :--------------------: | :----------------------: | :------------------: | :-----------------: |
| GPT-4o           |  **82.72**   |           91.19           |            77.34            |          **86.43**          |         **72.52**          |
| Llama-3.1-70B    |  66.45   |           79.05           |            66.19            |          69.00          |         47.28          |
| Qwen2.5-70B |  81.83   |           90.95           |            79.86            |          84.89          |         68.37          |
| DeepSeek-V2.5-236B    |  73.32   |           77.86           |            73.38            |          77.67          |         62.94          |
| Hunyuan-Large      |  82.56   |           **91.43**           |            **86.33**            |          84.44          |         65.50          |

#### Results on Different Length Sets


| Model Name       | 0K-8K | 8K-32K | 32K-64K | 64K-128K |
| ---------------- | :--------------------: | :----------------------: | :------------------: | :-----------------: |
| GPT-4o           |           92.49           |            80.73            |          79.53          |         **73.62**          |
| Llama-3.1-70B    |           82.51           |            67.26            |          54.01          |         49.54          |
| Qwen2.5-70B |           **92.58**           |            80.62            |          75.39          |         72.68          |
| DeepSeek-V2.5-236B    |           90.11           |            75.63            |          56.70          |         56.07          |
| Hunyuan-Large      |           90.87           |            **81.60**            |          **80.52**          |         72.36          |



### Data

The dataset is located at huggingface datasethub: [TODO]

The dataset includes a variety of document types (e.g., books,financial reports, legal documents, academic papers) ranging from 1K to 128K characters in length. Task types include information extraction, summarization, content analysis, reasoning, etc., with varying levels of difficulty. Multi-turn dialogue data simulates real-world interactions. 

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

## Caution
This open-source dataset is intended solely for evaluation and research purposes and must not be used otherwise. It may contain third-party content, including third-party notices and copyright information, the terms of which must be observed and followed. While we make every effort to ensure the content, we cannot guarantee it is free from infringements. Please exercise caution and review the  licensing terms carefully before using it. If you encounter any issues or have concerns regarding this dataset, please contact us at 3965243706@qq.com. We will respond and address the issue timely in accordance with applicable legal standards.


## Citation

If you use PenguinScrolls in your research, please cite it as follows:


```
TODO
```


