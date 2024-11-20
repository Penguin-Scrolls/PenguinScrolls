![](1.gif)

# PenguinScrolls: A User-Aligned Fine-Grained Benchmark for Long-Context Language Model Evaluation

PenguinScrolls (***企鹅卷轴***) is a comprehensive benchmark designed to evaluate and enhance the long-text processing capabilities of large language models (LLMs).

Current benchmarks for evaluating long-context language models often rely on synthetic tasks that fail to  adequately reflect real user needs, leading to a weak correlation between benchmark scores and actual user perceptions of model performance. To bridge this gap,  we conducted an in-depth investigation into the requirements of user groups that rely on long-text processing, gaining a thorough understanding of their demands. 
Building on these insights, we established a multi-level task classification framework oriented toward real user needs. Centered around this classification framework, we created PenguinScrolls, a comprehensive long-text dataset that encompasses a broad spectrum of document lengths, types, and interaction modes, including both single-turn and multi-turn exchanges.

Overall, the PenguinScrolls dataset encompasses four major categories of tasks—Information Extraction (568 items), Information Localization (278 items), Qualitative Analysis (324 items), and Numerical Reasoning (330 items)—amounting to a total of 1,500 single-turn data instances. 



## Key Characteristics

* **Fine-grained Task Types**: Features multi-level tasks of varying difficulty, constructing a comprehensive task classification system rooted in long-context processing abilities.
* **Multi-turn Dialogue Data**: Incorporates human-simulated questioning to create authentic long-context multi-turn dialogue scenarios.
* **Document Diversity**: Includes a wide range of natural long-form texts, including books, financial reports, legal documents, and academic papers, with contexts extending up to 128K tokens.

## News
[2024-11-20] The multi-turn instances are on the way !

[2024-11-20] A detailed paper introducing the PenguinScrolls dataset is being diligently prepared and will be released within the next two to three weeks. please feel free to contact me at penguinscrolls@tencent.com.

## Leaderboard
Here is the average scores (%) on the four major categories including 1 commercial LLMs an 4 open-source LLMs.


#### English
| Model Name       |  Avg  | Information Extraction | Information Localization | Qualitative Analysis | Numerical Reasoning |
| ---------------- | :---: | :--------------------: | :----------------------: | :------------------: | :-----------------: |
| GPT-4o           |  **82.73**   |           92.78           |            76.97            |          **85.20**          |         **68.79**          |
| Llama-3.1-70B    |  67.70   |           83.62           |            63.67            |          64.24          |         45.75          |
| Qwen2.5-70B |  82.58   |           **92.95**           |            81.65            |          84.16          |         64.54          |
| DeepSeek-V2.5-236B    |  74.76   |           82.39           |            74.46            |          76.87          |         60.60          |
| Hunyuan-Large      |  82.28   |           91.37           |            **84.89**            |          85.10          |         62.72          |





### Data

The dataset is located at huggingface datasethub: [TODO]

The dataset includes a variety of document types (e.g., books,financial reports, legal documents, academic papers) ranging from 1K to 128K characters in length. Task types include information extraction, summarization, content analysis, reasoning, etc., with varying levels of difficulty. Multi-turn dialogue data simulates real-world interactions. Both Chinese and English data are provided.

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
For any questions or issues, please contact penguinscrolls@tencent.com.

## Citation

If you use PenguinScrolls in your research, please cite it as follows:


```
TODO
```


