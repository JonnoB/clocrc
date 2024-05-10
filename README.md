# CLOCR-C: Context Leveraging OCR Correction using Language Models

CLOCR-C (Clock-Er-Sea) is a post-OCR correction approach that utilizes the infilling and context-adaptive abilities of transformer-based language models (LMs) to improve OCR quality. This repository contains the academic work used to demonstrate the concept.

## Key Notebooks

This work is arranged by notebooks. The key notebooks are numbered as follows:

1. `01_prompt_testing.ipynb`: Testing the range of sub-prompt combinations on the three datasets across all models using the dev set.
2. `02_llm_comparison.ipynb`: Using the best performing prompt run tests across the test set.
3. `03_downstream_ner.ipynb`: Measure how the corrected texts have improved NER.
4. `04_sociocultural_context.ipynb`: Explore the importance of the socio-cultural context of the prompt and the Task Induced In Context Learning.

## Dataset

This study uses three datasets: the 19th Century Serials Edition (NCSE) and two datasets from the Overproof collection. The NCSE dataset, consisting of 91 transcribed articles with a total of 40 thousand words, is available in a separate data repository at [link to repo].

## Requirements

To run the notebooks, you will need the following:

- Python 3.x
- Jupyter Notebook
- Required Python packages are found in the requirements.txt file

## Models

The experiments in this study were conducted using seven language models:

- GPT-4
- GPT-3.5
- Llama 3
- Gemma
- Mixtral 8x7b
- Claude 3 (Opus)
- Claude 3 (Haiku)

API key's for the models are required to reproduce this work

## License

This project is licensed under the [License Name]. See the `LICENSE` file for more information.

## Contact

For any questions or inquiries, please contact [Your Name] at [your-email@example.com].

## Abstract

The digitisation of historical print media archives is crucial for increasing accessibility to contemporary records. However, the process of Optical Character Recognition (OCR) used to convert physical records to digital text is prone to errors, particularly in the case of newspapers and periodicals due to their complex layouts. This paper introduces Context Leveraging OCR Correction (CLOCR-C), which utilises the infilling and context-adaptive abilities of transformer-based language models (LMs) to improve OCR quality. The study aims to determine if LMs can perform post-OCR correction, improve downstream NLP tasks, and the value of providing the socio-cultural context as part of the correction process. Experiments were conducted using seven LMs, including GPT-4, GPT-3.5, Llama 3, Gemma, Mixtral 8x7b, and Claude 3 (Opus and Haiku), on three datasets: the 19th Century Serials Edition and two datasets from the Overproof collection. The results demonstrate that some LMs can significantly reduce error rates, with the top-performing model achieving over a 60\% reduction in character error rate on the NCSE dataset. Furthermore, the study shows that providing socio-cultural context in the prompts improves performance, while misleading prompts lower performance. The OCR improvements extend to downstream tasks, such as Named Entity Recognition, with increased Cosine Named Entity Similarity and F1 scores. The findings suggest that CLOCR-C is a promising approach for enhancing the quality of existing digital archives by leveraging the socio-cultural information embedded in the LMs and the text requiring correction.

## Citing

[Citation information]