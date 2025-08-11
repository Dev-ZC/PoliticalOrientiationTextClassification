# PoliticalOrientiationTextClassification

# Model Tuning

We fine-tuned the **Mistral-7B-Instruct-v0.2** model using **Hugging Face Transformers** and the **PEFT** library with **LoRA (Low-Rank Adaptation)** to classify political statements into one of three categories:

* **Left-Leaning**
* **Right-Leaning**
* **Neutral**

The input data consisted of transcriptions of social media videos (generated via **Whisper**) labeled with political-leaning annotations.

---

## Data Preparation

* **Dataset Formatting**: Used instructional prompts with `[INST] ... [/INST]` tags.
  Example:

  ```
  Classify the following political statement as 'Left-Leaning', 'Right-Leaning', or 'Neutral': [statement]
  ```
* **Tokenization**:

  * Used the model’s native Byte-Pair Encoding (BPE) tokenizer via **SentencePiece** (accessed through `AutoTokenizer`).
  * Applied **padding** and **truncation** for consistent sequence lengths across batches.

---

## Fine-Tuning Approach

* **Quantization**:

  * **8-bit quantization** during training for memory efficiency.
  * **Mixed-precision training (FP16)** to reduce resource usage.

* **LoRA Application**:

  * Adapted a subset of model weights targeting the **causal language modeling head**.
  * Majority of base parameters were frozen to minimize training cost and memory use.

* **Training Pipeline**:

  * Used **Hugging Face’s Trainer API** with custom training arguments.
  * Evaluation performed after each epoch.

* **Post-Training Optimization**:

  * Quantized the model to **4-bit** using **BitsAndBytes** for efficient inference.
  * Deployed via a Hugging Face `pipeline`.

---

## Prompt Engineering Experiments

We evaluated two structured prompt types:

1. **Baseline Prompt** – concise, modeled after Gemini framework.
2. **Example-Augmented Prompt** – included representative labeled excerpts to guide classification.

### Findings

* The **baseline prompt** consistently outperformed the example-augmented variant.
* **Hypothesis**: Minimal prompting encourages the model to leverage internal, fine-tuned representations rather than overfitting to specific lexical patterns.
* Verbose prompts with many examples can reduce flexibility and generalization.

---

## Post-Processing & Evaluation

* **Output Processing**:

  * Extracted standardized, single-word labels.
  * Cleaned results to handle null outputs and normalize spelling inconsistencies.

* **Manual Evaluation**:

  * Randomly sampled **200 instances** and manually annotated for precision assessment.

* **Large-Scale Inference**:

  * Dataset split into 5 equal segments.
  * Processed in parallel on **five NVIDIA A100 GPUs** for maximum efficiency.

---

## Key Takeaways

* LoRA with quantization significantly reduced resource requirements.
* Concise, well-structured prompts worked better than verbose example-filled prompts.
* Parallel GPU processing enabled efficient large-scale classification.
