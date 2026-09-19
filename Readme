# Teaching an LLM to Reason with GRPO

A hands-on Generative AI project exploring how reinforcement learning can
teach an instruction-tuned language model to perform reliable step-by-step
letter counting.

## Overview

The project trains a Qwen 2.5 3B Instruct model to answer questions such as:

> How many of the letter "g" are there in the word "engage"?

The model is encouraged to explicitly spell the word, maintain a running
count, and return the final answer in a structured format.

## Techniques

- Qwen 2.5 3B Instruct
- LoRA / PEFT
- 4-bit quantization
- GRPO (Group Relative Policy Optimization)
- Reward shaping
- Chain-of-thought prompting
- Hugging Face Transformers
- Unsloth
- vLLM

## Reward Design

The training objective uses multiple reward signals:

1. Sequential numbering
2. Correct spelling
3. Running-count accuracy
4. Output formatting
5. Final-answer correctness

## Workflow

```text
Base Qwen Model
      ↓
Prompt Engineering
      ↓
Letter-counting Dataset
      ↓
Reward Functions
      ↓
GRPO + LoRA Training
      ↓
Before/After Evaluation
