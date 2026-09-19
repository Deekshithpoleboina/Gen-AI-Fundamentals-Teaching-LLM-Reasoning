# 🧠 Teaching an LLM to Reason — GRPO Letter Counting

This project demonstrates how a **large language model (LLM)** can be trained to use **step-by-step reasoning** to answer a deceptively simple question:

> **“How many X's are there in the word Y?”**

The project uses **Group Relative Policy Optimization (GRPO)** to teach an instruction-tuned model to break a word into individual letters, keep a running count of a requested letter, and return the final answer in a structured format.

The notebook focuses on **reward shaping**, **parameter-efficient fine-tuning with LoRA**, prompt engineering, and practical GRPO training with a compact language model.

---

## 🚀 Features

- **Step-by-Step Letter Reasoning**  
  Encourages the model to inspect a word letter by letter while maintaining a running count.

- **GRPO Reinforcement Learning**  
  Uses **Group Relative Policy Optimization (GRPO)** to optimize the model directly against task-specific rewards.

- **LoRA Fine-Tuning**  
  Uses **Low-Rank Adaptation (LoRA)** for parameter-efficient training rather than updating all model weights.

- **Structured Reward Shaping**  
  Combines several reward functions to reinforce intermediate reasoning behavior as well as final correctness.

- **Prompt Engineering**  
  Uses clear instructions, Chain-of-Thought-style decomposition, and an example format to guide the model.

- **Before-and-After Evaluation**  
  Compares the original model with the LoRA-adapted model on the letter-counting task.

- **General Knowledge Check**  
  Tests whether the fine-tuned model still answers a basic factual question after task-specific training.

---

## 🛠️ Tech Stack

### Language Model
- Qwen **2.5-3B-Instruct**

### Training & Optimization
- GRPO (Group Relative Policy Optimization)
- LoRA / PEFT
- Unsloth
- TRL (Transformers Reinforcement Learning)
- vLLM

### Python Ecosystem
- Python
- PyTorch
- Hugging Face Datasets
- Pandas
- Matplotlib
- Regular Expressions (`re`)

### Execution Environment
- Jupyter Notebook
- Udacity / Vocareum GPU environment
- NVIDIA T4-class GPU configuration

---

## 📂 Project Structure

```text
Teaching-LLM-to-Reason/
├── gen_ai_fundamentals_project.ipynb
├── README.md
│
├── grpo_saved_lora/
│   └── saved LoRA adapter files
│
└── outputs/
    └── training outputs and logs
```

> `grpo_saved_lora/` and `outputs/` are generated during training and can be excluded from version control when the repository is intended to contain only the source notebook.

---

## 🧩 Project Workflow

The project is organized into the following stages:

```text
Prompt Engineering
       ↓
Letter-Counting Dataset
       ↓
Reward Function Design
       ↓
GRPO Configuration
       ↓
Quick Training Run
       ↓
Full Training Run
       ↓
LoRA Adapter Saving
       ↓
Old vs New Model Comparison
       ↓
General Knowledge Check
```

---

## 📚 Dataset

The notebook builds a **Hugging Face Dataset** from a curated collection of words with different lengths.

For each word, the dataset creates examples for:

- Letters that actually occur in the word
- Selected letters that do not occur in the word
- The expected number of occurrences for the requested letter

Each record contains fields conceptually equivalent to:

```text
words   → input word
letters → letter to count
counts  → expected count
```

A prompt is then constructed for every example using a system instruction and a user question such as:

```text
How many of the letter "g" are there in the word "engage"
```

---

## 🧠 How the AI Model Works

The model is trained to produce a response using a structured reasoning format:

```text
<reasoning>
1. e - 0 g's so far
2. n - 0 g's so far
3. g - 1 g's so far
4. a - 1 g's so far
5. g - 2 g's so far
6. e - 2 g's so far

The letter "g" appears 2 times in the word "engage".
</reasoning>

<answer>
2
</answer>
```

The core idea is to turn the task into a sequence of small, verifiable reasoning steps instead of relying only on the model's final prediction.

---

## 🎯 Reward Functions

The training objective uses multiple reward signals so that the model is encouraged to learn both the **process** and the **answer**.

| Reward Function | Purpose |
|---|---|
| `line_order_reward` | Rewards correctly ordered and numbered letters |
| `spelling_reward_func` | Rewards correct letter-by-letter spelling |
| `count_accuracy_reward` | Rewards accurate running counts |
| `format_reward_func` | Rewards the required `<reasoning>` / `<answer>` structure |
| `correct_answer_reward_func` | Rewards the final answer when it exactly matches the target count |

These rewards are combined and supplied to the GRPO trainer as the task objective.

---

## ⚙️ LoRA Configuration

The notebook uses LoRA for parameter-efficient adaptation.

Key project settings include:

```text
Max sequence length : 384
LoRA rank            : 64
4-bit loading        : Enabled
Fast inference       : Enabled
Gradient checkpoint  : Unsloth
```

The LoRA adapters target the major linear projections in both the attention and MLP blocks:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

This allows the model to learn the new letter-counting behavior while keeping the base model largely unchanged.

---

## ⚡ GRPO Training

The notebook uses a two-stage training workflow.

### 1️⃣ Quick Training Run

A short run is performed first to verify that:

- The dataset is correctly formatted
- The reward functions behave as intended
- The trainer can generate and score completions
- The training configuration is compatible with the available GPU

The reference notebook uses:

```text
max_steps = 5
```

### 2️⃣ Full Training Run

After validating the setup, the notebook performs a longer GRPO experiment.

The reference configuration uses:

```text
learning_rate = 1e-5
beta = 0.0001
per_device_train_batch_size = 16
num_generations = 4
gradient_accumulation_steps = 1
optim = adamw_8bit
lr_scheduler_type = cosine
num_train_epochs = 1
max_steps = 100
use_vllm = True
```

Training metrics include the overall reward and reward components, allowing the progression of the learned behavior to be inspected.

> Training time and reward values can vary depending on the GPU environment, software versions, random sampling, and the exact reward configuration.

---

## ⚙️ Installation & Local / Notebook Setup

### 1️⃣ Prerequisites

For a practical reproduction of the project, the environment should provide:

- Python
- NVIDIA GPU with sufficient VRAM
- PyTorch with CUDA support
- Unsloth
- vLLM
- TRL
- Hugging Face Datasets
- Pandas
- Matplotlib

---

## 2️⃣ Udacity / Vocareum Setup

The project notebook is designed to run in the **Udacity / Vocareum Jupyter environment**.

Select the appropriate Jupyter kernel before execution:

```text
Select Kernel
    ↓
Jupyter Kernel
    ↓
Python (venv2)
```

Then run the notebook cells from top to bottom.

The environment is expected to provide the GPU and the main LLM tooling required by the project.

---

## 3️⃣ Colab Setup

For a separate Google Colab environment, the notebook may require installation of the project dependencies before loading the model.

```bash
pip install -q unsloth vllm
```

A compatible GPU runtime should be selected before running the model and training cells.

---

## 📖 How to Run the Project

1. Open the Jupyter notebook.
2. Select the correct Python environment.
3. Verify GPU availability.
4. Load the Qwen 2.5 3B Instruct model with 4-bit quantization.
5. Configure the LoRA adapters.
6. Test prompt engineering on the letter-counting task.
7. Build the letter-counting dataset.
8. Implement and test the reward functions.
9. Run the short GRPO experiment.
10. Run the longer training experiment.
11. Save the LoRA adapter.
12. Compare the original and adapted models.
13. Run the general-knowledge check.

---

## 🔍 Example Task

### Input

```text
How many of the letter "o" are there in the word "room"?
```

### Expected reasoning pattern

```text
<reasoning>
1. r - 0 o's so far
2. o - 1 o's so far
3. o - 2 o's so far
4. m - 2 o's so far

The letter "o" appears 2 times in the word "room".
</reasoning>
```

### Expected final answer

```text
<answer>
2
</answer>
```

The important part is that the model learns to produce an explicit sequence of intermediate steps before giving the final count.

---

## 📊 Evaluation

The notebook evaluates the adapted model in two ways.

### Letter-Counting Evaluation

The trained model is compared with the original model on a dataset prompt to inspect whether the adapted model follows the requested reasoning and counting format more reliably.

### General Knowledge Evaluation

A separate factual prompt is used to check whether the model still retains basic knowledge after task-specific GRPO training.

The notebook specifically uses a well-known capital-city question for this check.

> Results may vary with training duration, hardware, sampling parameters, reward weights, and model behavior.

---

## 💾 Saving the LoRA Adapter

After training, the notebook saves the learned adapter separately from the base model:

```python
model.save_lora("grpo_saved_lora")
```

This makes it possible to evaluate the learned behavior without saving a complete copy of the base language model.

---

## 🎯 Learning Outcomes

- Understanding **GRPO-based reinforcement learning for LLMs**
- Designing **task-specific reward functions**
- Applying **reward shaping**
- Using **LoRA for parameter-efficient fine-tuning**
- Working with **Hugging Face Datasets**
- Using **Unsloth** for efficient model adaptation
- Using **vLLM** for accelerated generation
- Applying **prompt engineering and reasoning-oriented prompting**
- Designing small experiments before longer training runs
- Evaluating changes in model behavior after fine-tuning

---

## ⚠️ Important Notes

This repository contains an **educational LLM fine-tuning project**.

Training results are not guaranteed to match the examples shown in the notebook because reinforcement-learning experiments can vary with:

- GPU hardware
- Package and CUDA versions
- Random generation
- Reward definitions
- Training duration
- Sampling parameters

Generated model adapters and training outputs can also consume significant storage and usually do not need to be committed to GitHub.

---

## 🤝 Contributing

This project is intended for learning, experimentation, and improvement.

Areas that can be explored include:

- New reward formulations
- Alternative prompting strategies
- Different LoRA ranks
- Additional word samples
- Longer or shorter GRPO experiments
- Alternative evaluation tasks
- More robust response parsing

---

## 📄 License

No separate license file is specified for this project.

Please add a license to the repository before distributing the project as reusable open-source software.
