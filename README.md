# 🪶 Qareed (قريض)

> **Arabic Poetry Fine-Tuning & Generation powered by Qwen2.5**  
> نموذج لغوي متخصص في توليد الشعر العربي الفصيح الموزون بناءً على الأنماط والموضوعات المختلفة وعدد الأبيات المطلوبة.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qareed--Qwen2.5--1.5B--Poet--v1-yellow)](https://huggingface.co/Youssefx64/Qareed-Qwen2.5-1.5B-Poet-v1)
[![Base Model](https://img.shields.io/badge/Base_Model-Qwen2.5--1.5B--Instruct-purple)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Hugging%20Face%20%7C%20PEFT-orange)](https://github.com/huggingface/transformers)

---

## 📖 Overview | نبذة عن المشروع

**Qareed (قريض)** — named after the classical Arabic term for metered poetry (*القريض*) — is an instruction-tuned language model designed to compose Arabic poetry. 

Fine-tuned on top of **`Qwen/Qwen2.5-1.5B-Instruct`** using **QLoRA** (4-bit quantization + LoRA adapters), the model can generate between 1 to 4 poetic verses (*أبيات*) conditioned on specific classical and modern poetic themes (*أغراض الشعر*) or arbitrary free-form instructions.

### Supported Themes | أغراض الشعر المدعومة
- **الغزل** (Love / Romance)
- **الشوق** (Longing & Yearning)
- **الفراق** (Parting & Separation)
- **الحزن** (Melancholy & Sorrow)
- **الرثاء** (Elegies & Lamentation)
- **المدح** (Praise & Eulogy)
- **العتاب** (Admonition & Reproach)
- **الهجاء** (Satire & Invective)
- **الوطنية** (Patriotism & Homeland)
- **الدين والتقوى** (Devotion & Piety)
- **موضوع حر** (Custom open prompts)

---

## 📁 Repository Structure | هيكل المشروع

```plaintext
Qareed/
├── Qareed-Qwen2.5-1.5B-Poet-v1-training.ipynb  # Full training notebook (data prep & QLoRA SFT)
├── qareed_train.jsonl                          # SFT training dataset (instruction-response pairs)
├── qareed_val.jsonl                            # SFT validation dataset
├── space/                                      # Hugging Face Space & Gradio demo
│   ├── app.py                                  # Streaming Gradio web application
│   ├── requirements.txt                        # Space dependencies
│   ├── README.md                               # Space configuration metadata
│   └── Qareed-Poet-Demo.ipynb                  # Demo launch notebook
├── .gitignore                                  # Git ignore rules for ML/Python artifacts
├── LICENSE                                     # Apache 2.0 License
└── README.md                                   # Project documentation
```

---

## 🚀 Quickstart | التشغيل السريع

### 1. Direct Python Inference (`transformers`)

You can load and query the model directly via the Hugging Face `transformers` library:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

MODEL_ID = "Youssefx64/Qareed-Qwen2.5-1.5B-Poet-v1"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# System prompt used during fine-tuning
SYSTEM_PROMPT = (
    "أنت «قريض»، شاعر عربي متمكّن من اللغة والعروض والقافية. "
    "تكتب شعرًا عربيًا فصيحًا موزونًا ومقفّى، وتلتزم بعدد الأبيات المطلوب دون شرح أو تعليق."
)

prompt = "اكتب بيتين من الشعر في الشوق."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### 2. Running the Gradio Web Demo Locally

To launch the interactive streaming web interface:

```bash
# Clone the repository
git clone https://github.com/Youssefx64/Qareed.git
cd Qareed/space

# Install dependencies
pip install -r requirements.txt gradio

# Run the app
python app.py
```

Open `http://localhost:7860` in your browser.

---

## 🛠️ Dataset & Training | البيانات والتدريب

### Data Pipeline
- **Source**: Sampled and curated from [`arbml/Ashaar_dataset`](https://huggingface.co/datasets/arbml/Ashaar_dataset).
- **Preprocessing**:
  - Reconstructed verses into classical two-hemistich format (*صدر وعجز*) separated by standard delimiters (`...`).
  - Generated diverse Arabic instruction prompts requesting 1 to 4 verses across different poetic themes.
  - Train/Val split: Saved as JSONL (`qareed_train.jsonl` and `qareed_val.jsonl`).

Example training record:
```json
{
  "prompt": "انظم بيتين في العتاب.",
  "answer": "عتبت على دهري بأفعاله التي ... براني بها بري السهام من الهم\nليصرف عني فادحات نوائب ... أضاق بها ذرعي وأضنى بها جسمي",
  "n_baits": 2,
  "theme": "عتاب"
}
```

### Training Configuration (QLoRA)
The model was fine-tuned using Hugging Face's `TRL` (Transformer Reinforcement Learning) and `PEFT`:

| Parameter | Value |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Quantization** | 4-bit NormalFloat (NF4) via `bitsandbytes` |
| **LoRA Rank ($r$)** | `16` |
| **LoRA Alpha ($\alpha$)** | `32` |
| **LoRA Dropout** | `0.05` |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Optimizer** | `paged_adamw_8bit` |
| **Learning Rate** | `2e-4` (Cosine schedule, warmup=40 steps) |
| **Batch Size** | 8 per device (gradient accumulation steps: 2) |
| **Precision** | FP16 |
| **Hardware** | Single NVIDIA T4 GPU (Google Colab) |

---

## ⚠️ Limitations & Notes (v1) | القيود والملاحظات

- **Experimental (v1)**: While the model generates syntactically rich and stylistically authentic Arabic poetic verses, precise adherence to classical meters (*بحور الخليل الستة عشر*) and continuous monorhyme (*الروي والقافية*) is not strictly guaranteed in v1. Outputs should be reviewed and verified by a human reader.
- **Length Constraint**: Optimized for short poems (1–4 verses). Longer generations may diverge in meter consistency.

---

## 🗺️ Roadmap | الخطة المستقبلية

- [ ] **Meter Conditioning (تحديد البحر العروضي)**: Enable prompting with specific meters (e.g., *الطويل*, *البسيط*, *الكامل*, *الخفيف*).
- [ ] **Rhyme Enforcement (التحكم بالقافية)**: Constrain output generation to specific end-rhymes (روي).
- [ ] **Larger Scale Base Models**: Expand fine-tuning to larger architectures (e.g., Qwen2.5-7B/14B).
- [ ] **Tashkeel / Diacritization Support**: Improve verse vocalization for easier meter validation.

---

## 📜 License | الترخيص

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for complete details.

---

## 🙏 Acknowledgments & Credits

- [Qwen Team (Alibaba Cloud)](https://github.com/QwenLM/Qwen2.5) for the base `Qwen2.5-1.5B-Instruct` model.
- [ARBML Community](https://github.com/arbml) for open-sourcing the `Ashaar_dataset`.
- [Hugging Face](https://huggingface.co/) for the `transformers`, `peft`, and `trl` ecosystems.
