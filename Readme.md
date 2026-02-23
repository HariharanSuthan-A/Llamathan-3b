## Model Description

Llamathan-3B is a compact conversational language model designed for chat-based and interactive applications. It is built on the LLaMA 2 decoder-only transformer architecture and contains approximately 3 billion parameters, enabling low-latency inference while maintaining strong conversational coherence.

The model is optimized for:

Instruction-following dialogue

Context-aware conversational flows

Code-mixed Tamil (Tanglish) interactions

Educational and technical Q&A use cases

---

![Model Banner](assets/image%200.jpeg)
---

# 🧩 GGUF Version – llamathanQ4_K_M

## 📦 Quantized Model Details

**Model Name (GGUF):** `llamathanQ4_K_M.gguf`  
**Base Parameters:** 3 Billion (3B)  
**Architecture:** LLaMA 2 (Decoder-only Transformer)  
**Quantization Type:** Q4_K_M (4-bit K-Quant Medium)

---

## 🔢 Model Size

| Item | Value |
|------|--------|
| Parameter Count | ~3B |
| Quantization | 4-bit (Q4_K_M) |
| Approx File Size | ~1.8 – 2.2 GB |
| Original FP16 Size | ~6.0 GB |

> Q4_K_M provides a good balance between quality and memory efficiency.

---

## 🧠 Context Length

| Setting | Value |
|----------|--------|
| Default Context Length | 2048 tokens |
| Max Supported (if extended) | 4096* (depends on rope scaling & runtime) |

⚠️ Recommended: Use 2048 for stable performance.

---

## 💾 VRAM Requirements

### Minimum GPU VRAM (Recommended)

| Context | Required VRAM |
|----------|---------------|
| 2048 tokens | ~3 – 4 GB |
| 4096 tokens | ~4 – 5 GB |

### CPU Mode
- Can run fully on CPU (slower inference)
- Requires ~4–6 GB RAM

---

## 🖥 Running in LLM Studio

Compatible with:
- LM Studio
- llama.cpp based runtimes
- Ollama (if converted/packaged)
- Text Generation UIs supporting GGUF

### Steps for LM Studio

1. Open LM Studio
2. Go to **Models → Import Model**
3. Select `llamathanQ4_K_M.gguf`
4. Set:
   - Context Length: 2048
   - GPU Layers: Auto (or max if VRAM allows)
5. Start Chat

---

## ⚙️ Recommended Inference Settings

| Setting | Recommended Value |
|----------|-------------------|
| Temperature | 0.7 |
| Top_p | 0.9 |
| Top_k | 40 |
| Repeat Penalty | 1.1 |
| Max Tokens | 512–1024 |

---

## 🎯 Performance Profile

- Optimized for Tamil instruction-following
- Efficient on consumer GPUs (4GB+)
- Stable educational assistant usage
- Good Tanglish handling


---
## 🤗 Hosted On

This model is publicly available on:

👉 https://huggingface.co/Hariharan05/Llamathan-3B

---

# 🧩 GGUF Version – llamathanQ4_K_M

## 📦 Quantized Model Details

**Model Name (GGUF):** `llamathanQ4_K_M.gguf`  
**Base Parameters:** 3 Billion (3B)  
**Architecture:** LLaMA 2 (Decoder-only Transformer)  
**Quantization Type:** Q4_K_M (4-bit K-Quant Medium)

---

## 🔢 Model Size

| Item | Value |
|------|--------|
| Parameter Count | ~3B |
| Quantization | 4-bit (Q4_K_M) |
| Approx File Size | ~1.8 – 2.2 GB |
| Original FP16 Size | ~6.0 GB |

> Q4_K_M provides a good balance between quality and memory efficiency.

---

## 🧠 Context Length

| Setting | Value |
|----------|--------|
| Default Context Length | 2048 tokens |

⚠️ Recommended: Use 2048 for stable performance.

---

## 💾 VRAM Requirements

### Minimum GPU VRAM (Recommended)

| Context | Required VRAM |
|----------|---------------|
| 2048 tokens | ~3 – 4 GB |
| 4096 tokens | ~4 – 5 GB |

### CPU Mode
- Can run fully on CPU (slower inference)
- Requires ~4–6 GB RAM

---

## 🖥 Running in LLM Studio

Compatible with:
- LM Studio
- llama.cpp based runtimes
- Ollama (if converted/packaged)
- Text Generation UIs supporting GGUF

### Steps for LM Studio

1. Open LM Studio
2. Go to **Models → Import Model**
3. Select `llamathanQ4_K_M.gguf`
4. Set:
   - Context Length: 2048
   - GPU Layers: Auto (or max if VRAM allows)
5. Start Chat

---

## ⚙️ Recommended Inference Settings

| Setting | Recommended Value |
|----------|-------------------|
| Temperature | 0.7 |
| Top_p | 0.9 |
| Top_k | 40 |
| Repeat Penalty | 1.1 |
| Max Tokens | 256-512 |

---

## 🎯 Performance Profile

- Optimized for Tamil instruction-following
- Efficient on consumer GPUs (4GB+)
- Stable educational assistant usage
- Good Tanglish handling




---
## 🎯 Model Purpose

Llamathan-3B is optimized for:

- ✅ Tamil instruction-following
- ✅ Code-mixed Tamil (Tanglish) explanations
- ✅ Technical concept explanations in simplified Tamil
- ✅ Educational Q&A style prompts
- ✅ SQL query explanation & generation
- ✅ Multi-step reasoning
- ✅ Chain-of-Thought style responses

---

## 🧠 Model Description

Llamathan-3B is a fine-tuned version of Llama 2 3B Instruct, adapted for Tamil-centric instructional tasks.

The model specializes in:
- AI/ML explanations in simplified Tamil
- Technical concept breakdowns
- Conversational educational guidance
- Structured reasoning responses

It was trained using a structured instruction dataset in the format:



```json
{
  "instruction": "Explanation of Mixture of Experts (MoE).",
  "input": "Mixtral models-la 'MoE' na enna logic?",
  "output": "Motha model-aiyum orey nerathula use pannaama, specific question-ku endha 'Expert' best-nu router choose pannum. Performance high aagum aana cost kammi."
}
```


![Model Banner](assets/image%201.jpeg)
## 🗂 Dataset Format

Each training sample includes:

instruction → Task definition

input → User query (Tamil / Tanglish / Technical)

output → Expected response

## Prompt Template Used During Training
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}

The model was trained to autoregressively predict only the Response portion.

## ⚙️ Training Details
```
Parameter	Value
Epochs: 3
Batch Size: 8
Learning Rate: 5e-5
Optimizer: AdamW
LR Scheduler: Cosine Decay
Max Sequence Length: 2048
Precision: bfloat16 / fp16
Gradient Accumulation: Enabled
Training Hardware: NVIDIA T4
```
## 🧪 Evaluation Strategy

Evaluation was conducted using:

  Manual qualitative assessment

  Instruction-following accuracy checks

  Tamil fluency & coherence validation

  Technical correctness review

## 🚀 Inference
```python 

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Hariharan05/Llamathan-3B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.float16,
                                             device_map="auto")

prompt = """### Instruction:
Explanation of Mixture of Experts (MoE).

### Input:
Mixtral models-la 'MoE' na enna logic?

### Output:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```
## 🎓 Suitable Use Cases

Tamil educational assistants

AI/ML concept explanation chatbot

SQL query tutor (Tamil explanation)

Code-mixed Tamil conversational agents

Technical interview preparation assistant

## ⚠️ Limitations

Small dataset size (3,202 samples)

May hallucinate in unseen domains

Terminology mixing between Tamil & English

Limited reasoning depth compared to larger models


---

## License : 
@Llama2
