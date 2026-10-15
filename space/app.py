import threading
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Youssefx64/Qareed-Qwen2.5-1.5B-Poet-v1"

SYSTEM_PROMPT = (
    "أنت «قريض»، شاعر عربي متمكّن من اللغة والعروض والقافية. "
    "تكتب شعرًا عربيًا فصيحًا موزونًا ومقفّى، وتلتزم بعدد الأبيات المطلوب دون شرح أو تعليق."
)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")
model.eval()

NUM_WORDS = {1: "بيتًا واحدًا", 2: "بيتين", 3: "ثلاثة أبيات", 4: "أربعة أبيات"}
THEMES = ["بدون موضوع", "الحزن", "الغزل", "المدح", "العتاب", "الهجاء",
          "الرثاء", "الشوق", "الفراق", "الوطنية", "الدين والتقوى"]


def build_prompt(free_text, n, theme):
    if free_text and free_text.strip():
        return free_text.strip()
    n_txt = NUM_WORDS[int(n)]
    if theme == "بدون موضوع":
        return f"اكتب {n_txt} من الشعر العربي."
    return f"اكتب {n_txt} من الشعر في {theme}."


def respond(free_text, n, theme, temperature, max_new_tokens):
    prompt = build_prompt(free_text, n, theme)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=int(max_new_tokens),
                  do_sample=True, temperature=float(temperature), top_p=0.9,
                  repetition_penalty=1.1, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    threading.Thread(target=model.generate, kwargs=kwargs).start()

    out = ""
    for piece in streamer:
        out += piece
        yield out


with gr.Blocks(title="قريض — Qareed") as demo:
    gr.Markdown("# 🪶 قريض (Qareed)\nتجريبي (v1): القافية والوزن غير مضبوطين، والمخرجات تحتاج مراجعة بشرية.")
    with gr.Row():
        with gr.Column():
            free = gr.Textbox(label="طلب حر (اختياري، يتجاهل الخيارات تحته)", rtl=True,
                              placeholder="مثال: اكتب بيتين في الصبر")
            n = gr.Dropdown([1, 2, 3, 4], value=2, label="عدد الأبيات")
            theme = gr.Dropdown(THEMES, value="الشوق", label="الموضوع")
            temp = gr.Slider(0.3, 1.2, value=0.8, step=0.05, label="Temperature")
            max_new = gr.Slider(40, 200, value=120, step=10, label="Max new tokens")
            btn = gr.Button("اكتب", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="الناتج", lines=10, rtl=True)
    btn.click(respond, [free, n, theme, temp, max_new], out)

demo.launch(share=True, debug=True)