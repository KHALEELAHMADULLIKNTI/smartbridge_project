import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
import fitz  # PyMuPDF

# 📦 Load Granite Model
model_name = "ibm-granite/granite-3.3-2b-instruct"
print("🚀 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Model loaded on {device}")

# 💾 Load or Initialize Users
user_file = "users.json"
if os.path.exists(user_file):
    with open(user_file, "r") as f:
        users = json.load(f)
else:
    users = {
        "alice": {"password": "1234", "role": "student", "progress": {}},
        "bob": {"password": "abcd", "role": "teacher", "progress": {}},
        "admin": {"password": "admin", "role": "admin", "progress": {}}
    }

# 💾 Save Progress
def save_users():
    with open(user_file, "w") as f:
        json.dump(users, f, indent=2)

# 🧠 Session State
session_state = {"user": None}

# 🔐 Register & Login
def register(username, password, role):
    if username in users:
        return "❌ Username already exists!"
    if role not in ["student", "teacher", "admin"]:
        return "❌ Role must be student, teacher, or admin"
    users[username] = {"password": password, "role": role, "progress": {}}
    save_users()
    return f"✅ Registered {username} as {role}!"

def login(username, password):
    user = users.get(username)
    if user and user["password"] == password:
        session_state["user"] = {"name": username, "role": user["role"]}
        return f"✅ Logged in as {username} ({user['role']})"
    else:
        return "❌ Login failed"

# 📚 Tutor
def ai_tutor(subject, topic):
    if not session_state["user"]:
        return "⚠ Please login first."
    role = session_state["user"]["role"]
    prompt = (
        f"You are a helpful AI tutor for a {role}. Explain the following topic in {subject}:\n\n"
        f"Topic: {topic}\n\nExplanation:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    user = session_state["user"]["name"]
    users[user]["progress"][f"Tutor: {topic}"] = "Learned"
    save_users()
    return response

# 📝 Topic Quiz
def generate_quiz(subject, topic):
    if not session_state["user"]:
        return "⚠ Please login first."
    role = session_state["user"]["role"]
    prompt = (
        f"You are an AI quiz generator for a {role}. "
        f"Create 3 short quiz questions with answers about {topic} in {subject}.\n\n"
        "Format:\nQ1: ...\nA1: ...\nQ2: ...\nA2: ...\nQ3: ...\nA3: ..."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    quiz = tokenizer.decode(outputs[0], skip_special_tokens=True)
    user = session_state["user"]["name"]
    users[user]["progress"][f"Quiz: {topic}"] = "Generated"
    save_users()
    return quiz

# 📄 PDF Text Extractor
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

# 🧾 PDF Quiz
def generate_quiz_from_pdf(file):
    if not session_state["user"]:
        return "⚠ Please login first."
    text = extract_text_from_pdf(file)
    prompt = f"You are a teacher. Generate 5 questions with answers from this PDF:\n\n{text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    quiz = tokenizer.decode(outputs[0], skip_special_tokens=True)
    user = session_state["user"]["name"]
    users[user]["progress"]["PDF Quiz"] = "Generated"
    save_users()
    return quiz

# 📘 PDF Summary + Explanation
def summarize_pdf(file):
    if not session_state["user"]:
        return "⚠ Please login first."
    text = extract_text_from_pdf(file)
    summary_prompt = f"Summarize the following text in bullet points:\n\n{text}"
    summary = generate_response(summary_prompt)
    explain_prompt = f"Explain the following for a 15-year-old student:\n\n{summary}"
    explanation = generate_response(explain_prompt)
    user = session_state["user"]["name"]
    users[user]["progress"]["PDF Summary"] = "Completed"
    save_users()
    return f"🔹 Summary:\n{summary}\n\n📘 Explanation:\n{explanation}"

# 🔍 Utility
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 📈 View Progress
def view_progress():
    if not session_state["user"]:
        return "⚠ Please login first."
    user = session_state["user"]["name"]
    progress = users[user]["progress"]
    return "\n".join([f"{k}: {v}" for k, v in progress.items()]) or "No progress yet."

# 🌐 Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎓 EduTutor AI with PDF, Quiz & Progress Tracker")

    with gr.Tab("🔐 Register"):
        reg_username = gr.Textbox(label="Choose Username")
        reg_password = gr.Textbox(label="Choose Password", type="password")
        reg_role = gr.Dropdown(choices=["student", "teacher", "admin"], label="Role")
        reg_btn = gr.Button("Register")
        reg_status = gr.Textbox(label="Status", interactive=False)
        reg_btn.click(register, [reg_username, reg_password, reg_role], reg_status)

    with gr.Tab("🔓 Login"):
        login_username = gr.Textbox(label="Username")
        login_password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_status = gr.Textbox(label="Status", interactive=False)
        login_btn.click(login, [login_username, login_password], login_status)

    with gr.Tab("📚 AI Tutor"):
        subject = gr.Textbox(label="Subject (e.g. Math)")
        topic = gr.Textbox(label="Topic to explain")
        tutor_btn = gr.Button("Ask Tutor")
        tutor_out = gr.Textbox(label="Explanation")
        tutor_btn.click(ai_tutor, [subject, topic], tutor_out)

    with gr.Tab("📝 Topic Quiz"):
        q_subject = gr.Textbox(label="Subject")
        q_topic = gr.Textbox(label="Topic")
        quiz_btn = gr.Button("Generate Quiz")
        quiz_out = gr.Textbox(label="Quiz")
        quiz_btn.click(generate_quiz, [q_subject, q_topic], quiz_out)

    with gr.Tab("📄 PDF Quiz"):
        pdf_file = gr.File(label="Upload PDF", type="binary")
        pdf_quiz_btn = gr.Button("Generate Quiz from PDF")
        pdf_quiz_out = gr.Textbox(label="PDF-based Quiz")
        pdf_quiz_btn.click(generate_quiz_from_pdf, inputs=pdf_file, outputs=pdf_quiz_out)

    with gr.Tab("📘 PDF Summary"):
        pdf_sum_file = gr.File(label="Upload PDF", type="binary")
        sum_btn = gr.Button("Summarize & Explain")
        sum_out = gr.Textbox(label="Summary & Explanation")
        sum_btn.click(summarize_pdf, inputs=pdf_sum_file, outputs=sum_out)

    with gr.Tab("📈 Progress Tracker"):
        prog_btn = gr.Button("Show My Progress")
        prog_out = gr.Textbox(label="Progress")
        prog_btn.click(view_progress, outputs=prog_out)

demo.launch(share=True)