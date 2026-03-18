"""
app.py — Gradio UI for Fasih-Docs 
"""
 
import gradio as gr
from pathlib import Path
 
from rag_pipeline import RAGPipeline
import config
 
print("Starting Fasih-Docs...")
pipeline = RAGPipeline()
 
EXAMPLE_QUESTIONS = [
    "What is the main purpose of this document?",
    "What are the most important technical requirements?",
    "What file formats are supported?",
    "Explain the key design rules.",
    "ما هو الهدف الرئيسي من هذا الوثيقة؟",
]
 
 
def format_sources_html(sources: list) -> str:
    if not sources:
        return "<p style='color:#888;font-size:13px;'>No sources retrieved.</p>"
    html = "<div style='font-size:13px;'>"
    for i, src in enumerate(sources, 1):
        html += f"""
        <div style='background:#1e2a3a;border-left:3px solid #4a9eff;
                    padding:10px 14px;margin-bottom:8px;border-radius:4px;'>
            <b style='color:#4a9eff;'>Source {i}</b>
            <span style='color:#aaa;margin-left:10px;'>
                {src['file']} - Page {src['page']}
            </span>
            <p style='color:#ccc;margin:6px 0 0 0;font-size:12px;font-style:italic;'>
                "{src['preview']}"
            </p>
        </div>"""
    html += "</div>"
    return html
 
 
def answer_question(question: str, history: list):
    if not question.strip():
        return history, "", "<p style='color:#888;'>Ask a question above.</p>"
 
    result = pipeline.query(question)
    answer = result["answer"]
    sources = result["sources"]
 
    history = history or []
    history.append([question, answer])
 
    return history, "", format_sources_html(sources)
 
 
def clear_chat():
    return [], "", "<p style='color:#888;'>Sources will appear here.</p>"
 
 
CSS = """
.gradio-container { max-width: 1100px !important; margin: auto !important; }
"""
 
with gr.Blocks(theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
               css=CSS, title="Fasih-Docs") as demo:
 
    gr.HTML("""
        <div style='background:linear-gradient(135deg,#0f1923,#1a2d44);
                    border:1px solid #2a4a6b;border-radius:10px;
                    padding:20px 28px;margin-bottom:16px;'>
            <h1 style='color:#4a9eff;margin:0;font-size:26px;font-weight:700;'>
                Fasih-Docs
            </h1>
            <p style='color:#8ba8c4;margin:6px 0 0 0;font-size:14px;'>
                AI-powered assistant for engineering documentation - Arabic and English - Mistral 7B + RAG
            </p>
        </div>
    """)
 
    with gr.Row():
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=480,
                show_label=False,
                type="tuples",
            )
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Ask anything about the loaded documentation... (Arabic or English)",
                    label="",
                    lines=2,
                    scale=5,
                )
                with gr.Column(scale=1, min_width=80):
                    submit_btn = gr.Button("Ask", variant="primary", size="lg")
                    clear_btn  = gr.Button("Clear", size="sm")
 
            gr.Examples(examples=EXAMPLE_QUESTIONS,
                        inputs=question_input,
                        label="Example Questions")
 
        with gr.Column(scale=4):
            gr.HTML("<p style='color:#8ba8c4;font-size:13px;font-weight:600;"
                    "margin-bottom:8px;'>RETRIEVED SOURCES</p>")
            sources_display = gr.HTML(
                value="<p style='color:#888;font-size:13px;'>"
                      "Sources will appear here after your first question.</p>"
            )
            gr.HTML(f"""
                <div style='background:#0f1923;border:1px solid #2a4a6b;
                            border-radius:8px;padding:12px 14px;margin-top:12px;
                            font-size:12px;color:#8ba8c4;'>
                    <b style='color:#4a9eff;'>System Info</b><br>
                    Model: Mistral 7B (local via Ollama)<br>
                    Embeddings: multilingual-e5-small<br>
                    Vector DB: ChromaDB (local)<br>
                    Logs: {config.LOG_FILE}
                </div>
            """)
 
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input, sources_display],
    )
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input, sources_display],
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, question_input, sources_display],
    )
 
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )
 