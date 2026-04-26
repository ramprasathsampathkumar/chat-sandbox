import sys
from config.settings import check_ollama_health, settings
from ui.gradio_app import build_ui


def preflight_checks() -> None:
    issues = []

    if not settings.openai_api_key:
        issues.append("  - OPENAI_API_KEY not set (OpenAI models will be unavailable)")

    if not check_ollama_health():
        issues.append(
            f"  - Ollama not reachable at {settings.ollama_base_url}\n"
            "    Start it with: ollama serve"
        )

    if issues:
        print("Startup warnings:")
        for issue in issues:
            print(issue)
        print()


if __name__ == "__main__":
    preflight_checks()
    demo = build_ui()
    import gradio as gr
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
