import modal

app = modal.App("modal-reader")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi",
        "gradio",
        "kokoro",
        "torch",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    min_containers=1,
    scaledown_window=60 * 20,
    # gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 100 concurrent inputs
    max_containers=1,
    gpu="A10G",
    timeout=600,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import os
    import torch
    import gradio as gr
    from fastapi import FastAPI
    from kokoro import KModel, KPipeline
    from gradio.routes import mount_gradio_app

    CHAR_LIMIT = None
    CUDA_AVAILABLE = torch.cuda.is_available()

    models = {
        gpu: KModel().to("cuda" if gpu else "cpu").eval()
        for gpu in [False] + ([True] if CUDA_AVAILABLE else [])
    }
    pipelines = {
        lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in "ab"
    }
    pipelines["a"].g2p.lexicon.golds["kokoro"] = "kˈOkəɹO"
    pipelines["b"].g2p.lexicon.golds["kokoro"] = "kˈQkəɹQ"

    def forward_gpu(ps, ref_s, speed):
        return models[True](ps, ref_s, speed)

    def generate_first(text, voice="af_heart", speed=1, use_gpu=CUDA_AVAILABLE):
        text = text if CHAR_LIMIT is None else text.strip()[:CHAR_LIMIT]
        pipeline = pipelines[voice[0]]
        pack = pipeline.load_voice(voice)
        use_gpu = use_gpu and CUDA_AVAILABLE
        for _, ps, _ in pipeline(text, voice, speed):
            ref_s = pack[len(ps) - 1]
            try:
                if use_gpu:
                    audio = forward_gpu(ps, ref_s, speed)
                else:
                    audio = models[False](ps, ref_s, speed)
            except gr.exceptions.Error as e:
                if use_gpu:
                    gr.Warning(str(e))
                    gr.Info(
                        "Retrying with CPU. To avoid this error, change Hardware to CPU."
                    )
                    audio = models[False](ps, ref_s, speed)
                else:
                    raise gr.Error(e)
            return (24000, audio.numpy()), ps
        return None, ""

    # Arena API
    def predict(text, voice="af_heart", speed=1):
        return generate_first(text, voice, speed, use_gpu=False)[0]

    def tokenize_first(text, voice="af_heart"):
        pipeline = pipelines[voice[0]]
        for _, ps, _ in pipeline(text, voice):
            return ps
        return ""

    def generate_all(text, voice="af_heart", speed=1, use_gpu=CUDA_AVAILABLE):
        text = text if CHAR_LIMIT is None else text.strip()[:CHAR_LIMIT]
        pipeline = pipelines[voice[0]]
        pack = pipeline.load_voice(voice)
        use_gpu = use_gpu and CUDA_AVAILABLE
        first = True
        for _, ps, _ in pipeline(text, voice, speed):
            ref_s = pack[len(ps) - 1]
            try:
                if use_gpu:
                    audio = forward_gpu(ps, ref_s, speed)
                else:
                    audio = models[False](ps, ref_s, speed)
            except gr.exceptions.Error as e:
                if use_gpu:
                    gr.Warning(str(e))
                    gr.Info("Switching to CPU")
                    audio = models[False](ps, ref_s, speed)
                else:
                    raise gr.Error(e)
            yield 24000, audio.numpy()
            if first:
                first = False
                yield 24000, torch.zeros(1).numpy()

    CHOICES = {
        "🇺🇸 🚺 Heart ❤️": "af_heart",
        "🇺🇸 🚺 Bella 🔥": "af_bella",
        "🇺🇸 🚺 Nicole 🎧": "af_nicole",
        "🇺🇸 🚺 Aoede": "af_aoede",
        "🇺🇸 🚺 Kore": "af_kore",
        "🇺🇸 🚺 Sarah": "af_sarah",
        "🇺🇸 🚺 Nova": "af_nova",
        "🇺🇸 🚺 Sky": "af_sky",
        "🇺🇸 🚺 Alloy": "af_alloy",
        "🇺🇸 🚺 Jessica": "af_jessica",
        "🇺🇸 🚺 River": "af_river",
        "🇺🇸 🚹 Michael": "am_michael",
        "🇺🇸 🚹 Fenrir": "am_fenrir",
        "🇺🇸 🚹 Puck": "am_puck",
        "🇺🇸 🚹 Echo": "am_echo",
        "🇺🇸 🚹 Eric": "am_eric",
        "🇺🇸 🚹 Liam": "am_liam",
        "🇺🇸 🚹 Onyx": "am_onyx",
        "🇺🇸 🚹 Santa": "am_santa",
        "🇺🇸 🚹 Adam": "am_adam",
        "🇬🇧 🚺 Emma": "bf_emma",
        "🇬🇧 🚺 Isabella": "bf_isabella",
        "🇬🇧 🚺 Alice": "bf_alice",
        "🇬🇧 🚺 Lily": "bf_lily",
        "🇬🇧 🚹 George": "bm_george",
        "🇬🇧 🚹 Fable": "bm_fable",
        "🇬🇧 🚹 Lewis": "bm_lewis",
        "🇬🇧 🚹 Daniel": "bm_daniel",
    }
    for v in CHOICES.values():
        pipelines[v[0]].load_voice(v)

    TOKEN_NOTE = """
    💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`
    💬 To adjust intonation, try punctuation `;:,.!?—…"()“”` or stress `ˈ` and `ˌ`
    ⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`
    ⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)
    """

    with gr.Blocks() as generate_tab:
        out_audio = gr.Audio(
            label="Output Audio", interactive=False, streaming=False, autoplay=True
        )
        generate_btn = gr.Button("Generate", variant="primary")
        with gr.Accordion("Output Tokens", open=True):
            out_ps = gr.Textbox(
                interactive=False,
                show_label=False,
                info="Tokens used to generate the audio, up to 510 context length.",
            )
            tokenize_btn = gr.Button("Tokenize", variant="secondary")
            gr.Markdown(TOKEN_NOTE)
            predict_btn = gr.Button("Predict", variant="secondary", visible=False)

    STREAM_NOTE = [
        "⚠️ There is an unknown Gradio bug that might yield no audio the first time you click `Stream`."
    ]
    if CHAR_LIMIT is not None:
        STREAM_NOTE.append(f"✂️ Each stream is capped at {CHAR_LIMIT} characters.")
        STREAM_NOTE.append(
            "🚀 Want more characters? You can [use Kokoro directly](https://huggingface.co/hexgrad/Kokoro-82M#usage) or duplicate this space:"
        )
    STREAM_NOTE = "\n\n".join(STREAM_NOTE)

    with gr.Blocks() as stream_tab:
        out_stream = gr.Audio(
            label="Output Audio Stream",
            interactive=False,
            streaming=True,
            autoplay=True,
        )
        with gr.Row():
            stream_btn = gr.Button("Stream", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")
        with gr.Accordion("Note", open=True):
            gr.Markdown(STREAM_NOTE)
            gr.DuplicateButton()

    BANNER_TEXT = """
    [***Kokoro*** **is an open-weight TTS model with 82 million parameters.**](https://huggingface.co/hexgrad/Kokoro-82M)
    This demo only showcases English, but you can directly use the model to access other languages.
    """
    API_OPEN = os.getenv("SPACE_ID") != "hexgrad/Kokoro-TTS"
    API_NAME = None if API_OPEN else False
    with gr.Blocks() as app:
        with gr.Row():
            gr.Markdown(BANNER_TEXT, container=True)
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Input Text",
                    info=f"Up to ~500 characters per Generate, or {'∞' if CHAR_LIMIT is None else CHAR_LIMIT} characters per Stream",
                )
                with gr.Row():
                    voice = gr.Dropdown(
                        list(CHOICES.items()),
                        value="af_heart",
                        label="Voice",
                        info="Quality and availability vary by language",
                    )
                    use_gpu = gr.Dropdown(
                        [("ZeroGPU 🚀", True), ("CPU 🐌", False)],
                        value=CUDA_AVAILABLE,
                        label="Hardware",
                        info="GPU is usually faster, but has a usage quota",
                        interactive=CUDA_AVAILABLE,
                    )
                speed = gr.Slider(
                    minimum=0.5, maximum=2, value=1, step=0.1, label="Speed"
                )
            with gr.Column():
                gr.TabbedInterface([generate_tab, stream_tab], ["Generate", "Stream"])

        generate_btn.click(
            fn=generate_first,
            inputs=[text, voice, speed, use_gpu],
            outputs=[out_audio, out_ps],
            api_name=API_NAME,
        )
        tokenize_btn.click(
            fn=tokenize_first, inputs=[text, voice], outputs=[out_ps], api_name=API_NAME
        )
        stream_event = stream_btn.click(
            fn=generate_all,
            inputs=[text, voice, speed, use_gpu],
            outputs=[out_stream],
            api_name=API_NAME,
        )
        stop_btn.click(fn=None, cancels=stream_event)
        predict_btn.click(
            fn=predict,
            inputs=[text, voice, speed],
            outputs=[out_audio],
            api_name=API_NAME,
        )

    app.queue(api_open=API_OPEN).launch(show_api=API_OPEN, ssr_mode=True)

    return mount_gradio_app(app=FastAPI(), blocks=app, path="/")
