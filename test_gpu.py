from pathlib import Path
import onnxruntime_genai as og

MODEL_DIR = Path("models/phi3/cuda/cuda-int4-rtn-block-32")

def main():
    if not (MODEL_DIR / "genai_config.json").exists():
        raise FileNotFoundError(f"Model not found: {MODEL_DIR.resolve()}")

    print("Loading Phi-3 Mini CUDA model...")
    model = og.Model(str(MODEL_DIR))
    tokenizer = og.Tokenizer(model)
    stream = tokenizer.create_stream()

    prompt = (
        "<|user|>\n"
        "Explain artificial intelligence in three simple sentences."
        "<|end|>\n<|assistant|>\n"
    )

    input_tokens = tokenizer.encode(prompt)
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=len(input_tokens) + 80,
        do_sample=False,
        top_k=1,
    )

    generator = og.Generator(model, params)
    try:
        generator.append_tokens(input_tokens)
        print("\nOpenMind AI response:\n")

        while not generator.is_done():
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            print(stream.decode(token), end="", flush=True)

        print("\n\nGeneration finished.")
    finally:
        del generator

if __name__ == "__main__":
    main()
