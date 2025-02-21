from transformers import pipeline
import os, sys

asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")

def process_file(file_path):
    result = asr_model(file_path, return_timestamps=True)
    output_text = ""
    for chunk in result.get("chunks", []):
        start, end = chunk.get("timestamp", [None, None])
        text = chunk.get("text", "")
        if start is not None and end is not None:
            output_text += f"[{start:.2f} - {end:.2f}] {text}\n"
        else:
            output_text += f"{text}\n"
    base_name = os.path.splitext(file_path)[0]
    output_file = f"{base_name}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)
    print(f"Transcription saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python whis.py <file_path>")
        file_path = input("Enter file path: ")
    else:
        file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print("Provided path is not a file.")
        sys.exit(1)
    process_file(file_path)

