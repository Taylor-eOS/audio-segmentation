import os, sys, re
from pydub import AudioSegment

if len(sys.argv) != 3:
    print("Usage: python slice_audio.py <audio_file> <transcript_file>")
    sys.exit(1)
audio_file = sys.argv[1]
transcript_file = sys.argv[2]
output_folder = "segments"
os.makedirs(output_folder, exist_ok=True)
manifest_path = os.path.join(output_folder, "segments_manifest.txt")
pattern = re.compile(r"\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*(.+)")
audio = AudioSegment.from_file(audio_file)
manifest_lines = []
with open(transcript_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
for i, line in enumerate(lines, 1):
    m = pattern.match(line.strip())
    if not m:
        continue
    start_sec, end_sec, text = float(m.group(1)), float(m.group(2)), m.group(3)
    segment = audio[start_sec * 1000: end_sec * 1000]
    segment_filename = f"segment_{i:03d}.wav"
    segment_path = os.path.join(output_folder, segment_filename)
    segment.export(segment_path, format="wav")
    manifest_lines.append(f"{segment_filename}\t<text>{text}</text>")
with open(manifest_path, "w", encoding="utf-8") as f:
    f.write("\n".join(manifest_lines))
print("Done")

