import json

file_path = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\InsightFace\result_s2ep1\speaker_subtitle_annotated_video_annotation.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

subtitles = sorted(
    [value for key, value in data.items() if 'subtitle_id' in value],
    key=lambda x: x['subtitle_id']
)

skip_ids = [58, 59, 470]

for subtitle in subtitles:
    if subtitle['subtitle_id'] in skip_ids:
        continue
    print(subtitle.get('speaker_id'))