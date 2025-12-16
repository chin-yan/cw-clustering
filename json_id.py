import json
import os

def extract_subtitle_data(json_file_path):
    # --- CONFIGURATION ---
    subtitle_ids_to_skip = [37, 38, 473] 
    # ---------------------

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        items = list(data.values())
        items.sort(key=lambda x: x.get('subtitle_id', 0))

        text_list = []
        speaker_id_list = []

        for item in items:
            s_id = item.get('subtitle_id')
            
            if s_id in subtitle_ids_to_skip:
                continue

            # Process Text
            raw_text = item.get('text', '')
            # Replace newline with space
            clean_text = raw_text.replace('\n', ' ').strip()
            
            # CRITICAL FIX: If text is empty, use a placeholder so the line is visible
            if not clean_text:
                clean_text = "[EMPTY_TEXT]"
            
            text_list.append(clean_text)

            # Process Speaker ID
            sp_id = item.get('speaker_id')
            if sp_id is None:
                speaker_id_list.append("None")
            else:
                speaker_id_list.append(str(sp_id))

        # --- DEBUG INFO ---
        print(f"DEBUG INFO: Found {len(text_list)} texts and {len(speaker_id_list)} speaker_ids.")
        if len(text_list) == len(speaker_id_list):
            print("counts match! (Please make sure to copy [EMPTY_TEXT] lines)")
        else:
            print("Counts do not match. Please check logic.")
        print("-" * 30)
        print("\n")

        # OUTPUT SECTION
        print("--- COPY TEXT BELOW ---")
        for t in text_list:
            print(t)
        
        print("\n" * 3)
        
        print("--- COPY SPEAKER_ID BELOW ---")
        for s in speaker_id_list:
            print(s)

    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    filename = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\InsightFace\result_s2ep7\speaker_subtitle_annotated_video_3.0_annotation_3.0.json"
    extract_subtitle_data(filename)