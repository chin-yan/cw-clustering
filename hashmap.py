import pandas as pd
import os

# File path settings
file_path = r"C:\Users\VIPLAB\Downloads\1207.xlsx"
# Transformation mapping
transformation_map = {
    '4': 6,
    '6': 8,
    '8': 10,
    '9': 7,
    '10': 0,
    '11': 17,
    '7': 4,
    '12': 4,
    '14': 4,
    '15': 4,
    '16': 4,
    '13': 51
}

def transform_ground_truth(val):
    if pd.isna(val):
        return None  # Return None for existing NaNs
        
    str_val = str(val).strip()
    
    # Handle float conversion artifacts (e.g. 2.0 -> 2)
    if str_val.endswith('.0'):
        str_val = str_val[:-2]
        
    # 1. First check if it's in the map
    if str_val in transformation_map:
        return transformation_map[str_val]
    
    # 2. If not in map, check if it is a number
    # isdigit() handles positive integers. 
    # For more complex numbers (negatives/decimals not ending in .0), use try-except float.
    if str_val.isdigit() or (str_val.startswith('-') and str_val[1:].isdigit()):
        return val # It's a number (and not in the map), so keep it.
    
    # 3. If it's not in the map AND not a number, return None
    return None

def main():
    try:
        # Ensure all rows are printed
        pd.set_option('display.max_rows', None)

        # Load data
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        if 'ground truth' in df.columns:
            # Apply transformation
            result_series = df['ground truth'].apply(transform_ground_truth)
            
            # Print only the values directly to terminal for copying
            # to_string with header=False prints just the values.
            # na_rep='None' ensures None/NaN values are printed as the string "None"
            print(result_series.to_string(index=False, header=False, na_rep='None'))
            
        else:
            print("Error: Column 'ground truth' not found.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()