"""
Script to process generated images and update a CSV file with their information.
"""
import os
import argparse

import pandas as pd

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process generated images and update CSV file')
    parser.add_argument('--csv_path', type=str, 
                        default='/export/home/c5mcv03/mcv-c5-group-3/archive/drinks_dataset/annotations_llm_resuts_in_progress.csv',
                        help='Path to the CSV file')
    parser.add_argument('--image_dir', type=str, 
                        default='outputs/drinks/final_negative_prompting',
                        help='Directory containing generated images')
    parser.add_argument('--output_csv', type=str,
                        default='annotation_llm_results_with_generated.csv',
                        help='Output CSV file name')
    
    args = parser.parse_args()
    
    # Read the CSV file
    df = pd.read_csv(args.csv_path)
    
    # Get image files
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith('.png')]
    
    new_rows = []
    for image_name in image_files:
        title = image_name.replace('-gen.png', '').replace('-', ' ')
        
        new_row = {
            'Title': title,
            'Image_Name': image_name,
            'Classification': 'generated',
            'Annotation': 2,
        }
        new_rows.append(new_row)
    
    new_df = pd.DataFrame(new_rows)
    
    combined_df = pd.concat([df, new_df], ignore_index=True)
    combined_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
