import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image # Used to handle potential image loading errors
import shutil
import argparse
import numpy as np # Used for NaN representation
import textwrap # Used for wrapping long titles

# --- Configuration ---
CSV_FILE_PATH = '/projects/master/c5/mcv-c5-group-3/archive/food-ingredients-and-recipe-dataset-with-images/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'       # Path to your input CSV file
IMAGE_FOLDER_PATH = '/projects/master/c5/mcv-c5-group-3/archive/food-ingredients-and-recipe-dataset-with-images/Food Images/Food Images' # Path to the folder containing images
INTERMEDIATE_CSV_PATH = '/projects/master/c5/mcv-c5-group-3/archive/filtered_food_dataset/annotations_in_progress.csv' # File to save progress
FINAL_CSV_FILENAME = '/projects/master/c5/mcv-c5-group-3/archive/filtered_food_dataset/final_filtered_data.csv' # Output CSV for filtered data
FINAL_IMAGE_FOLDER_PATH = '/projects/master/c5/mcv-c5-group-3/archive/filtered_food_dataset/food_images' # Output folder for filtered images
ANNOTATION_COLUMN_NAME = 'Annotation'
# ---------------------

def load_data(csv_path, intermediate_path, annotation_col):
    """Loads data, starting from intermediate file if it exists."""
    if os.path.exists(intermediate_path):
        print(f"Resuming from intermediate file: {intermediate_path}")
        df = pd.read_csv(intermediate_path)
        # Ensure Annotation column exists and handle potential type issues after loading
        if annotation_col not in df.columns:
             df[annotation_col] = np.nan
        # Convert non-numeric entries (like empty strings from previous saves) back to NaN
        df[annotation_col] = pd.to_numeric(df[annotation_col], errors='coerce')

    else:
        print(f"Starting fresh from: {csv_path}")
        df = pd.read_csv(csv_path)
        df[annotation_col] = np.nan # Add the annotation column, initialized with NaN
    return df

def display_image(image_path, title_caption):
    """Displays the image using matplotlib with title and filename."""
    try:
        # Try opening with PIL first to catch more errors
        img_pil = Image.open(image_path)
        img_pil.verify() # Verify image integrity
        # Re-open after verify
        img = mpimg.imread(image_path)
        fig, ax = plt.subplots() # Get figure and axes objects
        ax.imshow(img)

        # --- Display Title and Filename ---
        # Wrap long titles to prevent them running off screen
        wrapped_title = "\n".join(textwrap.wrap(str(title_caption), width=60)) # Adjust width as needed
        display_text = f"Title: {wrapped_title}\nFile: {os.path.basename(image_path)}"
        fig.suptitle(display_text, fontsize=10)

        ax.axis('off') # Hide axes
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap, adjust rect as needed
        plt.show(block=False) # Show plot without blocking script execution
        return True
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        plt.close() # Close any potentially open plot window
        return False
    except Exception as e:
        print(f"Error displaying image {image_path}: {e}")
        plt.close()
        return False


def get_user_annotation():
    """Gets and validates user input for annotation."""
    while True:
        user_input = input("Enter annotation (0: Keep, 1: Filter, 2+: Class, 's': Save & Exit, 'q': Quit without saving): ").strip().lower()
        if user_input in ['s', 'q']:
            return user_input
        try:
            annotation = int(user_input)
            if annotation >= 0:
                return annotation
            else:
                print("Invalid input. Annotation must be 0 or greater.")
        except ValueError:
            print("Invalid input. Please enter a number (0, 1, 2,...), 's', or 'q'.")


def save_progress(df, intermediate_path):
    """Saves the DataFrame to the intermediate CSV."""
    # Ensure annotations are stored as integers or empty strings for better compatibility
    # Convert NaNs to empty string before saving, pandas reads empty string as NaN by default
    df_copy = df.copy()
    df_copy[ANNOTATION_COLUMN_NAME] = df_copy[ANNOTATION_COLUMN_NAME].apply(lambda x: '' if pd.isna(x) else int(x))
    df_copy.to_csv(intermediate_path, index=False)
    print(f"Progress saved to {intermediate_path}")

def finalize_results(df, annotation_col, image_folder, final_csv_path, final_image_folder):
    """Filters data, saves final CSV, and copies filtered images."""
    print("\nFinalizing results...")

    # Make sure the annotation column is numeric before filtering
    df[annotation_col] = pd.to_numeric(df[annotation_col], errors='coerce')

    # Filter out rows marked with 1 and rows that were never annotated (NaN)
    final_df = df[(df[annotation_col] != 1) & (df[annotation_col].notna())].copy()

    # Convert annotation column to integer type for the final output
    # Check if the column contains only finite values before converting
    if pd.api.types.is_numeric_dtype(final_df[annotation_col]) and final_df[annotation_col].notna().all():
         final_df[annotation_col] = final_df[annotation_col].astype(int)
    else:
         print(f"Warning: Could not convert {annotation_col} to integer type. May contain non-numeric or NaN values.")

    # Save the final filtered CSV
    final_df.to_csv(final_csv_path, index=False)
    print(f"Final filtered data saved to: {final_csv_path}")

    # Create the final image folder
    if not os.path.exists(final_image_folder):
        os.makedirs(final_image_folder)
    elif not os.path.isdir(final_image_folder):
         print(f"Error: {final_image_folder} exists but is not a directory.")
         return # Stop if the destination path is a file


    # Copy the corresponding images
    copied_count = 0
    skipped_count = 0
    print(f"Copying images to: {final_image_folder}...")
    for index, row in final_df.iterrows():
        # Construct image name again, assuming CSV doesn't have extension
        image_name_csv = row['Image_Name']
        image_name_with_ext = image_name_csv + ".jpg" # Consistent with main loop logic

        source_path = os.path.join(image_folder, image_name_with_ext)
        destination_path = os.path.join(final_image_folder, image_name_with_ext)

        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, destination_path) # copy2 preserves metadata
                copied_count += 1
            except Exception as e:
                print(f"Error copying {image_name_with_ext}: {e}")
                skipped_count += 1
        else:
            # Check if original name from CSV exists if name+.jpg failed
            source_path_orig = os.path.join(image_folder, image_name_csv)
            if os.path.exists(source_path_orig):
                 destination_path_orig = os.path.join(final_image_folder, image_name_csv)
                 try:
                    shutil.copy2(source_path_orig, destination_path_orig)
                    copied_count += 1
                    print(f"Warning: Copied image using name '{image_name_csv}' (without .jpg extension added)")
                 except Exception as e:
                    print(f"Error copying {image_name_csv}: {e}")
                    skipped_count += 1
            else:
                 print(f"Warning: Source image not found, skipping: {source_path} (and also checked {source_path_orig})")
                 skipped_count += 1


    print(f"Finished copying. {copied_count} images copied, {skipped_count} skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Food dataset annotation tool')
    parser.add_argument('--annotate_only_drinks', action='store_true',
                        help='If set, only prompt annotation for rows whose Classification == "drink"')
    args = parser.parse_args()

    dataframe = load_data(CSV_FILE_PATH, INTERMEDIATE_CSV_PATH, ANNOTATION_COLUMN_NAME)
    if args.annotate_drinks:
        if 'Classification' not in dataframe.columns:
            raise KeyError("`Classification` column not found in your CSV.")
        # case‑insensitive match
        is_drink = dataframe['Classification'].str.lower() == 'drink'
        # mark non‑drinks as already “kept”
        dataframe.loc[~is_drink, ANNOTATION_COLUMN_NAME] = 0

    total_items = len(dataframe)
    annotated_count = dataframe[ANNOTATION_COLUMN_NAME].notna().sum()
    print(f"Total items: {total_items}. Already annotated: {annotated_count}")

    total_items = len(dataframe)
    # Recalculate annotated count after loading and potential type coercion
    annotated_count = dataframe[ANNOTATION_COLUMN_NAME].notna().sum()
    print(f"Total items: {total_items}. Already annotated: {annotated_count}")

    # Get indices of rows where annotation is NaN
    indices_to_annotate = dataframe[dataframe[ANNOTATION_COLUMN_NAME].isna()].index

    # Iterate using indices to ensure we can modify the original DataFrame correctly
    for index in indices_to_annotate:
        row = dataframe.loc[index] # Get the row data
        image_name_csv = row['Image_Name'] # Original name from CSV
        title = row['Title']
        cleaned_ingedrients = row['Cleaned_Ingredients']

        # --- Construct Image Name and Path ---
        image_name_with_ext = str(image_name_csv) + ".jpg"
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_name_with_ext)
        # ---

        current_item_number = dataframe.index.get_loc(index) + 1 # Get user-friendly item number

        print(f"\n--- Annotating item {current_item_number}/{total_items} (Index: {index}) ---")
        print(f"Title: {title}")
        print(f"Image: {image_name_with_ext}") # Show name being used
        print(f"Cleaned Ingredients: {cleaned_ingedrients}")

        # Pass title to display_image function
        if display_image(image_path, title):
            annotation = get_user_annotation()
            plt.close() # Close the image window after getting input

            if isinstance(annotation, str): # User chose 's' or 'q'
                if annotation == 's':
                    # Ensure the last annotation is recorded before saving if it was numeric
                    # (This case is handled by the final save outside the loop)
                    print("Saving progress and exiting...")
                    save_progress(dataframe, INTERMEDIATE_CSV_PATH)
                else: # annotation == 'q'
                     print("Quitting without saving last annotation.")
                break # Exit the annotation loop

            dataframe.loc[index, ANNOTATION_COLUMN_NAME] = annotation
            # Save progress after every few annotations
            if (dataframe.index.get_loc(index) + 1) % 10 == 0:
               print(f"Auto-saving progress at item {current_item_number}...")
               save_progress(dataframe, INTERMEDIATE_CSV_PATH)

        else: # Image display failed
            print("Skipping this item due to image display/load error.")
            continue


    # --- After the loop (finished or exited) ---
    # Final save if loop completed naturally or user saved on exit
    # Check if 'annotation' was defined and wasn't 'q'
    if 'annotation' not in locals() or (isinstance(annotation, str) and annotation != 'q') or isinstance(annotation, int):
         print("Performing final save of annotation progress...")
         save_progress(dataframe, INTERMEDIATE_CSV_PATH) # Save final state before filtering

    # Ask user if they want to finalize (filter CSV and copy files)
    finalize_choice = input("\nAnnotation session finished. Do you want to generate the final filtered CSV and image folder? (y/n): ").strip().lower()
    if finalize_choice == 'y':
         finalize_results(dataframe, ANNOTATION_COLUMN_NAME, IMAGE_FOLDER_PATH, FINAL_CSV_FILENAME, FINAL_IMAGE_FOLDER_PATH)
    else:
         print("Final results not generated. You can run the script again to resume or finalize later.")

    print("\nScript finished.")