import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image # Used to handle potential image loading errors
import shutil
import numpy as np # Used for NaN representation

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

def display_image(image_path):
    """Displays the image using matplotlib."""
    try:
        # Try opening with PIL first to catch more errors
        img_pil = Image.open(image_path)
        img_pil.verify() # Verify image integrity
        # Re-open after verify
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.title(os.path.basename(image_path))
        plt.axis('off') # Hide axes
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
    df.to_csv(intermediate_path, index=False)
    print(f"Progress saved to {intermediate_path}")

def finalize_results(df, annotation_col, image_folder, final_csv_path, final_image_folder):
    """Filters data, saves final CSV, and copies filtered images."""
    print("\nFinalizing results...")

    # Filter out rows marked with 1 and rows that were never annotated (NaN)
    final_df = df[(df[annotation_col] != 1) & (df[annotation_col].notna())].copy()

    # Convert annotation column to integer type if possible
    final_df[annotation_col] = final_df[annotation_col].astype(int)

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
        image_name = row['Image_Name']
        source_path = os.path.join(image_folder, image_name)
        destination_path = os.path.join(final_image_folder, image_name)

        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, destination_path) # copy2 preserves metadata
                copied_count += 1
            except Exception as e:
                print(f"Error copying {image_name}: {e}")
                skipped_count += 1
        else:
            print(f"Warning: Source image not found, skipping: {source_path}")
            skipped_count += 1

    print(f"Finished copying. {copied_count} images copied, {skipped_count} skipped.")


# --- Main Script Logic ---
if __name__ == "__main__":
    dataframe = load_data(CSV_FILE_PATH, INTERMEDIATE_CSV_PATH, ANNOTATION_COLUMN_NAME)

    total_items = len(dataframe)
    annotated_count = dataframe[ANNOTATION_COLUMN_NAME].notna().sum()
    print(f"Total items: {total_items}. Already annotated: {annotated_count}")

    items_to_annotate = dataframe[dataframe[ANNOTATION_COLUMN_NAME].isna()]

    for index, row in items_to_annotate.iterrows():
        image_name = row['Image_Name'] + ".jpg"
        title = row['Title']
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)

        print(f"\n--- Annotating item {index+1}/{total_items} ---")
        print(f"Title: {title}")
        print(f"Image: {image_name}")

        if display_image(image_path):
            annotation = get_user_annotation()
            plt.close() # Close the image window after getting input

            if isinstance(annotation, str): # User chose 's' or 'q'
                if annotation == 's':
                    save_progress(dataframe, INTERMEDIATE_CSV_PATH)
                else: # annotation == 'q'
                     print("Quitting without saving last annotation.")
                break # Exit the annotation loop

            # Store valid numerical annotation
            dataframe.loc[index, ANNOTATION_COLUMN_NAME] = annotation
            # Save progress after every few annotations
            if (index + 1) % 10 == 0: # Save every 10 images
               save_progress(dataframe, INTERMEDIATE_CSV_PATH)

        else: # Image display failed
            print("Skipping this item due to image display error.")
            # Optionally mark as needing review or handle differently
            # dataframe.loc[index, ANNOTATION_COLUMN_NAME] = -1 # Example: Mark as error
            continue # Move to the next item


    # --- After the loop (finished or exited) ---
    # Final save if loop completed naturally or user saved on exit
    if 'annotation' not in locals() or annotation != 'q': # Check if loop finished or saved
         save_progress(dataframe, INTERMEDIATE_CSV_PATH) # Save final state before filtering

    # Ask user if they want to finalize (filter CSV and copy files)
    finalize_choice = input("\nAnnotation session finished. Do you want to generate the final filtered CSV and image folder? (y/n): ").strip().lower()
    if finalize_choice == 'y':
         # Ensure all NaN are handled before finalization if needed (e.g., treat as 'keep')
         # dataframe[ANNOTATION_COLUMN_NAME].fillna(0, inplace=True) # Example: fill NaN with 0 (Keep)
         finalize_results(dataframe, ANNOTATION_COLUMN_NAME, IMAGE_FOLDER_PATH, FINAL_CSV_FILENAME, FINAL_IMAGE_FOLDER_PATH)
    else:
         print("Final results not generated. You can run the script again to resume or finalize later.")

    print("\nScript finished.")