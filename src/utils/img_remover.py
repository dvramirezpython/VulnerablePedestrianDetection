import os

def delete_files_from_list(folder_path, txt_file_path):
    try:
        # Read the list of filenames from the txt file
        with open(txt_file_path, 'r') as file:
            files_to_delete = [line.strip() for line in file if line.strip()]

        # Delete each file if it exists in the folder
        for filename in files_to_delete:
            # Added to delete labels
            filename = os.path.splitext(filename)[0] + '.txt'
            
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            else:
                print(f"File not found: {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
folder = 'dataset/dataset_all_weather_pedestrian_vulnerable_v2/train/labels'
txt_file = 'runs/train.txt'
delete_files_from_list(folder, txt_file)
