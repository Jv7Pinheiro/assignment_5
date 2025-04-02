import json
import gzip

# Define file paths
input_json_file = "/deac/csc/classes/csc373/deolj19/assignment_5/data/steam_reviews_clean_data.json"   # Your input JSON file
output_gz_file = "/deac/csc/classes/csc373/deolj19/assignment_5/data/steam_reviews_clean_data.json.gz" # Output compressed file

with gzip.open(output_gz_file, 'wt', encoding='utf-8') as f_out:
    with open(input_json_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            json_obj = json.loads(line)  # Parse each line separately
            json.dump(json_obj, f_out)
            f_out.write("\n")  # Ensure each JSON object is on a new line

print(f"File compressed successfully: {output_gz_file}")