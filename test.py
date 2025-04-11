import numpy as np
import csv

feeling_names = ["awe", "frustration", "joy", "anger", "happy", "sad", "love", "grief", "fear", "compassion", "jealousy", "content", "grief", "relief", "excite", "disgust", "relax"]
feeling_times = np.zeros((len(feeling_names), 3))


feelings_file_name = "sub-02_task-ImaginedEmotion_events.txt"
with open(f"sources/{feelings_file_name}", 'r') as text_file:
    tsv_values = list(csv.reader(text_file, delimiter="\t"))

    # print(feeling_names[0], tsv_values[9][6].strip(), tsv_values[9][6].strip() == feeling_names[0])

    #for row in range(len(tsv_values)):
    #    print(row, tsv_values[row][6].strip(), tsv_values[row][6].strip() in feeling_names)

    counter = 0
    for row in range(len(tsv_values)):
        if(tsv_values[row][6].strip() in feeling_names):
            feeling_times[counter][0] = feeling_names.index(tsv_values[row][6].strip())
            feeling_times[counter][1] = float(tsv_values[row][0].strip())
            while(tsv_values[row][6].strip() != "exit"):
                row += 1
            feeling_times[counter][2] = float(tsv_values[row][0].strip())
            counter += 1

for feeling_no in range(len(feeling_names)):
    print(f'{feeling_no + 1}. Feeling: {feeling_names[int(feeling_times[feeling_no][0])]}, Start: {feeling_times[feeling_no][1]}, End: {feeling_times[feeling_no][2]}')
