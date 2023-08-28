import os
import  pandas as pd

# Get the list of all files and directories
dir_path = input("Enter the path: ")
# dir_path = "F:\pdf"
dir_files_name = os.listdir(dir_path)

# Lists containing data directory
files = []
paths = []
sizes = []

def get_size(start_path = dir_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            # print(f)
            fp = os.path.join(dirpath, f)
            fs = os.path.getsize(fp)
            files.append(f)
            paths.append(fp)
            sizes.append(fs)
            # print(fp)
            # print(fs)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size = os.path.getsize(fp)
    return total_size
print(get_size(),'bytes')

# storing in pandas dataframe
df = pd.DataFrame({'Film Name': files, 'Path': paths, 'Size': sizes})
# making csv using pandas
df.to_csv('files_info.csv', index=False, encoding='utf-8')