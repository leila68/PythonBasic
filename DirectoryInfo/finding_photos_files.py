import os
import pandas as pd
import img2pdf
from PIL import Image
from zipfile import ZipFile
import pathlib

def make_zip_file(directory):
    dir_files_name = os.listdir(directory)
    print(dir_files_name)
    with ZipFile('zip_files.zip', 'w') as zip_object:
        for file in dir_files_name:
            if file.endswith(".csv"):
                # pdf_files.append(file)
                # print(os.path.join(file))
                file_path = os.path.join(directory, file)
                zip_object.write(file_path, os.path.basename(file))
    if os.path.exists(directory):
        print("ZIP file created")
    else:
        print("ZIP file not created")

def img_to_pdf(directory):
    dir_files_name = os.listdir(directory)
    print(dir_files_name)
    image_list = []

    for file in dir_files_name:
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
             # print(os.path.join(dir_path, file))
            image_1 = Image.open(os.path.join(dir_path, file))
            im_1 = image_1.convert('RGB')
            image_list.append(im_1)

    im_1.save(r"F:\ontariotech\all_photos.pdf", save_all=True, append_images=image_list)

if __name__ == "__main__":
    dir_path = "F:\ontariotech"
make_zip_file(dir_path)
img_to_pdf(dir_path)




