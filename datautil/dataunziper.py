import os
import shutil
import zipfile
import tqdm

zip_dir_path = 'zip'
data_path = 'data'
data_name = 'data_{}.mat'

dir_in_zip = 'output/'

# Find all zip files in the zip directory
zip_files = [f for f in os.listdir(zip_dir_path) if f.endswith('.zip')]

# Unzip all files
ind = 0
for zip_file in tqdm.tqdm(zip_files):
    with zipfile.ZipFile(os.path.join(zip_dir_path, zip_file), 'r') as zip_ref:
        files_in_zip = zip_ref.namelist()
        mat_files = [f for f in files_in_zip if f.endswith('.mat')]
        # Extract all files in the zip file
        for f in mat_files:
            zip_ref.extract(f, data_path)
            shutil.move(os.path.join(data_path, f), os.path.join(data_path, data_name.format(ind)))
            ind += 1

shutil.rmtree(os.path.join(data_path, dir_in_zip))

print('Unzipped {} files'.format(ind))