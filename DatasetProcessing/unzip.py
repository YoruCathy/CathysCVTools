import glob
import zipfile

file = glob.glob("*.zip")
for f in file:
    z = zipfile.ZipFile(f, 'r')
    z.extractall()
    z.close()
    
#unzip files
