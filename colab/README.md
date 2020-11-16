## Google Colab Instructions for Project 2
In order to run the current version of the project inside Google Colab, please follow the following steps:
 * Import latest version of the notebook inside Google Colab
 * Go to top menu -> Runtime -> Change runtime type -> GPU
 * Upload colab.zip
 * Once upload is done, use the following command from within the notebook to unzip:
```python
import zipfile
with zipfile.ZipFile("colab.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
```
 * Then, inside the imports cell, right before tensorflow import, add:
```python
%tensorflow_version 1.x
```
 * Finally, go through the whole file and change very ../data into ./data - Be careful! cmd+f will not search inside wrapped cells
 * Run All and enjoy the speed of Google's GPU!
