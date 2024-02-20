import torch.serialization
import os

# Specify the path to the .safetensors file
file_path = "/home/angelique/.cache/huggingface/hub/models--codellama--CodeLlama-7b-hf/snapshots/7f22f0a5f7991355a2c3867923359ec4ed0b58bf/model-00001-of-00002.safetensors"

if os.path.exists(file_path):
    print(f"The file {file_path} exists.")
    with open(file_path, "rb") as f:
        data = torch.load(f)
        #torch.serialization.pickle.load(f)
        print('myData ', data)
else:
    print(f"The file {file_path} does not exist.")
# Load the contents of the .safetensors file
"""  """
# Access the tensors from the loaded data
# For example, if the file contains a single tensor:
""" your_tensor = data

# Now you can use 'your_tensor' as a regular PyTorch tensor
# Perform operations on the tensor or access its data as needed
your_tensor_numpy = your_tensor.numpy()

# Access the data from the NumPy array
print(your_tensor_numpy) """
