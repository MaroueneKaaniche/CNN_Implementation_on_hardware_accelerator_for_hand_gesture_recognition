# PyTorch Branch: CNN Implementation for Hand Gesture Recognition on MAX78000

This branch contains the PyTorch Python script for the CNN model and two folders for training and synthesis.

- The `training` folder contains:
  - Custom PyTorch model description using the MAX78000 custom methods.
  - Custom data loader responsible for data range transformation and providing PyTorch datasets.
  - Modified training script with introduced cross-validation.
  - Shell scripts to perform model training and quantization.

- The `synthesis` folder contains:
  - Network description YAML file describing the sequence of layers and hardware resource mapping.
  - Code generation shell script for calling the synthesizer.
  - Shell scripts for quantization and evaluation after quantization.

To use this repository:
1. Include training shell scripts, model description, and data loader in the MAX78000 training repository from Analog Devices' GitHub.
2. Include the network description and code generation shell script in the MAX78000 synthesis repository.
3. Train and quantize the model using the provided scripts.
4. Use the resulting checkpoint file as input for the synthesizer along with the YAML file and input data from the dataset.
5. The outcome will be a C project ready for deployment on the MAX78000 Feather Board.

