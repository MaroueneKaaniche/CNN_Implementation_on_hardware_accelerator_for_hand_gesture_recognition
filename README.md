# TensorFlow Branch: CNN Implementation for Hand Gesture Recognition on MAX78000

This branch contains the TensorFlow Python script for the CNN model and two folders for training and synthesis. 

- The `training` folder contains:
  - Custom model description using the MAX78000 custom methods.
  - Custom data loader for providing data to the model.
  - Modified training script with introduced cross-validation.
  - Shell scripts to perform model training and evaluation after quantization.

- The `synthesis` folder contains:
  - Network description YAML file describing the sequence of layers and hardware resource mapping.
  - Code generation shell script for calling the synthesizer.
  - Shell scripts for quantization using the synthesizer and post-quantization evaluation.

To use this repository:
1. Include training shell scripts, model description, and data loader in the MAX78000 training repository from Analog Devices' GitHub.
2. Include the network description and code generation shell script in the MAX78000 synthesis repository.
3. Train and quantize the model using the provided scripts.
4. Use the resulting ONNX file as input for the synthesizer along with the YAML file and input data from the dataset.
5. The outcome will be a C project ready for deployment on the MAX78000 Feather Board.

