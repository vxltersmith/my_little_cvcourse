# PyTorch Lessons

This repository contains a series of lessons on PyTorch, covering basic tensor operations, data handling, distributed training, and model export. Each lesson is contained in its own Python file.

## How to start

First, clone repository:

```bash
git clone https://github.com/vxltersmith/my_little_cvcourse.git
```

To run the code in this repository, you need to have the following Python packages installed:

```bash
pip install torch torchvision pandas pillow tqdm
```
Or simply run following:

```bash
pip install -r requirements.txt
```

## Table of Contents

1. [Lesson 1: Basic Tensor Operations](#lesson-1-basic-tensor-operations)
2. [Lesson 2: Data Handling](#lesson-2-data-handling)
3. [Lesson 3: Distributed Data Parallel (DDP)](#lesson-3-distributed-data-parallel-ddp)
4. [Lesson 4: Simple Neural Network and ONNX Export](#lesson-4-simple-neural-network-and-onnx-export)

## Lesson 1: Basic Tensor Operations

### File: `lesson1_basic.py`

This lesson covers basic tensor operations in PyTorch, including tensor creation, reshaping, transposing, and moving tensors between CPU and GPU.

### Key Points:
- Tensor creation and manipulation
- Tensor reshaping and transposing
- Moving tensors between CPU and GPU
- Gradient computation with `requires_grad`

### How to Run:
```bash
python lesson1_basic.py
```

## Lesson 2: Data Handling

### File: `lesson2_data.py`

This lesson demonstrates how to handle data in PyTorch, including custom datasets, data augmentation, and data loading with `DataLoader`.

### Key Points:
- Creating a custom dataset
- Data augmentation with `torchvision.transforms`
- Data loading with `DataLoader`
- Performance comparison between CPU and GPU

### How to Run:
```bash
python lesson2_data.py
```

## Lesson 3: Distributed Data Parallel (DDP)

### File: `lesson3_ddp.py`

This lesson covers distributed training using PyTorch's `DistributedDataParallel` (DDP) module. It includes setting up the distributed environment, training a simple model, and logging with TensorBoard.

### Key Points:
- Setting up a distributed environment
- Using `DistributedDataParallel` for distributed training
- Logging with TensorBoard

### How to Run:
```bash
torchrun --nproc_per_node=4 lesson3_ddp.py
```

## Lesson 4: Simple Neural Network and ONNX Export

### File: `lesson4_simplenn.py`

This lesson demonstrates how to define, train, and export a simple neural network in PyTorch. The model is exported to the ONNX format for interoperability with other frameworks.

### Key Points:
- Defining a simple neural network
- Training the model
- Exporting the model to ONNX format

### How to Run:
```bash
python lesson4_simplenn.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or feedback, please contact [ai.inc.inc@gmail.com](mailto:ai.inc.inc@gmail.com).
```

### Explanation:

1. **Table of Contents**: Provides a quick navigation to each lesson.
2. **Lesson Descriptions**: Each lesson has a brief description, key points, and instructions on how to run the code.
3. **Requirements**: Lists the necessary Python packages.
4. **License**: Specifies the license under which the project is distributed.
5. **Contributing**: Encourages contributions and provides instructions on how to contribute.
6. **Contact**: Provides a way to contact the maintainer for questions or feedback.

This README file should help users understand the purpose of each lesson, how to run the code, and how to contribute to the project.

# my_little_cvcourse
Yet another introduction to deep CV with pytorch
