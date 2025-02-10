---
title: 'GPU Utilization in Daytona for LLM Fine-Tuning and Inference'
description:
  'Learn how to setup gpu utilization in daytona and use it for llm 
  fine tuning and inference'
date: 2025-01-05
author: 'Tata Shalom'
tags: ['daytona', 'llm', 'gpu']
---

# GPU Utilization in Daytona for LLM Fine Tuning and Inference

## Introduction

GPU Utilization refers to the effective and practical real world usage of a GPU's resources to accomplish specific tasks. It is typically expressed as a percentage of the GPU's maximum capacity

This guide you will walk you through setting up a Daytona environment for [LLM Fine-Tuning](/definitions/20250105_definition_llm_fine_tuning.md) and [LLM Inference](/definitions/20250105_definition_llm_inference.md). This article assumes you have a [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus)
,and a Linux based system(Windows users have to use WSL).You will use [Docker](https://docs.docker.com/get-started/get-docker/) containers, [Daytona](https://www.daytona.io/docs/installation/installation/) and an IDE like [VS Code](https://code.visualstudio.com/download).

## TL;DR

- **Required Tools & Knowledge**: Tools and skills required for running this project.
- **Overview of LLM Inference and Fine-Tuning**: Overview and key details about LLM inference and fine-tuning with respect to GPU utilization.
- **Preparation Steps**: Important installation steps to ensure the setup works as intended.
- **Setting Up the Dev Container & Project Repository**: Required steps for creating the development environment.
- **LLM Inference and Fine-Tuning in Daytona**: Utilizing Nvidia GPU for the purpose of running fine-tuning and inference workflows.
- **Special Consideration for Windows Users**: considerations for Windows users with respect to driver and Nvidia Container Toolkit installation.
- **Common Issues and Troubleshooting**: Common Issues that could arise while using thisb guide and how to solve them.

## Prerequsites and System Architecture

Before starting the process for GPU-based LLM fine-tuning and
inference with Daytona, ensure your system meets the following
requirements:

**Hardware**:
- **CPU**: x86_64 (amd64) or ARM architecture(Nvidia Jetson devices)
- **GPU**: CUDA-compatible NVIDIA GPU (e.g. RTX 20xx,30xx, Quadro series, Axx,Hxx)
- **Memory**: Minimum of 4GB RAM recommended, 16GB or more for
better performance

**Software**:
- **Operating System**: Linux debian based distro like Ubuntu and
Windows via WSL2 Ubuntu distribution.
- **Docker**: Must be instsalled and cofigured for running
containers.
- **Nvidia Drivers**: Ensure the correct GPU drivers for your
GPU is installed. you can verify this by running `nvidia-smi`

**Environment**:
- **CUDA**: Install the correct version of the CUDA toolkit
matching your GPU drivers. Ensure compatibility between Docker
images and the driver version, Ideally you should use a docker image lower than the CUDA driver version i.e if your CUDA driver
version is `12.4` you should use a docker image with `12.3.x` tag.
- **WSL**: For windows users, ensure that WSL2 is properly
configured to allow GPU access within Linux environments.

## Overview of GPU Utilization for LLM Fine-Tuning

[Fine-tuning](/definitions/20250105_definition_llm_fine_tuning.md) is the process of adapting a pre-trained Large Language Model (LLM) to perform specific
tasks or excel in particular fields. Fine-tuning is necessary because pre-trained Large Language Models (LLMs),
while good at general language understanding, often lack the precision required for numerous fields.

### GPU Utilzation in LLM Fine-Tuning

- **Compute Activity** : Fine-Tuning is computationally intensive and often requires high GPU Utilization due to backpropagation and large datasets.
- **Memory Usage**: Fine-Tuning requires a lot of GPU memory becausee it stores the model, gradients, optimizer states and intermediate activations.

## Overview of LLM Inference

[LLM Inference](/definitions/20250105_definition_llm_inference.md) refers to the phase where a trained LLM is used to generate predictions or responses based on queries.
During inference, the model processes the input, retrieves relevant learned patterns, and outputs the most contextually
appropriate text.. The process begins with [tokenizing](/definitions/20250105_definition_tokenization.md) the input into smaller units, such as words or subwords, which are
numerically encoded for model processing. The encoded tokens are passed through the LLM, where neural networks identify
contextual relationships between tokens. Based on these computations, the model predicts the most likely next token(s)
iteratively, generating the output text one token at a time. This continues until a stopping criterion, such as a specific
token or maximum length, is met. Finally, the generated tokens are decoded back into text and formattedas the final response.

### GPU Utilzation in LLM Inference

- **Compute Activity**: Inference typically requires less computational power compared to training or Fine-Tuning. GPU Utilization during inference depends on factors like:
  - **Model Size**: Larger Models like GPT-4 require more GPU resources compared to smaller ones like GPT-2.
  - **Batch Size**: Processing multiple input at the same time increases GPU Utilization.
  - **Input Length**: Longer input or Ouput require more computation, hence increased GPU use.
- **Memory Usage**: Inference uses GPU memory to store the model weights and intermediate activations. Memory utilized is generally lower than training or fine-tuning.

## Step 1: Installation and Verification of all Essential Software

For this setup I am currently using a hardware setup(Physical GPU device) but if you dont have that you can follow along by  getting a gpu enable VM from cloud providers.

- **Step 1.1: Installation of Essential Build Tools**

  If you're using a GPU enabled Linux VM, install important
programs like git and build essential by running

  ```bash
    sudo apt-get install -y git build-essential
  ```

- **Step 1.2: Verify CUDA Compatibilty**
  
  You can verify if your Nvidia GPU is CUDA compatible by checking
  if your GPU model is on the list located [here](https://developer.nvidia.com/cuda-gpus)

- **Step 1.3: Verify/Install WSL(Windows users only)**  
For Windows Users, to verify if WSL is installed, Open up
  Powershell and run :
  
  ```bash
  wsl --status
  ````
  If WSL is installed it should show the default WSL version, kernel version and the default distribution. If WSL is not installed you can install it by running :

  ```bash
  wsl --install
  ```
- **Step 1.4: Verify Docker Installation**
  
  To verify if Docker is installed and running:

  Open up a terminal (Powershell, Linux shell e.t.c) and run
  this command :
  
  ```bash
  docker --version
  ```
  If Docker is installed it should output something like:

  ```bash
  Docker version 24.0.5, build a8a2b3b
  ```
- **Step 1.5: GPU Driver Installation**
  To verify [GPU driver](/definitions/20250105_definition_gpu_driver.md) status and install Nvidia Container Toolkit

  To verify your driver status open your terminal and run this command

  ```bash
  nvidia-smi
  ```
  You should get an output resembling the below image

  ![image of nvidia-smi output](assets/20250105_daytona_gpu_utilization_img_1.PNG)

  If you do not have drivers installed get them from [here](https://www.nvidia.com/en-us/drivers/)
  depending on your OS and GPU hardware,
- **Step 1.6: Install NVIDIA Container Toolkit**  
  If drivers are installled open your shell(WSL for Windows users) and check if the Nvidia Container Toolkit is installed with:

  ```bash
    dpkg -l | grep nvidia-container-toolkit
  ```
  - **Step 1.6.1: Add Production Repository**
  
    Run this command to run the production repository

    ```bash
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```
  - **Step 1.6.2: Update the Package List**
  
    Update the package list from the repository

    ```bash
      sudo apt-get update
    ```
  - **Step 1.6.3: Install the NVIDIA Container Toolkit packages**

    ```bash
      sudo apt-get install -y nvidia-container-toolkit
    ```
  - **Step 1.6.4: Configure the runtime**
  
    ```bash
      sudo nvidia-ctk runtime configure
      sudo systemctl restart docker
    ```
  Once everything is setup you can run a CUDA enabled container with GPU access with
  this command

  ```bash
    docker run --gpus all nvidia/cuda:12.0.1-base nvidia-smi
  ```
  Ensure that the Windows/Linux drivers are compatible with the CUDA version in your Docker image,
  i.e a driver version of 12.5 should be used with a CUDA docker image of version 12.5 or less
  since there is backward compatibilty between them.

## Step 2: Setup Dev Container Configuration for GPU Utilization
  
- **Step 2.1: Create a Dev Container configuration**
  
  Here, you're going to create a dev container consisting of a `devcontainer.json` file, a
  `Dockerfile` and Python Scripts.  
  
  Create a new directory and move inside of it

  ```bash
  mkdir daytona-gpu-utilization && cd daytona-gpu-utilization
  ```
- **Step 2.2 Create a .devcontainer directory**

    This is where your devcontainer.json and dockerfiles will live

    ```bash
    mkdir .devcontainer && cd .devcontainer
    ```
  - **Step 2.2.1 Create the `devcontainer.json` file**
  
      Inside the .devcontainer folder you are going to create a `devcontainer.json` file with the following code.
      This is the configuration file for the dev environment  specifying settings and dependencies.

    ```json
      {
      "name": "Daytona GPU Utilization",
      "build": {
          "context": "..",
          "dockerfile": "Dockerfile"
      },
      "customizations": {
        "vscode": {
              "extensions": [
                  "ms-python.python",
                  "ms-vscode.cpptools",
                  "ms-python.vscode-pylance"
              ]
          }
      },
      "runArgs": [
      "--gpus", "all"  // Ensures GPU access
    ],
      "forwardPorts": []
    } 
    ```
  Let's breakdown the `devcontainer.json` file

  - `name`: Specifies the name of the development environment.
  - `build`: Configures the `Dockerfile` and build related options
  - `customizations`:Allows customization of the development environment,
    specifically for VS Code.
  - `vscode`: Automatically installs Python extension for VS Code, enabling
    some IDE features (e.g. linting, definition, debugging e.t.c)
  - `runArgs`: This is the argument required to start the container. The variable `"[--gpus, "all"]`
    ensure the container starts with GPU access.

  - **Step 2.2.2: Create a Dockerfile**

    You will create a Dockerfile in the same directory

      ```dockerfile
      FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
      RUN apt-get update && apt-get install -y \
          python3-pip
      WORKDIR /workspace

      RUN pip install --upgrade pip \
          && pip install torch torchvision torchaudio transformers datasets accelerate torchmetrics

      COPY *.py /workspace/
      COPY data.txt /workspace/

      CMD [ "bash" ]
      ```
      Let's breakdown the `Dockerfile`

    - `FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`: This is the image provided by Nvidia which ensures GPU accelerated computations and provides optimized libraries for deep learning workloads. You may choose to use other image tags like `base`or `devel`
    depending on your specific need or complexity of operation
    - `RUN apt-get update && apt-get install -y \python3-pip`: Installs Python3 and `pip` Python's package
      manager. The flag `-y` automatically confirms prompts during installation.
    - `WORKDIR /workspace`: Sets the default working directory inside the container to `/workspace`, any subsequent
      operations like `COPY` and `RUN` will be relative to this directory.
    - `RUN pip install --upgrade pip \ pip install torch torchvision torchaudio transformers datasets accelerate torchmetrics`:
      Ensures the latest version of `pip` is being used and installs necessary libraries like `torch` is the core PyTorch library
      which is necessary for deep learning operations,`torchvision`which is an image processing utility for PyTorch,`torchaudio`
      is an audio processing utility for PyTorch,`transformers` is Huggingface library for NLP models like GPT,`datasets` is used
      to manage datasets, `accelerate` is used to simplify distributed training and `torchmetrics` is used to provide metrics for
      evaluating PyTorch models.
    - `COPY *.py /workspace/`: Copies all Python files from the host directory into the container.
    - `COPY data.txt /workspace/`: Copies the `train.txt` file to your workspace.
    - `CMD [ "bash" ]`: This opens up a shell prompt inside the container when you can execute your scripts manually.

- **Step 2.3: Create a GPU Availability Checker**

  Create a `gpu_check.py` script for the purpose of checking the GPU availability and GPU device name

  ```python
  import torch

  if torch.cuda.is_available():
      print("GPU is available.")
      print("Device name:", torch.cuda.get_device_name(0))

  else:
      print("GPU is not available.")
  ```
- **Step 2.4: Create Dataset file**
  
  Create a `data.txt` file which will provide the dataset for fine-tuning the LLM

  ```text
      Genghis Khan was born Temüjin, was the founder of the Mongol Empire, the largest contiguous empire in history. 
      Rising from humble beginnings, he united the Mongol tribes and became their leader in 1206. His military genius, 
      including strategic innovation and organized armies, led to vast conquests across Asia and parts of Europe. 
      Genghis Khan established a strong empire with efficient governance, promoting trade, religious tolerance, 
      and cultural exchange. Despite his brutal conquests, his legacy includes unifying the Mongols and laying the 
      foundation for global trade, with his descendants further expanding the empire.
  ```
- **Step 2.5: Create Fine-Tune Workflow Script**

  Create a `finetune_workflow.py` script that will be used to fine tune the LLM

  ```python
  from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
  from datasets import load_dataset

  # Load dataset
  dataset = load_dataset("text", data_files={"train": "data.txt"})

  # Load tokenizer and model
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  model = GPT2LMHeadModel.from_pretrained("gpt2")

  # Add or set a pad token
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as padding token

  # Tokenize data
  def tokenize_function(examples):
      tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=50)
      tokens["labels"] = tokens["input_ids"].copy()  # Add labels for computing the loss
      return tokens

  tokenized_dataset = dataset.map(tokenize_function, batched=True)

  # Handle small datasets: Skip split if the dataset is too small
  if len(tokenized_dataset["train"]) > 1:
      tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.5)
  else:
      # Use the same dataset for both train and test
      tokenized_dataset = {"train": tokenized_dataset["train"], "test": tokenized_dataset["train"]}

  # Training arguments
  training_args = TrainingArguments(
      output_dir="./results",
      overwrite_output_dir=True,
      num_train_epochs=300,
      per_device_train_batch_size=2,
      save_steps=10,
      save_total_limit=2,
      logging_dir="./logs",
      logging_steps=10,
  )

  # Trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_dataset["train"],
      eval_dataset=tokenized_dataset["test"],
  )

  # Fine-tune the model
  trainer.train()

  # Save the fine-tuned model
  model.save_pretrained("./fine_tuned_gpt2")
  tokenizer.save_pretrained("./fine_tuned_gpt2")
  ```
- **Step 2.6: Create Inference Workflow Script**

  Create a `inference_workflow.py` script which whill be used to run LLM inference and prompt the fine tuned
  GPT model

  ```python
  from transformers import GPT2Tokenizer, GPT2LMHeadModel

  # Load fine-tuned model
  tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
  model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")

  # Generate text
  input_text = "Who is Gengis Khan"
  input_ids = tokenizer.encode(input_text, return_tensors="pt")

  output = model.generate(input_ids, max_length=300, num_return_sequences=1)
  print(tokenizer.decode(output[0], skip_special_tokens=True))

  ```

- **Step 2.7 Commit To GitHub Repository**

  Initialize,commmit and create a GitHub repository

    ```bash
    git init
    git add .
    git commit -m "inital commit"
    ```

    After commiting your code you will push it to a remote repository of your
    choice.

    ```bash
    git remote add origin https://github.com/YOUR-GITHUB-USERNAME/YOUR-DIRECTORY-NAME.git
    git branch -M main
    git push -u origin main
    ```

## Step 3: Setup workspace environment in Daytona

If you are using a Linux VM you will need to SSH into the server
before attempting to build the workspace with Daytona. You can learn how to SSH into a Linux server [here](https://www.youtube.com/watch?v=QRlTJW8HYs4)
You should ensure `daytona` is
installed on you machine before proceeding.

- **Step 3.1: Start the Daytona Server**

  Execute the command provided below to start the `daytona` server daemon. when
  prompted to start the server in the current terminal session click `yes`

  ```bash
    daytona server
    ```
  Your output should look like the image below

    ![image of running daytona server](assets/20250105_daytona_gpu_utilization_img_2.png)

- **Step 3.2: Create a new workspace**

  Create a new workspace by running the below command
  ```bash
    daytona create
    ```
  Your output should look like the image below

    ![image of running daytona server](assets/20250105_daytona_gpu_utilization_img_3.png)

- **Step 3.3: Add Github Repository**

  Select and add the repository where your files are located and click enter, the workspace will start building,
  your output should look like the images below

  ![image of running daytona sworkspace creation 1](assets/20250105_daytona_gpu_utilization_img_4.png)
  ![image of running daytona workspace creation 2](assets/20250105_daytona_gpu_utilization_img_5.png)
  ![image of running daytona workspace creation 3](assets/20250105_daytona_gpu_utilization_img_6.png)

After the workspace completes the build process your preferred IDE will open automatically like the image
below
![image of running daytona server](assets/20250105_daytona_gpu_utilization_img_7.png)

## Step 4: Running LLM Fine-Tuning on GPT 2

- **Step 4.1: Verify GPU Availability**

  Before carrying out any operation you would want to verify if your GPU is available and being accessed, In
  order to achieve this navigate to your terminal and running this command

  ```bash
    python3 gpu_check.py
  ```
  Your output should look like the image below

    ![image of gpu check output](assets/20250105_daytona_gpu_utilization_img_8.png)

- **Step 4.2: Run Fine-Tuning Workflow Script**

  To perform LLM fine-tuning workflow run this command

  ```bash
    python3 finetune_workflow.py
  ```
  Your output should look like the image below

    ![image of running fine tune workflow](assets/20250105_daytona_gpu_utilization_img_9.png)

  - `loss`: The loss measures how far off the model's prediction are from true values. It quantifies the error
  during training. The training process aims to reduce this value.
  - `grad_norm`: The norm (magnitude) of the gradient vector, which indicates the size of
  the change in model parameters during optimization. This helps monitor and control the optimization process.
  If the gradient norm is too large, it can lead to unstable updates.
  - `learning_rate`: A hyperparameter that controls the size of the steps the optimization algorithm takes during
  training. It balances the speed and stability of training. If the learning rate is too high it may diverge and if
  it is too slow training may get stuck.

## Step 4: Running LLM Inference on Fine-tune GPT 2 model

To perform LLM fine-tuning workflow in the workspace, navigate to your terminal and run this command

```bash
  python3 inference_workflow.py

```
Your output should look like the image below

  ![image of running fine tune workflow](assets/20250105_daytona_gpu_utilization_img_10.png)

## Special Considerations for Windows Users

- Window users should install drivers from Nvidia on their host machine and should not install them on the WSL distribution  
- For Windows users, If you have Nvidia GPU drivers already installed on their
  system, CUDA becomes available within WSL2. The CUDA driver installed on windows will be stubbed
  inside the WSL2, therefore users must not install any Nvidia GPU Linux driver within WSL2 to
  avoid conflicts.
- Windows users should install Docker Desktop and run it by simply opening the desktop app on their host machine instead of installing the Docker engine in WSL to avoid
issues using the Docker Daemon.

## Common Issues and Troubleshooting

**GPU cannot be accessed or GPU is unavailable**: Check the status of your GPU drivers installed on your
host machine. Verify if Nvidia Container toolkit is installation status.

**Slow Performance during fine-tuning or inference**: Verify the script is using the GPU by running `nvidia-smi`.
Optimize script by quantitizing model or inceasing batch size.

**Build Target** : The build target may need to be set to docker manually. You
can set it with `daytona target set`

## Conclusion

In conclusion, by following this guide, you've successfully set up a Daytona GPU Powered enviroment.
With a reproducible and isolated environment at your disposal, you can now experiment, test, and
develop and run different LLM workflows without affecting any of your other projects.
This setup not only streamlines your workflow but also allows you to explore new ideas and technologies.

## References

- [GitHub Repository](https://github.com/stdthoth/daytona-gpu-utilization)

- [Daytona Documentation](https://www.daytona.io/docs)

- [Nvidia CUDA Toolkit Documentation](https://docs.nvidia.com/cuda)