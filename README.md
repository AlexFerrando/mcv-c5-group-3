# C5 Project
Team 3 repository of the [C5 - Visual Recognition](https://mcv.uab.cat/c5-visual-recognition/) subject of the Master in Computer Vision at UAB.

*   **Environment Setup:** We used Conda to manage the environment. To replicate our environment, follow these steps:

    1.  **Install Conda:** If you don't already have Conda installed, download and install Miniconda or Anaconda.

    2.  **Create the Environment:**  Navigate to the root directory of this project (where `environment.yml` is located) in your terminal and run:

        ```bash
        conda env create -f environment.yml
        ```

    3.  **Activate the Environment:**  Activate the newly created environment:

        ```bash
        conda activate <env_name>
        ```

    This `environment.yml` file contains all the necessary dependencies.

* The code is organized as follows:
    - Each model code is divided into its own folder [detectron2](/mcv-c5-group-3/detectron2/), [huggingface](/mcv-c5-group-3/huggingface/) and [ultralytics_](/mcv-c5-group-3/ultralytics_/).

## Week 1
Week 1 focused on object detection.  This section details our efforts in inference, training, and validation of three prominent deep learning models:

*   **Faster R-CNN**
*   **DeTR**
*   **YOLO**


The following tasks were addressed using the KITTI-MOTS dataset and as domain shift, CPPE-5.


### Task C: Run inference with pre-trained Faster R-CNN, DeTR and YOLO on KITTI-MOTS dataset.

We ran inference using pre-trained Faster R-CNN, DeTR, and YOLO models on the KITTI-MOTS dataset.  This involved setting up the environments for each model, loading the pre-trained weights, and processing the KITTI-MOTS data to generate object detection predictions.
**Implementation Details:**

*   **Environment Setup:** We used as suggested by the teachers Hugging Face for DeTR, Ultralytics for YOLO and Detectron2 for Faster R-CNN.
*   **Model Loading:**  Pre-trained weights were downloaded from the Hugging Face Hub, downloaded from ultralytics of from the already built-in Detetron2 methods, here and in the following tasks.
*   **Inference Execution:**  Inference was performed using scripts located in [`inference.py`](https://github.com/AlexFerrando/mcv-c5-group-3/blob/main/huggingface/inference.py) for DeTR, [`inference.py`](https://github.com/AlexFerrando/mcv-c5-group-3/blob/main/ultralytics_/inference.py) for YOLO11n and [`task_c.py`](https://github.com/AlexFerrando/mcv-c5-group-3/blob/main/detectron2/task_c.py). 



### Task D: Evaluate pre-trained Faster R-CNN, DeTR and YOLOv(>8) on KITTIMOTS dataset

### Task E: Fine-tune Faster R-CNN, DeTR and YOLO on KITTI-MOTS.

### Task F:  Fine-tune either Faster R-CNN and DeTR on Different Dataset

## Team 5
- Alex Ferrando ([email](mailto:alexferrando15@gmail.com)) 
- Pol Rosinés ([email](mailto:polrosines@gmail.com))
- Arnau Barrera ([email](mailto:arnau6baroy@gmail.com))
- Oriol Marín ([email](mailto:oriolmarin18@gmail.com))
