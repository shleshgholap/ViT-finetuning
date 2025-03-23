
# ViT-Finetuning

This project focuses on finetuning a pretrained **Vision Transformer (ViT)** model on the **Ants vs Bees dataset** (Hymenoptera dataset). The model and weights are **not imported using Hugging Face** but instead sourced from the repository: [jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch/tree/main). The necessary files, including `modeling.py`, `modeling_resnet.py`, and `configs.py`, were taken from this repository with **minor modifications**.

---

## 📂 Project Structure

```
├── hymenoptera
│   ├── train
│   │   ├── ants
│   │   ├── bees
│   ├── val
│   │   ├── ants
│   │   ├── bees
│
├── custom_dataset.py      # Custom dataset class
├── configs.py             # Configuration file for ViT
├── modeling.py            # Vision Transformer model implementation
├── modeling_resnet.py     # ResNet variant (not used in this project)
├── train.py               # Finetuning script
```

The dataset should be placed in the same directory as the rest of the code for smooth execution.

---

## 📝 Contributions

1. **Custom Dataset Class (`custom_dataset.py`)**  
   - Created a **custom PyTorch dataset class** for loading the **Hymenoptera dataset**.
   - The dataset class supports **any number of classes** stored in a similar format.
   - Provides a **label-to-index mapping** method for later use.

2. **Finetuning Script (`train.py`)**  
   - Implemented the **finetuning procedure**.
   - Used **data augmentation** with `TrivialAugmentWide`.
   - Configured the **ViT model**, loaded weights, and replaced the head layer.
   - Achieved **Val Loss: 0.2973, Val Acc: 0.9608** after 50 epochs.

---

## 📥 Dataset

The dataset is available on Kaggle:  
[🔗 Hymenoptera Dataset](https://www.kaggle.com/datasets/thedatasith/hymenoptera)

Ensure the dataset is structured as follows:

```
hymenoptera/
│── train/
│   ├── ants/
│   ├── bees/
│── val/
│   ├── ants/
│   ├── bees/
```

---

## 📦 Model Loading and Configuration

Unlike Hugging Face, the model **configuration and weights** need to be manually loaded:

1. Define the **ViT configuration** from `configs.py`:

    ```python
    config = get_b32_config()
    ```

2. Download the **pretrained weights** from Google's official ViT checkpoint repository:  
   🔗 [Google ViT Model Checkpoints](https://console.cloud.google.com/storage/vit_models/)

3. The weights are stored in a **JAX format (`.npz` file)**. To load them into PyTorch:

    ```python
    model = VisionTransformer(config).to(device)
    model.load_from(weights)
    ```

⚠ **Note:**  
- `load_from()` is used instead of the usual `load_state_dict()` because JAX-style weight keys differ from PyTorch/Hugging Face weight formats.
- A minor fix (`pjoin = lambda *args: "/".join(args)`) was introduced to handle Windows path inconsistencies.

---

## 🏋️ Training Setup

### 📌 Data Augmentation

For training, **data augmentation** is applied using `torchvision.transforms`:

```python
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

### 🏗️ Model Modification

To finetune the ViT model:
- The **original head layer** is removed.
- A **new linear layer** is added for **binary classification (Ants vs Bees)**.
- The **rest of the model parameters** are frozen.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model configuration
config = get_b32_config()
model = VisionTransformer(config).to(device)
model.load_from(weights)

# Freeze all model parameters
for param in model.parameters():
    param.requires_grad = False

# Modify the head for classification
in_features_head = model.head.in_features
model.head = nn.Linear(in_features=in_features_head, out_features=num_classes).to(device)
```

### 🎯 Loss Function & Optimizer

The model is trained using **CrossEntropyLoss** and **Adam optimizer**:

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)
```

---

## 🏆 Results

After finetuning for **50 epochs**, the model achieved:

✅ **Validation Loss:** `0.2973`  
✅ **Validation Accuracy:** `96.08%`

---

## 🚀 Running the Training

1️⃣ **Prepare the dataset** (Download from Kaggle and structure it as described).  
2️⃣ **Download the model weights** from [Google ViT Checkpoints](https://console.cloud.google.com/storage/vit_models/).  
3️⃣ **Run the training script:**

```bash
python train.py
```

---

## 📌 Future Improvements

- Extend to a **multi-class dataset** using the same dataset class.
- Experiment with **different ViT architectures** (`ViT-L`, `ViT-H`).
- Use **larger datasets** to improve generalization.

---

## 🤝 Acknowledgments

- [jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch/tree/main) for the ViT implementation.
- [Kaggle Hymenoptera Dataset](https://www.kaggle.com/datasets/thedatasith/hymenoptera).
- **Google AI** for providing ViT pretrained weights.

---
