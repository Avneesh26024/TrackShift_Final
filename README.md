# Visual Difference Engine

A **general-purpose semantic visual comparison system** designed to detect, localize, and classify meaningful changes across time‑series images.

Built originally for **F1-inspired use cases**, but engineered to generalize across manufacturing, infrastructure, retail, compliance monitoring, and security.

---

## 🚀 Overview

Modern environments (like F1 racing) generate massive amounts of visual data from pit stops, trackside cameras, onboard footage, and broadcast feeds. Detecting important changes — missing tools, debris on track, aerodynamic damage, or sponsor logo issues — must happen **in real time**, with high reliability.

Traditional image comparison tools fail:

* **SSIM / pixel-based methods** get confused by shadows, lighting, motion.
* **YOLO-style detectors** need thousands of labeled examples per class.
* **YOLO only sees what it is trained on** — unseen objects are ignored.
* **LLM-based vision approaches** are too slow for real-time frame rates.

The **Visual Difference Engine** solves this by focusing on **semantic change detection** (not pixel changes), requiring **zero training**, and running with **low latency** on live feeds.

---

## 🎯 Key Features

### **Semantic Change Detection**

Understands *meaningful* changes, not just pixel differences.

### **Patch-Level Localization**

Divides images into windows and highlights exactly where changes occur.

### **Zero-/Few-Shot Capability**

Works without labeled datasets. No retraining needed.

### **Defect Classification**

Uses text prompts to label change types — e.g., `"missing tool"`, `"new object"`, `"aero damage"`.

### **Temporal Tracking**

Keeps track of changes over time to detect patterns or gradual degradation.

### **Memory Bank Acceleration**

Caches embeddings from previous frames to drastically reduce latency.

### **General-Purpose by Design**

One engine, multiple industries & use cases.

---

## 🔧 Technical Architecture

### **1. SigLIP2 Encoder Backbone**

* Stronger and more robust than CLIP
* Trained on massive multimodal datasets
* Provides rich semantic embeddings across layers

### **2. WinCLIP-Style Window Engine**

* Splits the image into overlapping windows
* Extracts multi-layer features
* Computes similarity against text prompts or reference patches

### **3. Memory Bank**

* Stores embeddings of previous frames
* Recomputes only changed areas
* Enables near real-time performance

### **4. Semantic Scoring**

Uses cosine similarity thresholds to rank windows from "normal" to "high-change".

### **5. Optional Adapter Heads (Future)**

* Lightweight segmentation refinement
* Pixel-accurate masks for forensic inspection

---

## 🧠 Why It Beats Traditional Methods

### **Compared to SSIM / Pixel Diff**

* Ours understands context → shadows, reflections, lighting won't cause false alarms.

### **Compared to YOLO**

* No huge labeled datasets needed
* Works on unseen objects
* Detects micro-changes YOLO ignores
* No retraining cycles

### **Compared to LLM Vision Models**

* LLMs are too slow for F1-level real-time use
* Our engine is optimized for speed + precision

---

## 🏎️ F1 Use Cases

* Detect **missing/misplaced pit tools**
* Identify **small debris** on track
* Spot **aero damage** or part loss
* Monitor **sponsor logo damage**
* Compare **car condition changes** between laps
* Support **broadcast analysis** with instant highlights

---

## 🏭 Multi-Industry Use Cases

### **Manufacturing**

* Missing components
* Misalignment
* Scratch or surface defect detection

### **Infrastructure**

* Crack progression
* Rust formation
* Water leakage detection

### **Retail & Brand Compliance**

* Packaging consistency
* Shelf layout checks
* Logo and branding visibility

### **Security**

* Unattended objects
* Unexpected intrusions

---

## 📈 Future Roadmap

* **Ultra-low latency edge mode** (Jetson, Orin)
* **High-precision segmentation adapters**
* **Temporal reasoning** for predictive alerts
* **Multi-camera fusion** to reduce blindspots
* **Auto-tagging**: label anomaly type automatically

---

## 📦 Installation (Coming Soon)

```
pip install visual-difference-engine
```

---

## 🖥️ Usage (Pseudo-Code)

```python
from engine import VDEngine

model = VDEngine(backbone="siglip2")

result = model.compare(frame_prev, frame_curr,
                       prompts=["missing tool", "new object", "damage"])

print(result.changed_regions)
print(result.defect_labels)
```

---

## 📊 Benchmark Summary

* **90%+ patch-level accuracy** on PCB change dataset (baseline WinCLIP)
* Improved stability using SigLIP2
* Pixel-level segmentation improving with adapters

---

## 👥 Team

* **Aditya Channa** — ML/Backend Developer
  [LinkedIn](https://www.linkedin.com/in/adityachanna/) | Email: [adityachannadelhi@gmail.com](mailto:adityachannadelhi@gmail.com)
* **Jastej Singh** — Full Stack Developer
  [LinkedIn](https://www.linkedin.com/in/jastej-singh-27940a290/) | Email: [jastej28.singh@gmail.com](mailto:jastej28.singh@gmail.com)
* **Avneesh** — ML/Data Engineer
  [LinkedIn](https://www.linkedin.com/in/avneesh-avneesh-99a40b284/) | Email: [avneesh26024@gmail.com](mailto:avneesh26024@gmail.com)

---

> **A single general-purpose engine that understands what changed, where it changed, and what it means — without training.**
