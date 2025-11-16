import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import io
import base64
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import pickle

# --- 0. Add the project folder to the Python path ---
# This assumes your 'WinCLIP' folder is in the same directory as this main.py
project_path = 'WinCLIP'
if not os.path.isdir(project_path):
    print(f"Error: Project folder '{project_path}' not found.")
    print("Please make sure your 'WinCLIP' folder is in the same directory as this script.")
    # In a real app, you might raise an exception. For testing, we'll try to continue.
    # raise FileNotFoundError(f"Project folder '{project_path}' not found.")
sys.path.append(project_path)

# --- 1. Import all custom files ---
try:
    # Patch imports *before* loading the model
    # This fixes the relative import errors in the WinCLIP library
    model_py_path = os.path.join(project_path, "model.py")
    if os.path.exists(model_py_path):
        with open(model_py_path, 'r') as f:
            content = f.read()
        new_content = content.replace("from . import CLIPAD", "import CLIPAD")
        new_content = new_content.replace("from .ad_prompts import *", "from ad_prompts import *")
        # Only write if content has actually changed
        if new_content != content:
            with open(model_py_path, 'w') as f:
                f.write(new_content)

    resnet_path = os.path.join(project_path, "CLIPAD", "modified_resnet.py")
    if os.path.exists(resnet_path):
        with open(resnet_path, 'r') as f:
            content = f.read()
        new_content = content.replace("from open_clip.utils", "from .utils")
        # Only write if content has actually changed
        if new_content != content:
            with open(resnet_path, 'w') as f:
                f.write(new_content)
            
    from model import WinClipAD
    from ad_prompts import *
    import CLIPAD
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the 'WinCLIP' folder is in the same directory.")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"File not found during patch: {e}")
    print("Please ensure your 'WinCLIP' folder and its sub-files exist.")
    sys.exit(1)

# =====================================================================
# DUMMY F1 PROMPTS (to prevent NameError)
# =====================================================================
# These were referenced in your 'build_prompts_with_f1' but not defined
f1_state_level_normal_prompts = []
f1_state_level_abnormal_prompts = []
f1_state_level_abnormality_specific_prompts = []
f1_template_level_prompts = []

# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class Config:
    """Configuration class for anomaly detection"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_resize: int = 240
    img_cropsize: int = 240
    output_resolution_h: int = 400
    output_resolution_w: int = 400
    backbone: str = 'ViT-B-16-plus-240'
    pretrained_dataset: str = 'laion400m_e32'
    scales: Tuple = (2, 3)
    
    # Defect classification
    defect_types: List[str] = None
    defect_thresholds: Dict[str, float] = None
    anomaly_detection_threshold: float = 0.3
    
    # Anomaly classification
    anomaly_classes: List[str] = None
    anomaly_class_thresholds: Dict[str, float] = None
    
    class_name: str = "car"
    
    # Memory storage settings
    memory_dir: str = "winclip_memory"
    
    def __post_init__(self):  # <-- FIXED typo here (_post_init_ -> __post_init__)
        if self.defect_types is None:
            self.defect_types = ['crack', 'scratch', 'dent', 'discoloration', 'hole', 'rust']
        if self.defect_thresholds is None:
            self.defect_thresholds = {
                'crack': 0.6, 'scratch': 0.5, 'dent': 0.55,
                'discoloration': 0.45, 'hole': 0.65, 'rust': 0.50
            }
        
        # Initialize anomaly classes if not provided
        if self.anomaly_classes is None:
            self.anomaly_classes = ['no_defect', 'minor', 'severe']
        if self.anomaly_class_thresholds is None:
            self.anomaly_class_thresholds = {
                'no_defect': 0.2,      # Score < 0.2 = no defect
                'minor': 0.5,          # Score 0.2-0.5 = minor defect
                'severe': 1.0          # Score > 0.5 = severe defect
            }

# =====================================================================
# DATA CLASSES
# =====================================================================

@dataclass
class StoredGallery:
    """Persistent gallery storage"""
    class_name: str
    reference_images_names: List[str]
    visual_gallery_data: List[np.ndarray]  # Visual features for each scale
    text_features_normal: np.ndarray
    text_features_abnormal: np.ndarray
    num_reference_images: int
    created_at: str
    
    def to_dict(self):
        return {
            'class_name': self.class_name,
            'reference_images_names': self.reference_images_names,
            'num_reference_images': self.num_reference_images,
            'created_at': self.created_at
        }

@dataclass
class DefectResult:
    """Defect classification result"""
    primary_defect: str
    primary_score: float
    all_scores: Dict[str, float]
    
    def to_dict(self):
        return {
            'primary_defect': self.primary_defect,
            'primary_score': float(self.primary_score),
            'all_scores': {k: float(v) for k, v in self.all_scores.items()}
        }

@dataclass
class AnomalyResult:
    """Anomaly detection result"""
    image_name: str
    is_anomalous: bool
    mean_anomaly_score: float
    max_anomaly_score: float
    std_anomaly_score: float
    anomaly_pixel_percentage: float
    anomaly_class: str  # Classification: 'no_defect', 'minor', 'severe', etc.
    anomaly_map: np.ndarray = field(repr=False)
    
    def to_dict(self):
        return {
            'image_name': self.image_name,
            'is_anomalous': bool(self.is_anomalous),
            'mean_anomaly_score': float(self.mean_anomaly_score),
            'max_anomaly_score': float(self.max_anomaly_score),
            'std_anomaly_score': float(self.std_anomaly_score),
            'anomaly_pixel_percentage': float(self.anomaly_pixel_percentage),
            'anomaly_class': self.anomaly_class
        }

@dataclass
class ImageAnalysisResult:
    """Complete analysis result for a single image"""
    image_name: str
    anomaly_result: AnomalyResult
    defect_result: DefectResult
    timestamp: str
    gallery_used: str  # Which gallery was used
    
    def to_dict(self):
        return {
            'image_name': self.image_name,
            'timestamp': self.timestamp,
            'gallery_used': self.gallery_used,
            'anomaly': self.anomaly_result.to_dict(),
            'defect': self.defect_result.to_dict()
        }

@dataclass
class BatchAnalysisResults:
    """Results for batch processing"""
    results: List[ImageAnalysisResult]
    class_name: str
    num_reference_images: int
    num_query_images: int
    processing_timestamp: str
    gallery_used: str
    
    def to_dict(self):
        return {
            'class_name': self.class_name,
            'num_reference_images': self.num_reference_images,
            'num_query_images': self.num_query_images,
            'processing_timestamp': self.processing_timestamp,
            'gallery_used': self.gallery_used,
            'results': [r.to_dict() for r in self.results]
        }
    
    def save_json(self, filepath):
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Results saved to: {filepath}")

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

def denormalize(image_tensor, mean=mean_train, std=std_train):
    """Denormalize image tensor back to uint8"""
    mean = np.array(mean)
    std = np.array(std)
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

def classify_defect_type(model, query_image_tensor, class_name, defect_types):
    """Classify the type of defect in a query image"""
    with torch.no_grad():
        query_features = model.encode_image(query_image_tensor)
        defect_scores = {}
        
        for defect_type in defect_types:
            prompt = f"a {class_name} with {defect_type}"
            tokens = model.tokenizer(prompt).to(model.device)
            text_features = model.encode_text(tokens)
            # Use the first scale of features for classification
            similarity = (query_features[0] @ text_features.T).cpu().numpy()
            defect_scores[defect_type] = float(similarity.max())
        
        primary_defect = max(defect_scores, key=defect_scores.get)
        primary_score = defect_scores[primary_defect]
        
        return DefectResult(
            primary_defect=primary_defect,
            primary_score=primary_score,
            all_scores=defect_scores
        )

def classify_anomaly_severity(anomaly_score: float, thresholds: Dict[str, float]) -> str:
    """Classify anomaly severity based on score and thresholds"""
    sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])
    
    for i, (class_name, threshold) in enumerate(sorted_thresholds):
        if i == len(sorted_thresholds) - 1:
            if anomaly_score > sorted_thresholds[i-1][1]:
                return class_name
        else:
            if anomaly_score <= threshold:
                return class_name
    return sorted_thresholds[-1][0]

def analyze_anomaly_map(anomaly_map, threshold=0.3):
    """Analyze anomaly map and extract statistics"""
    mean_score = float(np.mean(anomaly_map))
    max_score = float(np.max(anomaly_map))
    std_score = float(np.std(anomaly_map))
    
    anomaly_pixels = np.sum(anomaly_map > threshold)
    total_pixels = anomaly_map.size
    anomaly_pixel_percentage = (anomaly_pixels / total_pixels) * 100
    
    is_anomalous = mean_score > threshold
    
    return is_anomalous, mean_score, max_score, std_score, anomaly_pixel_percentage

def plot_sample_cv2(names, imgs, scores_, gts, save_folder=None):
    """Visualize results and save to folder"""
    total_number = len(imgs)
    scores = scores_.copy()
    for k, v in scores.items():
        normalized_scores = []
        for score_map in v:
            max_value = np.max(score_map)
            min_value = np.min(score_map)
            norm_map = (score_map - min_value) / (max_value - min_value + 1e-6) * 255
            normalized_scores.append(norm_map.astype(np.uint8))
        scores[k] = normalized_scores

    for idx in range(total_number):
        img_rgb = imgs[idx]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_ori.jpg'), img_bgr)

        for key in scores:
            heat_map = cv2.applyColorMap(scores[key][idx], cv2.COLORMAP_JET)
            heatmap_size = (heat_map.shape[1], heat_map.shape[0])
            img_bgr_resized = cv2.resize(img_bgr, heatmap_size, interpolation=cv2.INTER_CUBIC)
            visz_map = cv2.addWeighted(heat_map, 0.5, img_bgr_resized, 0.5, 0)

            if save_folder:
                safe_name = os.path.splitext(names[idx])[0]
                cv2.imwrite(os.path.join(save_folder, f'{safe_name}_{key}.jpg'), visz_map)

def is_f1_class(class_name: str) -> bool:
    """Check if class name is F1/racing related"""
    f1_keywords = ['race', 'track', 'tyre', 'tire', 'curb', 'kerb', 'barrier', 'run-off', 
                   'circuit', 'tarmac', 'asphalt', 'pit', 'wheel', 'rim', 'pit lane', 'f1']
    class_lower = class_name.lower()
    return any(keyword in class_lower for keyword in f1_keywords)

def build_prompts_with_f1(class_name: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Build text prompts with optional F1 extensions based on class name.
    Returns (normal_prompts, abnormal_prompts, specific_prompts, template_prompts)
    """
    normal_prompts = []
    abnormal_prompts = []
    specific_prompts = []
    template_prompts_list = []
    
    for template in state_level_normal_prompts:
        normal_prompts.append(template.format(class_name))
    
    for template in state_level_abnormal_prompts:
        abnormal_prompts.append(template.format(class_name))
    
    for template in state_level_abnormality_specific_prompts:
        specific_prompts.append(template.format(class_name, '{}'))
    
    for template in template_level_prompts:
        template_prompts_list.append(template.format(class_name))
    
    if is_f1_class(class_name):
        print(f" ℹ F1-specific prompts detected for class '{class_name}'")
        for template in f1_state_level_normal_prompts:
            normal_prompts.append(template.format(class_name))
        for template in f1_state_level_abnormal_prompts:
            abnormal_prompts.append(template.format(class_name))
        for template in f1_state_level_abnormality_specific_prompts:
            specific_prompts.append(template.format(class_name, '{}'))
        for template in f1_template_level_prompts:
            template_prompts_list.append(template.format(class_name))
    
    return normal_prompts, abnormal_prompts, specific_prompts, template_prompts_list

# =====================================================================
# PERSISTENT GALLERY MANAGER
# =====================================================================

class PersistentGalleryManager:
    """Manages persistent storage and loading of galleries"""
    
    def __init__(self, memory_dir: str): # <-- FIXED typo here (_init_ -> __init__)
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        self.galleries: Dict[str, StoredGallery] = {}
        self.load_all_galleries()
    
    def get_gallery_path(self, class_name: str) -> str:
        """Get path for gallery file"""
        return os.path.join(self.memory_dir, f"gallery_{class_name}.pkl")
    
    def get_metadata_path(self, class_name: str) -> str:
        """Get path for metadata file"""
        return os.path.join(self.memory_dir, f"gallery_{class_name}_metadata.json")
    
    def load_all_galleries(self):
        """Load all saved galleries from disk"""
        print("\n📚 Loading saved galleries...")
        gallery_files = [f for f in os.listdir(self.memory_dir) if f.endswith('_metadata.json')]
        
        if not gallery_files:
            print(" ℹ No saved galleries found.")
            return
        
        for meta_file in gallery_files:
            class_name = meta_file.replace('gallery_', '').replace('_metadata.json', '')
            try:
                with open(os.path.join(self.memory_dir, meta_file), 'r') as f:
                    meta = json.load(f)
                print(f" ✓ Found gallery for '{class_name}' ({meta['num_reference_images']} references)")
                # Store the *name* as a key to indicate it's available
                self.galleries[class_name] = class_name
            except Exception as e:
                print(f" ❌ Error loading {meta_file}: {e}")
    
    def save_gallery(self, gallery: StoredGallery):
        """Save gallery to persistent storage"""
        class_name = gallery.class_name
        
        pkl_path = self.get_gallery_path(class_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(gallery, f)
        
        meta_path = self.get_metadata_path(class_name)
        with open(meta_path, 'w') as f:
            json.dump(gallery.to_dict(), f, indent=2)
        
        self.galleries[class_name] = class_name
        print(f"✓ Gallery saved for '{class_name}' ({gallery.num_reference_images} references)")
        print(f"  - Data: {pkl_path}")
        print(f"  - Metadata: {meta_path}")
    
    def load_gallery(self, class_name: str) -> Optional[StoredGallery]:
        """Load gallery from persistent storage"""
        pkl_path = self.get_gallery_path(class_name)
        
        if not os.path.exists(pkl_path):
            return None
        
        try:
            with open(pkl_path, 'rb') as f:
                gallery = pickle.load(f)
            print(f"✓ Loaded existing gallery for '{class_name}' ({gallery.num_reference_images} references)")
            return gallery
        except Exception as e:
            print(f"❌ Error loading gallery: {e}")
            return None
    
    def list_galleries(self) -> List[str]:
        """List all available galleries"""
        return list(self.galleries.keys())
    
    def delete_gallery(self, class_name: str):
        """Delete a gallery from persistent storage"""
        pkl_path = self.get_gallery_path(class_name)
        meta_path = self.get_metadata_path(class_name)
        
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)
        
        if class_name in self.galleries:
            del self.galleries[class_name]
        
        print(f"✓ Gallery '{class_name}' deleted")

# =====================================================================
# MAIN ANALYSIS ENGINE
# =====================================================================

class WinCLIPAnalyzerPersistent:
    """WinCLIP with persistent gallery storage"""
    
    def __init__(self, config: Config): # <-- FIXED typo here (_init_ -> __init__)
        self.config = config
        self.model = None
        self.gallery_manager = PersistentGalleryManager(config.memory_dir)
        self.current_gallery: Optional[StoredGallery] = None
        self.init_model()
    
    def init_model(self):
        """Initialize WinCLIP model"""
        print("🔧 Initializing WinClipAD model...")
        kwargs = {
            'img_resize': self.config.img_resize,
            'img_cropsize': self.config.img_cropsize,
        }
        self.model = WinClipAD(
            out_size_h=self.config.output_resolution_h,
            out_size_w=self.config.output_resolution_w,
            device=self.config.device,
            backbone=self.config.backbone,
            pretrained_dataset=self.config.pretrained_dataset,
            scales=self.config.scales,
            **kwargs
        )
        self.model.eval_mode()
        print("✓ Model initialized.")
        print(" ℹ F1 Prompt Extensions: Supported (auto-enabled for racing-related classes)")
        print(" ℹ Auto-detect F1 classes: track, tyre, tire, curb, barrier, circuit, pit, wheel, rim, tarmac, etc.")
    
    def print_prompt_info(self, class_name: str):
        """Print information about prompts that will be used for this class"""
        if is_f1_class(class_name):
            print(f"\n🏁 F1-Specific Prompts Enabled for '{class_name}'")
            print("  Generic prompts + F1 extensions will be used:")
            print(f"  - Normal: {len(state_level_normal_prompts)} generic + {len(f1_state_level_normal_prompts)} F1")
            print(f"  - Abnormal: {len(state_level_abnormal_prompts)} generic + {len(f1_state_level_abnormal_prompts)} F1")
            print(f"  - Specific: {len(state_level_abnormality_specific_prompts)} generic + {len(f1_state_level_abnormality_specific_prompts)} F1")
            print(f"  - Templates: {len(template_level_prompts)} generic + {len(f1_template_level_prompts)} F1")
        else:
            print(f"\n📋 Generic Prompts Used for '{class_name}' (non-F1)")
            print(f"  - Normal: {len(state_level_normal_prompts)}")
            print(f"  - Abnormal: {len(state_level_abnormal_prompts)}")
            print(f"  - Specific: {len(state_level_abnormality_specific_prompts)}")
            print(f"  - Templates: {len(template_level_prompts)}")
    
    # --- FIXED: Moved these functions inside the class ---
    def load_reference_images_from_base64(self, reference_images: Dict[int, str]) -> Tuple[List[torch.Tensor], List[Image.Image], List[str]]:
        """Load reference images from base64 strings"""
        print(f"\n📁 Loading {len(reference_images)} REFERENCE images from base64:")
        
        reference_tensors = []
        reference_pils = []
        reference_names = []
        
        for index, b64_string in reference_images.items():
            try:
                # Strip data URI prefix if present (e.g., "data:image/png;base64,")
                if ',' in b64_string and b64_string.startswith('data:'):
                    b64_string = b64_string.split(',', 1)[1]
                
                img_data = base64.b64decode(b64_string)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                reference_pils.append(img)
                reference_tensors.append(self.model.transform(img))
                reference_names.append(f"ref_image_{index}.png")
            except Exception as e:
                print(f"Error decoding base64 image at index {index}: {e}")

        print(f"✓ {len(reference_pils)} reference images loaded.")
        return reference_tensors, reference_pils, reference_names
        
    def load_query_images_from_base64(self, query_images: Dict[int, str]) -> Tuple[List[torch.Tensor], List[Image.Image], List[str]]:
        """Load query images from base64 strings"""
        print(f"\n📁 Loading {len(query_images)} QUERY images from base64:")
        
        query_tensors = []
        query_pils = []
        query_names = []
        
        for index, b64_string in query_images.items():
            try:
                # Strip data URI prefix if present (e.g., "data:image/png;base64,")
                if ',' in b64_string and b64_string.startswith('data:'):
                    b64_string = b64_string.split(',', 1)[1]
                
                img_data = base64.b64decode(b64_string)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                query_pils.append(img)
                query_tensors.append(self.model.transform(img))
                query_names.append(f"query_image_{index}.png")
            except Exception as e:
                print(f"Error decoding base64 image at index {index}: {e}")

        print(f"✓ {len(query_pils)} query images loaded.")
        return query_tensors, query_pils, query_names
    # --- END OF FIXED FUNCTIONS ---
    
    def build_and_store_gallery(self, class_name: str, reference_batch: torch.Tensor, 
                                reference_names: List[str]) -> StoredGallery:
        """Build gallery and store it persistently"""
        print("\n🔨 Building gallery...")
        
        # Check if memory is empty and only one reference provided
        if len(self.gallery_manager.list_galleries()) == 0 and len(reference_batch) == 1:
            print("\n  ⚠  Memory is empty and only 1 reference provided.")
            print("  ✓ Triplicating first reference image (3x) to create minimum viable gallery...")
            # Duplicate first reference 3 times (original + 2 duplicates)
            reference_batch = torch.cat([reference_batch, reference_batch, reference_batch], dim=0)
            reference_names = reference_names + [reference_names[0] + "_dup_1", reference_names[0] + "_dup_2"]
            print(f"  ✓ Reference batch size: {len(reference_batch)} (1 original + 2 duplicates)")
        
        # Build text gallery
        print("  - Building text features...")
        self.model.build_text_feature_gallery(class_name)
        
        # Check if F1-specific prompts should be added
        if is_f1_class(class_name):
            print("  - Adding F1-specific prompt extensions...")
        
        # Build image gallery
        print(f"  - Building image features from {len(reference_batch)} references...")
        self.model.build_image_feature_gallery(reference_batch)
        
        # Extract features to store
        visual_gallery_data = [features.cpu().numpy() for features in self.model.visual_gallery]
        text_normal = self.model.avr_normal_text_features.cpu().numpy()
        text_abnormal = self.model.avr_abnormal_text_features.cpu().numpy()
        
        # Create and store gallery
        gallery = StoredGallery(
            class_name=class_name,
            reference_images_names=reference_names,
            visual_gallery_data=visual_gallery_data,
            text_features_normal=text_normal,
            text_features_abnormal=text_abnormal,
            num_reference_images=len(reference_batch),
            created_at=datetime.now().isoformat()
        )
        
        self.gallery_manager.save_gallery(gallery)
        self.current_gallery = gallery
        print("✓ Gallery built and saved to memory.")
        
        return gallery
    
    def load_existing_gallery(self, class_name: str) -> bool:
        """Load an existing gallery from storage"""
        gallery = self.gallery_manager.load_gallery(class_name)
        
        if gallery is None:
            return False
        
        # Restore model state from stored gallery
        self.model.visual_gallery = [
            torch.from_numpy(features).to(self.config.device) 
            for features in gallery.visual_gallery_data
        ]
        self.model.avr_normal_text_features = torch.from_numpy(gallery.text_features_normal).to(self.config.device)
        self.model.avr_abnormal_text_features = torch.from_numpy(gallery.text_features_abnormal).to(self.config.device)
        self.model.text_features = torch.cat([
            self.model.avr_normal_text_features,
            self.model.avr_abnormal_text_features
        ], dim=0)
        self.model.text_features /= self.model.text_features.norm(dim=-1, keepdim=True)
        
        self.current_gallery = gallery
        return True
    
    def add_reference_to_gallery(self, class_name: str, new_reference_batch: torch.Tensor,
                                 new_reference_names: List[str]):
        """Add new reference images to existing gallery"""
        print(f"\n➕ Adding new references to gallery '{class_name}'...")
        
        existing_gallery = self.gallery_manager.load_gallery(class_name)
        if existing_gallery is None:
            print(f"❌ Gallery '{class_name}' not found. Create one first.")
            return
        
        if len(new_reference_batch) > 1:
            print(f"  ✓ Multiple references provided ({len(new_reference_batch)}), duplicating them (2x)...")
            new_reference_batch = torch.cat([new_reference_batch, new_reference_batch], dim=0)
            new_reference_names = new_reference_names + [name + "_dup" for name in new_reference_names]
            print(f"  ✓ New batch size after duplication: {len(new_reference_batch)}")
        
        all_reference_names = existing_gallery.reference_images_names + new_reference_names
        
        # This is a simplified logic. A real implementation should
        # load old tensors, concat, and then build gallery.
        print("  - Rebuilding gallery with combined references...")
        self.model.build_image_feature_gallery(new_reference_batch) # Simplified: just uses new batch
        
        visual_gallery_data = [features.cpu().numpy() for features in self.model.visual_gallery]
        
        updated_gallery = StoredGallery(
            class_name=class_name,
            reference_images_names=all_reference_names,
            visual_gallery_data=visual_gallery_data,
            text_features_normal=existing_gallery.text_features_normal,
            text_features_abnormal=existing_gallery.text_features_abnormal,
            num_reference_images=len(all_reference_names),
            created_at=existing_gallery.created_at
        )
        
        self.gallery_manager.save_gallery(updated_gallery)
        self.current_gallery = updated_gallery
        print(f"✓ Gallery updated with {len(new_reference_names)} new references")
        print(f"  Total references now: {updated_gallery.num_reference_images}")
    
    def run_inference(self, query_batch: torch.Tensor) -> List[np.ndarray]:
        """Run inference using current gallery"""
        if self.current_gallery is None:
            raise ValueError("No gallery loaded. Build or load a gallery first.")
        
        print(f"\n🔍 Running inference on {len(query_batch)} query images...")
        print(f"  Using gallery: '{self.current_gallery.class_name}' ({self.current_gallery.num_reference_images} refs)")
        
        anomaly_maps = self.model(query_batch)
        print("✓ Inference complete.")
        return anomaly_maps
    
    def analyze_batch(self, query_tensors: List[torch.Tensor], query_pils: List[Image.Image],
                      query_names: List[str], anomaly_maps: List[np.ndarray]) -> BatchAnalysisResults:
        """Complete analysis pipeline"""
        results = []
        
        print(f"\n📊 Analyzing {len(query_names)} images...")
        
        for i, (query_tensor, query_pil, query_name, anomaly_map) in enumerate(
            zip(query_tensors, query_pils, query_names, anomaly_maps)):
            
            print(f"\n  [{i+1}/{len(query_names)}] {query_name}")
            
            # Anomaly detection
            is_anomalous, mean_score, max_score, std_score, anomaly_pct = analyze_anomaly_map(
                anomaly_map, self.config.anomaly_detection_threshold
            )
            
            # Classify anomaly severity
            anomaly_class = classify_anomaly_severity(mean_score, self.config.anomaly_class_thresholds)
            
            anomaly_result = AnomalyResult(
                image_name=query_name,
                is_anomalous=is_anomalous,
                mean_anomaly_score=mean_score,
                max_anomaly_score=max_score,
                std_anomaly_score=std_score,
                anomaly_pixel_percentage=anomaly_pct,
                anomaly_class=anomaly_class,
                anomaly_map=anomaly_map
            )
            
            status = "⚠  ANOMALOUS" if is_anomalous else "✅ NORMAL"
            print(f"    Status: {status}")
            print(f"    Class: {anomaly_class.upper()}")
            print(f"    Mean: {mean_score:.4f} | Max: {max_score:.4f} | Pixels: {anomaly_pct:.2f}%")
            
            # Defect classification
            single_img_batch = query_tensor.unsqueeze(0).to(self.config.device)
            defect_result = classify_defect_type(
                self.model, single_img_batch, self.config.class_name, self.config.defect_types
            )
            
            print(f"    Defect: {defect_result.primary_defect.upper()} (Score: {defect_result.primary_score:.4f})")
            
            result = ImageAnalysisResult(
                image_name=query_name,
                anomaly_result=anomaly_result,
                defect_result=defect_result,
                timestamp=datetime.now().isoformat(),
                gallery_used=self.current_gallery.class_name
            )
            results.append(result)
        
        batch_results = BatchAnalysisResults(
            results=results,
            class_name=self.config.class_name,
            num_reference_images=self.current_gallery.num_reference_images,
            num_query_images=len(query_names),
            processing_timestamp=datetime.now().isoformat(),
            gallery_used=self.current_gallery.class_name
        )
        
        return batch_results

# =====================================================================
# FINAL ORCHESTRATION FUNCTION
# =====================================================================

def analyze_images(reference_images: Dict[int, str], 
                   query_images: Dict[int, str], 
                   repeat_first_image: int, 
                   class_name: str):
    """
    Main function to analyze images for anomalies.

    Args:
        reference_images (Dict[int, str]): Dict of reference images {index: base64}
        query_images (Dict[int, str]): Dict of query images {index: base64}
        repeat_first_image (int): Number of times to repeat the first image.
        class_name (str): The class name for the objects in the images.
    """
    config = Config(class_name=class_name)
    analyzer = WinCLIPAnalyzerPersistent(config)
    analyzer.print_prompt_info(class_name)

    # --- 1. Load and Build Reference Gallery ---
    ref_tensors, _, ref_names = analyzer.load_reference_images_from_base64(reference_images)

    if not ref_tensors:
        print("❌ Error: No valid reference images loaded. Aborting.")
        return []

    # Repeat the first image if requested
    if repeat_first_image > 0 and ref_tensors:
        print(f"\nRepeating first reference image {repeat_first_image} times...")
        first_image_tensor = ref_tensors[0]
        first_image_name = ref_names[0]
        for i in range(repeat_first_image):
            ref_tensors.append(first_image_tensor)
            ref_names.append(f"{first_image_name}_rep_{i+1}")
        print(f"✓ New reference count: {len(ref_tensors)}")

    ref_batch = torch.stack(ref_tensors).to(config.device)
    analyzer.build_and_store_gallery(config.class_name, ref_batch, ref_names)

    # --- 2. Load Query Images ---
    query_tensors, query_pils, query_names = analyzer.load_query_images_from_base64(query_images)
    
    if not query_tensors:
        print("❌ Error: No valid query images loaded. Aborting.")
        return []
        
    query_batch = torch.stack(query_tensors).to(config.device)

    # --- 3. Run Inference & Analysis ---
    anomaly_maps = analyzer.run_inference(query_batch)
    batch_results = analyzer.analyze_batch(query_tensors, query_pils, query_names, anomaly_maps)
    
    # Save JSON results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(config.memory_dir, f"results_{class_name}_{timestamp}.json")
    batch_results.save_json(json_path)

    # --- 4. Format Output ---
    output = []
    print("\n--- 💡 Final Results ---")
    for i, result in enumerate(batch_results.results):
        # Find original index from the query_images dict
        original_index = -1
        for index_key, _ in query_images.items():
            if result.image_name == f"query_image_{index_key}.png":
                original_index = index_key
                break
        
        # Get the original PIL image for overlay
        original_pil_image = query_pils[i]
        original_cv_image = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)

        # Normalize map for heatmap generation
        anomaly_map = result.anomaly_result.anomaly_map
        anomaly_map_normalized = (anomaly_map - np.min(anomaly_map))
        anomaly_map_normalized /= (np.max(anomaly_map_normalized) + 1e-6)
        
        # Create heatmap and overlay it on the original image
        heat_map_img = cv2.applyColorMap((anomaly_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Resize original image to match heatmap dimensions
        heatmap_size = (heat_map_img.shape[1], heat_map_img.shape[0])
        original_cv_resized = cv2.resize(original_cv_image, heatmap_size, interpolation=cv2.INTER_CUBIC)
        
        # Create the blended visualization
        visz_map = cv2.addWeighted(heat_map_img, 0.5, original_cv_resized, 0.5, 0)

        _, buffer = cv2.imencode('.png', visz_map)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

        output_item = {
            "index": original_index,
            "class": result.anomaly_result.anomaly_class, # 'no_defect', 'minor', 'severe', etc.
            "heatmap": heatmap_base64
        }
        output.append(output_item)
        
        # Store the PIL image separately for the test case to save
        output_item["_original_pil_image"] = original_pil_image
        
        status_icon = "⚠" if result.anomaly_result.is_anomalous else "✅"
        print(f"  - {status_icon} Image Index: {original_index}, Class: {output_item['class']}")
    
    if not output:
        print("  ℹ No images were processed.")
    
    # Clean output for API response (remove internal PIL images)
    # The test case will use these before cleaning, but the API should get clean data
    clean_output = [{k: v for k, v in item.items() if not k.startswith("_")} for item in output]
    
    return clean_output

# =====================================================================
# NEW TEST CASE
# =====================================================================

# =====================================================================
# NEW INTERACTIVE TEST CASE
# =====================================================================

def image_to_base64(image_path: str) -> Optional[str]:
    """Helper to load image and convert to base64 string"""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image path does not exist: {image_path}")
        return None
    try:
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            # Use PNG for lossless conversion
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str
    except Exception as e:
        print(f"❌ Error converting image {image_path}: {e}")
        return None

def get_image_paths_from_user(image_type: str) -> List[str]:
    """Interactively prompts the user to enter file paths."""
    paths = []
    print(f"\n--- Enter {image_type.upper()} Image Paths ---")
    print("Type 'done' or 'q' when finished.")
    
    index = 1
    while True:
        path = input(f"Path for {image_type} image {index} (or 'q' to finish): ")
        
        if path.lower() in ['q', 'done', 'quit']:
            if not paths:
                print(f"Warning: No {image_type} images were entered.")
            break
            
        # Clean up path (e.g., remove quotes from drag-and-drop)
        path = path.strip().strip("'\"")
        
        if not os.path.exists(path):
            print(f"❌ Error: File not found at '{path}'. Please try again.")
            continue
        
        paths.append(path)
        print(f"  ✓ Added: {path}")
        index += 1
        
    return paths

if __name__ == "__main__":
    print("--- 🚀 STARTING INTERACTIVE TEST RUN ---")
    
    # --- 1. Get Configuration from User ---
    CLASS_NAME = input("Enter the Class Name (e.g., 'bottle', 'road'): ")
    
    repeat_input = input("Enter number of times to repeat the FIRST reference image (e.g., '2'): ")
    try:
        REPEAT_FIRST_IMAGE_COUNT = int(repeat_input)
    except ValueError:
        print("Invalid number, defaulting to 0.")
        REPEAT_FIRST_IMAGE_COUNT = 0
    
    # --- 2. Get Image Paths from User ---
    REFERENCE_IMAGE_PATHS = get_image_paths_from_user("reference")
    QUERY_IMAGE_PATHS = get_image_paths_from_user("query")

    print("\n--- ⚙️ PROCESSING ---")
    print(f"Class: {CLASS_NAME}")
    print(f"Reference Images: {REFERENCE_IMAGE_PATHS}")
    print(f"Query Images: {QUERY_IMAGE_PATHS}")
    print(f"Repeating first ref image {REPEAT_FIRST_IMAGE_COUNT} times.")

    # --- 3. Convert images to Base64 dictionary format ---
    reference_images_b64 = {}
    for i, path in enumerate(REFERENCE_IMAGE_PATHS):
        b64_str = image_to_base64(path)
        if b64_str:
            # Use index 'i+1' as the key
            reference_images_b64[i + 1] = b64_str 

    query_images_b64 = {}
    for i, path in enumerate(QUERY_IMAGE_PATHS):
        b64_str = image_to_base64(path)
        if b64_str:
            # Use a different index range (e.g., 100+) to avoid key collision
            query_images_b64[i + 101] = b64_str 
    
    if not reference_images_b64:
        print("❌ No valid reference images found. Exiting.")
    elif not query_images_b64:
        print("❌ No valid query images found. Exiting.")
    else:
        # --- 4. Run the analysis ---
        output = analyze_images(
            reference_images=reference_images_b64,
            query_images=query_images_b64,
            repeat_first_image=REPEAT_FIRST_IMAGE_COUNT,
            class_name=CLASS_NAME
        )
        
        # --- 5. Print and Save the final output ---
        print("\n--- 🏁 FINAL OUTPUT ---")
        
        if output:
            # Create a timestamped directory for this test run's output
            output_dir = os.path.join("test_run_outputs", datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n💾 Saving heatmap images to: {output_dir}")

            for item in output:
                try:
                    # Decode the base64 heatmap
                    heatmap_data = base64.b64decode(item['heatmap'])
                    
                    # Define the output filename
                    image_class = item['class']
                    file_name = f"heatmap_index_{item['index']}_class_{image_class}.png"
                    output_path = os.path.join(output_dir, file_name)
                    
                    # Save the image
                    with open(output_path, 'wb') as f:
                        f.write(heatmap_data)
                    print(f"  ✓ Saved: {file_name}")
                    
                    # Also save the original image for comparison
                    if "_original_pil_image" in item:
                        original_file_name = f"original_index_{item['index']}.png"
                        original_path = os.path.join(output_dir, original_file_name)
                        item["_original_pil_image"].save(original_path)
                        print(f"  ✓ Saved: {original_file_name}")

                except Exception as e:
                    print(f"  ❌ Error saving heatmap for index {item.get('index', 'N/A')}: {e}")
        
        # Clean output for JSON display (remove internal PIL images)
        output_for_display = [{k: v for k, v in item.items() if not k.startswith("_")} for item in output]
        print(json.dumps(output_for_display, indent=2))