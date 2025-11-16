from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import shutil
import os

# Import the main analysis function from your script
try:
    from app import analyze_images, Config
except ImportError as e:
    print(f"Error: Could not import 'analyze_images' from app.py: {e}")
    print("Please ensure app.py is in the same directory and has no syntax errors.")
    analyze_images = None
    Config = None

# --- Pydantic Models for Request and Response ---

class ImageInput(BaseModel):
    """
    Defines the structure for a single image input.
    """
    index: int
    image: str
    is_reference: bool = Field(False, description="Set to true if this is a reference image for the gallery.")

class AnalyzeRequest(BaseModel):
    """
    Defines the expected request body for the /analyze endpoint.
    """
    images: List[ImageInput]
    repeat_first_image: int = Field(0, description="Number of times to repeat the first reference image.", json_schema_extra={"example": 2})
    class_name: str = Field(..., description="The class name for the objects in the images (e.g., 'car', 'bottle').", json_schema_extra={"example": "road"})

class AnalyzeResponseItem(BaseModel):
    """
    Defines a single item in the response list.
    """
    index: int
    class_name: str = Field(..., alias="class")
    heatmap: str

# --- FastAPI Application ---

app = FastAPI(
    title="Visual Difference Engine API",
    description="An API to analyze images for anomalies using the WinCLIP model.",
    version="1.0.0"
)

# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
async def read_root():
    """
    Root endpoint for basic health check.
    """
    return {"status": "ok", "message": "Visual Difference Engine API is running."}

@app.post("/analyze", 
          response_model=List[AnalyzeResponseItem],
          tags=["Analysis"],
          summary="Analyze Query Images Against a Reference Gallery")
async def analyze_endpoint(request: AnalyzeRequest):
    """
    Receives a list of images, separates them into reference and query groups,
    builds a gallery from the reference images, and performs anomaly detection
    on the query images.

    - **images**: A list of image objects, each with an `index`, `image` (base64), and `is_reference` flag.
    - **repeat_first_image**: An integer specifying how many times to duplicate the first reference image to augment the gallery.
    - **class_name**: The category of the object being analyzed (e.g., 'road', 'bottle').

    Returns a list of analysis results, each containing the original index,
    the determined anomaly class, and a base64 encoded heatmap image.
    """
    if analyze_images is None:
        raise HTTPException(
            status_code=500,
            detail="Analysis function is not available due to an import error from app.py."
        )
    
    # Separate images into reference and query dictionaries
    reference_images = {img.index: img.image for img in request.images if img.is_reference}
    query_images = {img.index: img.image for img in request.images if not img.is_reference}

    if not reference_images:
        raise HTTPException(status_code=400, detail="No reference images provided.")
    if not query_images:
        raise HTTPException(status_code=400, detail="No query images provided.")

    try:
        # Call the core logic from app.py with the separated image dictionaries
        results = analyze_images(
            reference_images=reference_images,
            query_images=query_images,
            repeat_first_image=request.repeat_first_image,
            class_name=request.class_name
        )
        
        # The 'analyze_images' function already returns data in the desired format.
        # The pydantic model will automatically handle the alias for 'class'.
        return results

    except Exception as e:
        # Catch potential errors from the analysis script
        print(f"An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cleanup", tags=["Memory Management"])
async def cleanup_memory():
    """
    Clean up the memory directory by deleting all stored galleries and results.
    This endpoint clears all cached data to free up disk space.
    
    Returns:
        dict: Status message confirming memory cleanup
    """
    try:
        memory_dir = "winclip_memory"
        results_dir = "winclip_results_persistent"
        
        cleanup_result = {
            "status": "success",
            "message": "Memory cleaned successfully",
            "cleaned_directories": []
        }
        
        # Clean winclip_memory directory
        if os.path.exists(memory_dir):
            try:
                shutil.rmtree(memory_dir)
                os.makedirs(memory_dir, exist_ok=True)
                cleanup_result["cleaned_directories"].append(memory_dir)
                print(f"✓ Cleaned {memory_dir}")
            except Exception as e:
                print(f"Warning: Could not fully clean {memory_dir}: {e}")
        
        # Clean results directory
        if os.path.exists(results_dir):
            try:
                shutil.rmtree(results_dir)
                os.makedirs(results_dir, exist_ok=True)
                cleanup_result["cleaned_directories"].append(results_dir)
                print(f"✓ Cleaned {results_dir}")
            except Exception as e:
                print(f"Warning: Could not fully clean {results_dir}: {e}")
        
        return cleanup_result
    
    except Exception as e:
        print(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/status", tags=["Health Check"])
async def get_status():
    """
    Get the current status of stored galleries and memory usage.
    
    Returns:
        dict: Information about stored galleries and directories
    """
    try:
        memory_dir = "winclip_memory"
        galleries = []
        total_size = 0
        
        if os.path.exists(memory_dir):
            for file in os.listdir(memory_dir):
                if file.endswith('_metadata.json'):
                    gallery_name = file.replace('gallery_', '').replace('_metadata.json', '')
                    galleries.append(gallery_name)
                    file_path = os.path.join(memory_dir, file)
                    total_size += os.path.getsize(file_path)
        
        return {
            "status": "running",
            "stored_galleries": galleries,
            "gallery_count": len(galleries),
            "memory_directory": memory_dir,
            "approximate_size_bytes": total_size
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not get status: {str(e)}")

# --- How to Run ---
#
# 1. Make sure you have FastAPI and an ASGI server installed:
#    pip install "fastapi[all]"
#
# 2. Run the server from your terminal:
#    uvicorn api_server:app --reload
#
# 3. Access the interactive API documentation at:
#    http://127.0.0.1:8000/docs
#
