
import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { FileImage, Trash2, Edit2, Check, X, Loader2, AlertCircle } from "lucide-react";
import {
  DndContext,
  closestCenter,
  PointerSensor,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import {
  SortableContext,
  horizontalListSortingStrategy,
  useSortable,
  arrayMove,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

/* -------------------------
   UI Components
   ------------------------- */
function Button({ children, variant = "primary", onClick, disabled, className = "" }) {
  const base = "px-4 py-2 rounded-lg font-medium transition-all duration-200";
  const styles = {
    primary: "bg-blue-600 text-white hover:bg-blue-700 shadow-sm hover:shadow-md disabled:bg-blue-300",
    outline: "border-2 border-slate-300 text-slate-700 bg-white hover:bg-slate-50",
    ghost: "text-slate-600 bg-transparent hover:bg-slate-100",
    danger: "bg-red-600 text-white hover:bg-red-700",
  };
  return (
    <button
      className={`${base} ${styles[variant] || styles.primary} ${
        disabled ? "opacity-50 cursor-not-allowed" : ""
      } ${className}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

/* -------------------------
   Sortable Image Item
   ------------------------- */
function SortableImageItem({ id, img, index, removeAt, renameAt, toggleReference }) {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id });
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(img.name);

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  const handleSaveRename = () => {
    if (editName.trim()) {
      renameAt(index, editName.trim());
      setIsEditing(false);
    }
  };

  const handleCancelRename = () => {
    setEditName(img.name);
    setIsEditing(false);
  };

  return (
    <motion.div
      ref={setNodeRef}
      style={style}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      className="flex-shrink-0"
    >
      <div className={`w-48 rounded-xl overflow-hidden shadow-lg bg-white border-2 transition-all duration-300 hover:shadow-xl group ${
        img.is_reference ? 'border-green-400' : 'border-slate-200 hover:border-blue-400'
      }`}>
        {/* Image Preview */}
        <div className="relative h-40 bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center overflow-hidden">
          <img
            src={img.src}
            alt={img.name}
            className="object-contain h-full w-full"
          />
          
          {/* Overlay on hover */}
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-all duration-300 flex items-center justify-center opacity-0 group-hover:opacity-100">
            <div className="flex gap-2">
              <button
                onClick={() => setIsEditing(true)}
                className="p-2 bg-white rounded-full hover:bg-blue-100 transition-colors"
                title="Rename"
              >
                <Edit2 size={16} className="text-blue-600" />
              </button>
              <button
                onClick={() => removeAt(index)}
                className="p-2 bg-white rounded-full hover:bg-red-100 transition-colors"
                title="Remove"
              >
                <Trash2 size={16} className="text-red-600" />
              </button>
            </div>
          </div>

          {/* Drag handle */}
          <div 
            {...attributes} 
            {...listeners} 
            className="absolute top-2 right-2 p-1.5 bg-white/90 rounded-lg cursor-grab active:cursor-grabbing shadow-md"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <circle cx="5" cy="6" r="2" fill="#64748b" />
              <circle cx="12" cy="6" r="2" fill="#64748b" />
              <circle cx="19" cy="6" r="2" fill="#64748b" />
              <circle cx="5" cy="12" r="2" fill="#64748b" />
              <circle cx="12" cy="12" r="2" fill="#64748b" />
              <circle cx="19" cy="12" r="2" fill="#64748b" />
              <circle cx="5" cy="18" r="2" fill="#64748b" />
              <circle cx="12" cy="18" r="2" fill="#64748b" />
              <circle cx="19" cy="18" r="2" fill="#64748b" />
            </svg>
          </div>

          {/* Index badge */}
          <div className="absolute top-2 left-2 bg-blue-600 text-white text-xs font-bold px-2 py-1 rounded-full shadow-md">
            #{index + 1}
          </div>

          {/* Reference badge */}
          {img.is_reference && (
            <div className="absolute bottom-2 left-2 bg-green-600 text-white text-xs font-bold px-2 py-1 rounded-full shadow-md">
              Reference
            </div>
          )}
        </div>

        {/* File Info */}
        <div className="p-3">
          {isEditing ? (
            <div className="space-y-2">
              <input
                type="text"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                className="w-full px-2 py-1 text-sm border-2 border-blue-400 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                autoFocus
                onKeyPress={(e) => {
                  if (e.key === 'Enter') handleSaveRename();
                  if (e.key === 'Escape') handleCancelRename();
                }}
              />
              <div className="flex gap-1">
                <button
                  onClick={handleSaveRename}
                  className="flex-1 p-1 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                >
                  <Check size={14} className="mx-auto" />
                </button>
                <button
                  onClick={handleCancelRename}
                  className="flex-1 p-1 bg-gray-400 text-white rounded hover:bg-gray-500 transition-colors"
                >
                  <X size={14} className="mx-auto" />
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="font-semibold text-sm truncate text-slate-800" title={img.name}>
                {img.name}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {img.width} × {img.height}
              </div>
              <button
                onClick={() => toggleReference(index)}
                className={`mt-2 w-full text-xs py-1 rounded transition-colors ${
                  img.is_reference 
                    ? 'bg-green-100 text-green-700 hover:bg-green-200' 
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                {img.is_reference ? 'Reference ✓' : 'Mark as Reference'}
              </button>
            </>
          )}
        </div>
      </div>
    </motion.div>
  );
}

/* -------------------------
   Upload Dropzone
   ------------------------- */
function UploadDropzone({ onDrop, hasImages }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    multiple: true,
  });

  return (
    <div className="flex justify-center">
      <div
        {...getRootProps()}
        className={`rounded-2xl border-3 border-dashed transition-all duration-300 p-12 cursor-pointer w-full max-w-2xl
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50 scale-[1.02]' 
            : 'border-slate-300 bg-white hover:border-blue-400 hover:bg-blue-50/50'
          }
          ${hasImages ? 'mb-8' : ''}
        `}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center justify-center gap-4 text-center">
          <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center text-white shadow-lg">
            <FileImage size={36} />
          </div>
          <div>
            <div className="text-xl font-bold text-slate-800 mb-2">
              {isDragActive ? 'Drop images here' : 'Drag & drop your images'}
            </div>
            <div className="text-sm text-slate-500">
              Upload any number of images • Supported: JPG, PNG, GIF, WebP • Reorder by dragging
            </div>
          </div>
          <Button variant="outline" className="mt-2">
            Browse Files
          </Button>
        </div>
      </div>
    </div>
  );
}

/* -------------------------
   Analysis Configuration
   ------------------------- */
function AnalysisConfig({ className, setClassName }) {
  return (
    <div className="max-w-7xl mx-auto px-8 mb-6">
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Analysis Configuration</h3>
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Class Name
          </label>
          <input
            type="text"
            value={className}
            onChange={(e) => setClassName(e.target.value)}
            placeholder="e.g., road, bottle, car"
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>
    </div>
  );
}

/* -------------------------
   Results Display
   ------------------------- */
function ResultsDisplay({ results, images }) {
  if (!results || results.length === 0) return null;

  console.log('Results received:', results);
  console.log('Images array:', images.map((img, idx) => ({ idx, name: img.name, is_reference: img.is_reference })));

  // Get only query images (non-reference) in order
  const queryImages = images.filter(img => !img.is_reference);
  console.log('Query images:', queryImages.map((img, idx) => ({ idx, name: img.name })));

  return (
    <div className="max-w-7xl mx-auto px-8 mt-8">
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h3 className="text-xl font-semibold text-slate-800 mb-6">Analysis Results</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {results.map((result, resultIdx) => {
            // Match result to the corresponding query image
            // The API returns results only for query images in order
            const originalImage = queryImages[resultIdx];
            console.log(`Result ${resultIdx}: class=${result.class}, matched to query image:`, originalImage?.name);
            
            return (
              <div key={resultIdx} className="border border-slate-200 rounded-lg overflow-hidden">
                <div className="relative">
                  <img
                    src={originalImage?.src}
                    alt={originalImage?.name}
                    className="w-full h-48 object-cover"
                  />
                </div>
                <div className="p-4">
                  <h4 className="font-semibold text-sm mb-2">
                    {originalImage?.name || `Query Image #${resultIdx + 1}`}
                  </h4>
                  {result.heatmap && (
                    <div className="mt-3">
                      <p className="text-xs text-slate-600 mb-2">Heatmap:</p>
                      <img
                        src={`data:image/png;base64,${result.heatmap}`}
                        alt="Heatmap"
                        className="w-full rounded border border-slate-200"
                      />
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* -------------------------
   Main App
   ------------------------- */
export default function App() {
  const [images, setImages] = useState([]);
  const [className, setClassName] = useState("road");
  const [repeatFirst, setRepeatFirst] = useState(2);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const toDataUrl = (file) =>
    new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.readAsDataURL(file);
    });

  const onDrop = useCallback(
    async (acceptedFiles) => {
      const mapped = await Promise.all(
        acceptedFiles.map(async (file) => {
          const src = await toDataUrl(file);
          const img = new Image();
          img.src = src;
          await new Promise((r) => (img.onload = r));
          
          // Extract base64 without data URL prefix
          const base64 = src.split(',')[1];
          
          return {
            id: `${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
            file,
            src,
            base64, // Store base64 for API
            name: file.name,
            size: file.size,
            width: img.naturalWidth,
            height: img.naturalHeight,
            is_reference: false, // Default to not reference
          };
        })
      );
      setImages((prev) => [...prev, ...mapped]);
      setResults(null); // Clear previous results
    },
    []
  );

  const removeAt = (index) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
    setResults(null);
  };

  const renameAt = (index, newName) => {
    setImages((prev) =>
      prev.map((img, i) => (i === index ? { ...img, name: newName } : img))
    );
  };

  const toggleReference = (index) => {
    setImages((prev) =>
      prev.map((img, i) => (i === index ? { ...img, is_reference: !img.is_reference } : img))
    );
  };

  const clearAll = () => {
    setImages([]);
    setResults(null);
    setError(null);
  };

  const analyzeImages = async () => {
    if (images.length === 0) return;
    
    setIsAnalyzing(true);
    setError(null);
    
    try {
      // Prepare the request payload - ensure base64 is clean (no data URL prefix)
      const payload = {
        images: images.map((img, index) => ({
          index: index,
          image: img.base64, // Already cleaned in onDrop
          is_reference: img.is_reference
        })),
        repeat_first_image: repeatFirst,
        class_name: className
      };

      console.log('Sending payload:', {
        imageCount: payload.images.length,
        repeatFirst: payload.repeat_first_image,
        className: payload.class_name,
        firstImagePreview: payload.images[0]?.image.substring(0, 50) + '...',
        sampleImage: payload.images[0]
      });

      const requestBody = JSON.stringify(payload);
      console.log('Request body length:', requestBody.length);

      // Make API request - try direct connection first
      const API_URL = 'https://vltnbcrz-8000.inc1.devtunnels.ms/analyze';
      
      let response;
      try {
        // First, try direct connection (requires CORS enabled on backend)
        response = await fetch(API_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: requestBody
        });
      } catch (fetchError) {
        // If direct connection fails with CORS, try through proxy
        console.log('Direct connection failed, trying proxy...', fetchError.message);
        try {
          response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
            body: requestBody
          });
          console.log('Proxy response received:', response.status, response.statusText);
        } catch (proxyError) {
          console.error('Proxy request failed:', proxyError);
          throw proxyError;
        }
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('API Error Response:', errorData);
        throw new Error(`API Error: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const data = await response.json();
      console.log('API Response received successfully:', data);
      setResults(data);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const sensors = useSensors(useSensor(PointerSensor));

  const hasReferenceImages = images.some(img => img.is_reference);
  const hasQueryImages = images.some(img => !img.is_reference);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      {/* Navbar */}
      <nav className="bg-gradient-to-r from-blue-600 to-cyan-500 shadow-lg">
        <div className="max-w-7xl mx-auto px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-white/20 backdrop-blur-sm flex items-center justify-center text-white font-bold text-xl border border-white/30">
                V
              </div>
              <div className="text-white">
                <div className="text-sm font-medium opacity-90">Time-series image comparison</div>
              </div>
            </div>

            <div className="text-2xl font-bold text-white">
              Visual Difference Engine
            </div>

            <div className="flex items-center gap-6">
              <a href="#" className="text-sm text-white/90 hover:text-white font-medium transition-colors">
                About
              </a>
              <a href="#" className="text-sm text-white/90 hover:text-white font-medium transition-colors">
                Docs
              </a>
              <a href="#" className="text-sm text-white/90 hover:text-white font-medium transition-colors">
                GitHub
              </a>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="py-12">
        {/* Header Section */}
        <div className="max-w-7xl mx-auto px-8 mb-6">
          <h2 className="text-2xl font-bold text-slate-800 mb-2">Upload Time-Series Images</h2>
          <div className="flex items-center justify-between">
            <p className="text-slate-600">
              Timeline ({images.length} image{images.length !== 1 ? 's' : ''})
              {hasReferenceImages && <span className="ml-2 text-green-600 font-medium">
                • {images.filter(img => img.is_reference).length} reference
              </span>}
              {hasQueryImages && <span className="ml-2 text-blue-600 font-medium">
                • {images.filter(img => !img.is_reference).length} query
              </span>}
            </p>
            {images.length > 0 && (
              <Button variant="ghost" onClick={clearAll} className="text-red-600 hover:bg-red-50">
                Clear All
              </Button>
            )}
          </div>
        </div>

        {/* Analysis Configuration */}
        {images.length > 0 && (
          <AnalysisConfig
            className={className}
            setClassName={setClassName}
          />
        )}

        {/* Images Timeline - Full Width */}
        {images.length > 0 && (
          <div className="mb-8 w-full overflow-x-auto">
            <DndContext
              sensors={sensors}
              collisionDetection={closestCenter}
              onDragEnd={(event) => {
                const { active, over } = event;
                if (!over || active.id === over.id) return;
                const oldIndex = images.findIndex((it) => it.id === active.id);
                const newIndex = images.findIndex((it) => it.id === over.id);
                if (oldIndex !== -1 && newIndex !== -1) {
                  setImages((prev) => arrayMove(prev, oldIndex, newIndex));
                }
              }}
            >
              <SortableContext items={images.map((i) => i.id)} strategy={horizontalListSortingStrategy}>
                <div className="flex justify-center w-full">
                  <div className="flex gap-4 pb-4 px-8">
                    <AnimatePresence>
                      {images.map((img, index) => (
                        <SortableImageItem
                          key={img.id}
                          id={img.id}
                          img={img}
                          index={index}
                          removeAt={removeAt}
                          renameAt={renameAt}
                          toggleReference={toggleReference}
                        />
                      ))}
                    </AnimatePresence>
                  </div>
                </div>
              </SortableContext>
            </DndContext>
          </div>
        )}

        {/* Upload Dropzone */}
        <div className="max-w-7xl mx-auto px-8">
          <UploadDropzone onDrop={onDrop} hasImages={images.length > 0} />
        </div>

        {/* Error Display */}
        {error && (
          <div className="max-w-7xl mx-auto px-8 mt-6">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
              <AlertCircle className="text-red-600 flex-shrink-0 mt-0.5" size={20} />
              <div>
                <h4 className="font-semibold text-red-800">Analysis Failed</h4>
                <p className="text-sm text-red-600 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        {images.length >= 1 && (
          <div className="max-w-7xl mx-auto px-8 mt-8 flex justify-center gap-4">
            <Button 
              className="px-8 py-3 text-lg flex items-center gap-2" 
              onClick={analyzeImages}
              disabled={isAnalyzing || !hasReferenceImages || !hasQueryImages}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="animate-spin" size={20} />
                  Analyzing...
                </>
              ) : (
                `Analyze Images (${images.length})`
              )}
            </Button>
          </div>
        )}

        {!hasReferenceImages && images.length > 0 && (
          <div className="max-w-7xl mx-auto px-8 mt-4 text-center">
            <p className="text-sm text-amber-600">
              ⚠️ Please mark at least one image as "Reference" before analyzing
            </p>
          </div>
        )}

        {/* Results Display */}
        <ResultsDisplay results={results} images={images} />
      </main>
    </div>
  );
}