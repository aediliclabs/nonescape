import { ClassificationResult, LoadingProgress, LocalClassifier } from "@aedilic/nonescape";
import React, { useCallback, useEffect, useState } from "react";

interface ImageItem {
  id: string;
  file: File;
  url: string;
  name: string;
  result: ClassificationResult | null;
  isClassifying: boolean;
}

const NonescapeDemo: React.FC = () => {
  const [classifier, setClassifier] = useState<LocalClassifier | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState<LoadingProgress | null>(null);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);

  useEffect(() => {
    const initClassifier = async () => {
      try {
        const newClassifier = new LocalClassifier({
          onProgress: setLoadingProgress,
        });

        await newClassifier.initialize();
        setClassifier(newClassifier);
        setIsModelLoading(false);
      } catch (error) {
        console.error("Failed to initialize classifier:", error);
        setIsModelLoading(false);
      }
    };

    initClassifier();

    return () => {
      classifier?.dispose();
    };
  }, []);

  const classifyImage = useCallback(
    async (imageItem: ImageItem) => {
      if (!classifier) return;

      setImages((prev) => prev.map((img) => (img.id === imageItem.id ? { ...img, isClassifying: true } : img)));

      try {
        const result = await classifier.predict(imageItem.file);
        setImages((prev) =>
          prev.map((img) => (img.id === imageItem.id ? { ...img, result, isClassifying: false } : img))
        );
      } catch (error) {
        console.error("Error classifying image:", error);
        setImages((prev) =>
          prev.map((img) =>
            img.id === imageItem.id
              ? { ...img, result: { isSynthetic: false, confidence: 0 }, isClassifying: false }
              : img
          )
        );
      }
    },
    [classifier]
  );

  const handleFiles = useCallback(
    (files: FileList | File[]) => {
      const newImages: ImageItem[] = Array.from(files)
        .filter((file) => file.type.startsWith("image/"))
        .map((file) => ({
          id: `${Date.now()}-${Math.random()}`,
          file,
          url: URL.createObjectURL(file),
          name: file.name,
          result: null,
          isClassifying: false,
        }));

      setImages((prev) => [...prev, ...newImages]);

      if (classifier) {
        newImages.forEach(classifyImage);
      }
    },
    [classifier, classifyImage]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);

      if (e.dataTransfer.files) {
        handleFiles(e.dataTransfer.files);
      }
    },
    [handleFiles]
  );

  const removeImage = useCallback((imageId: string) => {
    setImages((prev) => {
      const filtered = prev.filter((img) => img.id !== imageId);
      const removed = prev.find((img) => img.id === imageId);
      if (removed) {
        URL.revokeObjectURL(removed.url);
      }
      return filtered;
    });
  }, []);

  const progressPercent = loadingProgress ? Math.round((loadingProgress.current / loadingProgress.total) * 100) : 0;

  if (isModelLoading) {
    return (
      <div style={{ padding: "20px", textAlign: "center" }}>
        <h2>Loading Nonescape Model...</h2>
        <div
          style={{
            width: "100%",
            height: "20px",
            backgroundColor: "#f0f0f0",
            borderRadius: "10px",
            overflow: "hidden",
            margin: "20px 0",
          }}
        >
          <div
            style={{
              width: `${progressPercent}%`,
              height: "100%",
              backgroundColor: "#007bff",
              transition: "width 0.3s ease",
            }}
          />
        </div>
        <p>{progressPercent}%</p>
      </div>
    );
  }

  return (
    <div style={{ padding: "20px", maxWidth: "800px", margin: "0 auto" }}>
      <h1>Nonescape Image Classifier</h1>
      <p>Upload images to detect if they are AI-generated or real.</p>

      <div
        style={{
          border: `2px dashed ${isDragOver ? "#007bff" : "#ccc"}`,
          borderRadius: "8px",
          padding: "40px",
          textAlign: "center",
          margin: "20px 0",
          cursor: "pointer",
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setIsDragOver(false);
        }}
        onDrop={handleDrop}
        onClick={() => {
          const input = document.createElement("input");
          input.type = "file";
          input.multiple = true;
          input.accept = "image/*";
          input.onchange = (e) => {
            const target = e.target as HTMLInputElement;
            if (target.files) {
              handleFiles(target.files);
            }
          };
          input.click();
        }}
      >
        <p>Click to upload images or drag and drop</p>
      </div>

      <div>
        {images.map((image) => (
          <div
            key={image.id}
            style={{
              margin: "20px 0",
              padding: "15px",
              borderRadius: "8px",
              border: "1px solid #ddd",
              backgroundColor: image.result ? (image.result.isSynthetic ? "#ffe6e6" : "#e6ffe6") : "#f9f9f9",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div style={{ flex: 1 }}>
                <h4>{image.name}</h4>
                <img
                  src={image.url}
                  alt="Preview"
                  style={{
                    maxWidth: "200px",
                    maxHeight: "200px",
                    borderRadius: "8px",
                    margin: "10px 0",
                  }}
                />

                {image.isClassifying && <p>Analyzing...</p>}

                {image.result && !image.isClassifying && (
                  <>
                    <p>
                      <strong>{image.result.isSynthetic ? "AI Generated" : "Real"}</strong>
                    </p>
                    <p>Confidence: {Math.ceil(image.result.confidence * 100)}%</p>
                  </>
                )}
              </div>

              <button
                onClick={() => removeImage(image.id)}
                style={{
                  background: "#dc3545",
                  color: "white",
                  border: "none",
                  padding: "5px 10px",
                  borderRadius: "4px",
                  cursor: "pointer",
                }}
              >
                Remove
              </button>
            </div>
          </div>
        ))}
      </div>

      {images.length > 1 && (
        <button
          onClick={async () => {
            if (!classifier) return;

            const unclassifiedImages = images.filter((img) => !img.result && !img.isClassifying);
            if (unclassifiedImages.length > 0) {
              const files = unclassifiedImages.map((img) => img.file);
              try {
                const results = await classifier.predict(files);

                setImages((prev) =>
                  prev.map((img) => {
                    const index = unclassifiedImages.findIndex((unclassified) => unclassified.id === img.id);
                    if (index !== -1) {
                      return { ...img, result: results[index], isClassifying: false };
                    }
                    return img;
                  })
                );
              } catch (error) {
                console.error("Batch prediction error:", error);
              }
            }
          }}
          style={{
            background: "#007bff",
            color: "white",
            border: "none",
            padding: "10px 20px",
            borderRadius: "4px",
            cursor: "pointer",
            margin: "20px 0",
          }}
        >
          Classify All Remaining Images
        </button>
      )}
    </div>
  );
};

export default NonescapeDemo;
