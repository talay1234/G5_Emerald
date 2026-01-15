import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Camera, Upload, X, History } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import axios from "axios";

// ---------- Types & Mappers ----------

/** Species model API response shape (model1) */
type SpeciesApiResponse = {
  all_probabilities: number[];
  class_index: number;
  confidence: number; // 0..1
  predicted_class: string; // EN label
  success: boolean;
  top_predictions: {
    class: string;
    class_index: number;
    confidence: number; // 0..1
  }[];
};

/** Venomous model API response shape (model2) */
type VenomApiResponse = {
  all_probabilities: number[];
  class_index: number;
  confidence: number; // 0..1
  predicted_class: "Venomous" | "Non Venomous";
  success: boolean;
  top_predictions: {
    class: "Venomous" | "Non Venomous";
    class_index: number;
    confidence: number; // 0..1
  }[];
};

type ModelType = "species" | "venomous";

/** EN -> TH mapping for species name shown to users */
const speciesThMap: Record<string, string> = {
  Banded_Krait: "งูสามเหลี่ยม",
  Eastern_Russells_Viper: "งูแมวเซา",
  Green_Pit_Viper: "งูเขียวหางไหม้",
  Indo_Chinese_Rat_Snake: "งูสิง",
  King_Cobra: "งูจงอาง",
  Malayan_Krait: "งูทับสมิงคลา",
  Malayan_Pit_Viper: "งูกะปะ",
  Monocled_Cobra: "งูเห่า",
};

/** EN -> TH mapping for venom status */
const venomThMap: Record<string, string> = {
  Venomous: "มีพิษ",
  "Non Venomous": "ไม่มีพิษ",
};

/** Helper: convert 0..1 to 0..100 integer percent */
function toPercent(x: number) {
  return Math.round(x * 100);
}

/** Map raw Species API response to app-facing result */
function mapSpeciesResult(api: SpeciesApiResponse) {
  const nameEn = api.predicted_class;
  const nameTh = speciesThMap[nameEn] ?? nameEn;
  return {
    name: nameTh, // TH shown to users
    nameEn,
    confidence: toPercent(api.confidence), // integer percent
    top3: api.top_predictions.map((p) => ({
      nameTh: speciesThMap[p.class] ?? p.class,
      nameEn: p.class,
      confidence: toPercent(p.confidence),
    })),
  };
}

/** Map raw Venom API response to app-facing result */
function mapVenomResult(api: VenomApiResponse) {
  const nameEn = api.predicted_class;
  const nameTh = venomThMap[nameEn] ?? nameEn;
  return {
    name: nameTh, // TH shown to users
    nameEn,
    confidence: toPercent(api.confidence), // integer percent
    top2: api.top_predictions.map((p) => ({
      nameTh: venomThMap[p.class] ?? p.class,
      nameEn: p.class,
      confidence: toPercent(p.confidence),
    })),
  };
}

const Capture = () => {
  const navigate = useNavigate();
  const { toast } = useToast();

  // UI states
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // Keep raw response if needed for debugging/analytics
  // const [predicted, setPredicted] = useState<any>({});
  const [selectedModel, setSelectedModel] = useState<ModelType>("species");

  // Refs for hidden inputs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  /** Validate and store selected image, and build preview */
  const handleImageSelect = (file: File) => {
    if (file.size > 10 * 1024 * 1024) {
      toast({
        title: "ไฟล์ใหญ่เกินไป",
        description: "กรุณาเลือกไฟล์ที่มีขนาดไม่เกิน 10MB",
        variant: "destructive",
      });
      return;
    }
    if (!file.type.startsWith("image/")) {
      toast({
        title: "ประเภทไฟล์ไม่ถูกต้อง",
        description: "กรุณาเลือกไฟล์รูปภาพ (JPG, PNG)",
        variant: "destructive",
      });
      return;
    }
    setSelectedImage(file);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  /** Handle file input change event */
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) handleImageSelect(file);
  };

  /** Visual feedback for drag-over */
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  /** Reset drag state when leaving drop area */
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  /** Drop handler to accept and validate the dropped file */
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleImageSelect(file);
  };

  /** Clear current image and preview */
  const clearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (cameraInputRef.current) cameraInputRef.current.value = "";
  };

  /**
   * Call backend prediction API (model1 or model2),
   * map raw API response into our standardized `result`,
   * and navigate to /result with full state.
   */
  const handlePredict = async () => {
    if (!selectedImage) return;
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", selectedImage);

    // NOTE: axios will set proper multipart boundary when sending FormData,
    // but we can still set the generic Content-Type for clarity.
    const _config = {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    };

    try {
      if (selectedModel === "species") {
        const resVenom = await axios.post<VenomApiResponse>(
          "https://g5tuesa.consolutechcloud.com/backend/model2/predict",
          formData,
          _config
        );
        if (
          resVenom.data?.success &&
          resVenom.data?.confidence > 0.6 &&
          resVenom.data?.predicted_class == "Venomous"
        ) {
          const resCat = await axios.post<SpeciesApiResponse>(
            "https://g5tuesa.consolutechcloud.com/backend/model1/predict",
            formData,
            _config
          );
          if (
            resCat.data?.success &&
            resCat.data?.predicted_class !== "Indo_Chinese_Rat_Snake"
          ) {
            const mapped = mapSpeciesResult(resCat.data);
            // setPredicted(res.data);
            navigate("/result", {
              state: {
                imageUrl: imagePreview,
                fileName: selectedImage.name,
                modelType: selectedModel as ModelType,
                predicted: resCat.data, // raw response
                result: mapped, // normalized for UI
              },
            });
          } else {
            toast({
              title: "ไม่สามารถทำนายได้",
              description: "ข้อมูลไม่เพียงพอ กรุณาลองใหม่อีกครั้ง",
              variant: "destructive",
            });
          }
        } else {
          toast({
            title: "ไม่สามารถทำนายได้",
            description: "ข้อมูลไม่เพียงพอ กรุณาลองใหม่อีกครั้ง",
            variant: "destructive",
          });
        }
      } else {
        const res = await axios.post<VenomApiResponse>(
          "https://g5tuesa.consolutechcloud.com/backend/model2/predict",
          formData,
          _config
        );
        if (res.data?.success && res.data?.confidence > 0.8) {
          const mapped = mapVenomResult(res.data);
          // setPredicted(res.data);
          navigate("/result", {
            state: {
              imageUrl: imagePreview,
              fileName: selectedImage.name,
              modelType: selectedModel as ModelType,
              predicted: res.data, // raw response
              result: mapped, // normalized for UI
            },
          });
        } else {
          toast({
            title: "ไม่สามารถทำนายได้",
            description: "ข้อมูลไม่เพียงพอ กรุณาลองใหม่อีกครั้ง",
            variant: "destructive",
          });
        }
      }
    } catch (err) {
      toast({
        title: "เกิดข้อผิดพลาด",
        description: "ไม่สามารถวิเคราะห์ภาพได้ ลองใหม่อีกครั้ง",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-md mx-auto space-y-6">
        {/* Header with History Button */}
        <div className="relative text-center pt-8 pb-4">
          <Button
            variant="outline"
            size="sm"
            className="absolute right-0 top-8 bg-gray-200"
            onClick={() => navigate("/history")}
            aria-label="Emerald history"
          >
            <History className="h-4 w-4" />
          </Button>
          <h1 className="text-2xl font-semibold text-foreground mb-2 mr-2">
            <span
              className="text-3xl md:text-4xl font-extrabold tracking-tight
  bg-gradient-to-r from-emerald-400 via-emerald-500 to-emerald-600
  bg-clip-text text-transparent"
            >
              Emerald
            </span>

            <p>ระบุชนิดงูจากภาพ</p>
          </h1>
          <p className="text-muted-foreground text-sm">
            ถ่ายรูปหรือเลือกไฟล์เพื่อวิเคราะห์ด้วย Emerald
          </p>
        </div>

        <div className="card-primary p-4">
          {/* Image Preview */}
          {imagePreview ? (
            <div className="card-primary p-4 relative">
              <div className="relative">
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="w-full h-64 object-cover rounded-lg"
                />
                <Button
                  variant="destructive"
                  size="sm"
                  className="absolute top-2 right-2"
                  onClick={clearImage}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <p className="text-xs text-muted-foreground mt-2 text-center">
                {selectedImage?.name}
              </p>
            </div>
          ) : (
            // Upload Area
            <div
              className={`upload-area p-8 text-center cursor-pointer ${
                isDragging ? "upload-area-active" : ""
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="font-medium text-foreground mb-2">
                อัปโหลดรูปภาพ
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                ลากไฟล์มาวาง หรือคลิกเพื่อเลือกไฟล์
              </p>
              <p className="text-xs text-muted-foreground">
                รองรับ JPG, PNG (ไม่เกิน 10MB)
              </p>
            </div>
          )}

          {/* Camera Button */}
          {!imagePreview && (
            <div className="pt-6">
              <Button
                variant="outline"
                className="w-full"
                onClick={() => cameraInputRef.current?.click()}
              >
                <Camera className="h-5 w-5 mr-2" />
                ถ่ายรูปด้วยกล้อง
              </Button>
            </div>
          )}
        </div>

        {/* Model Selection Dropdown */}
        <div className="card-primary p-2">
          <label className="text-sm font-medium text-foreground mb-2 block">
            เลือกโมเดล
          </label>
          <Select
            value={selectedModel}
            onValueChange={(value: ModelType) => setSelectedModel(value)}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="species">ทำนายชนิดงู</SelectItem>
              <SelectItem value="venomous">ทำนายว่างูมีพิษหรือไม่</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Predict Button */}
        <Button
          className="w-full !bg-emerald-600 hover:!bg-emerald-700 focus-visible:!ring-emerald-500"
          disabled={!selectedImage || isLoading}
          onClick={handlePredict}
        >
          {isLoading ? (
            <>
              <div className="spinner h-4 w-4 mr-2" />
              กำลังวิเคราะห์...
            </>
          ) : selectedModel === "species" ? (
            "ระบุชนิดงู"
          ) : (
            "ระบุประเภทงู"
          )}
        </Button>

        {selectedImage && (
          <p className="text-xs text-muted-foreground text-center">
            กดปุ่มเพื่อวิเคราะห์ชนิดงูจากรูปภาพ
          </p>
        )}

        {/* Hidden file inputs */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
          className="hidden"
        />
        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleFileUpload}
          className="hidden"
        />
      </div>
    </div>
  );
};

export default Capture;
