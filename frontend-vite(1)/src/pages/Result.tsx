import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { RotateCcw, CheckCircle, ChevronUp, ChevronDown } from "lucide-react";

type ModelType = "species" | "venomous";

/** State passed from Capture after mapping */
interface LocationState {
  predicted: any; // raw API response (for debugging/telemetry if needed)
  result: {
    name: string; // TH label shown to users
    nameEn?: string; // optional EN label
    confidence: number; // 0..100 integer percent
    top3?: { nameTh: string; nameEn: string; confidence: number }[]; // species only
    top2?: { nameTh: string; nameEn: string; confidence: number }[]; // venom only
  };
  imageUrl: string;
  fileName: string;
  modelType: ModelType;
}

const Result = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as LocationState | undefined;

  const topAnchorRef = useRef<HTMLDivElement>(null);
  const adviceRef = useRef<HTMLDivElement>(null);
  const [isAtAdvice, setIsAtAdvice] = useState(false);

  /**
   * On mount:
   * - guard invalid entry (no state) -> redirect home
   * - append a simple history record in localStorage
   */
  useEffect(() => {
    if (!state?.predicted || !state?.result) {
      navigate("/");
      return;
    }

    const historyData = localStorage.getItem("snakeHistory");
    const history = historyData ? JSON.parse(historyData) : [];
    const newEntry = {
      id: Date.now().toString(),
      snakeName: state.result.name,
      timestamp: new Date().toISOString(),
      modelType: state.modelType,
      confidence: state.result.confidence,
    };
    history.unshift(newEntry);
    localStorage.setItem("snakeHistory", JSON.stringify(history.slice(0, 50)));
  }, [state, navigate]);

  useEffect(() => {
    if (!adviceRef.current) return;

    const obs = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        setIsAtAdvice(entry.isIntersecting);
      },
      {
        root: null, // viewport
        threshold: 0.4, // consider "in view" when ~40% of advice is visible
      }
    );

    obs.observe(adviceRef.current);
    return () => obs.disconnect();
  }, []);

  /** Smoothly scroll to the advice section */
  const handleScrollToAdvice = () => {
    adviceRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  /** Smoothly scroll back to the page top */
  const handleScrollToTop = () => {
    topAnchorRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  if (!state?.result) return null;

  const { result, imageUrl, fileName, modelType } = state;

  /** Color helper for confidence text */
  // const getConfidenceColor = (confidence: number) => {
  //   if (confidence >= 80) return "text-secondary";
  //   if (confidence >= 60) return "text-yellow-600";
  //   return "text-orange-600";
  // };

  /** Human-readable confidence level */
  // const getConfidenceText = (confidence: number) => {
  //   if (confidence >= 80) return "ความมั่นใจสูง";
  //   if (confidence >= 60) return "ความมั่นใจปานกลาง";
  //   return "ความมั่นใจต่ำ";
  // };

  return (
    <div className="min-h-screen bg-background p-4" ref={topAnchorRef}>
      <div className="max-w-md mx-auto space-y-6 pt-8">
        {/* Header */}
        <div className="text-center">
          <div className="flex items-center justify-center gap-3 mb-2">
            <div className="flex items-center justify-center w-16 h-16 bg-secondary/10 rounded-full">
              <CheckCircle className="h-8 w-8 text-secondary" />
            </div>
            <span
              className="text-3xl md:text-4xl font-extrabold tracking-tight leading-none
      bg-gradient-to-r from-emerald-400 via-emerald-500 to-emerald-600
      bg-clip-text text-transparent"
            >
              Emerald
            </span>
          </div>
          <h1 className="text-xl font-semibold text-foreground mb-2">
            ผลการวิเคราะห์
          </h1>
        </div>

        {/* Result Card */}
        <div className="card-primary p-6 result-card-enter">
          {/* Species/Status */}
          <div className="text-center mb-6">
            <h2 className="text-2xl font-semibold text-foreground mb-2">
              {result.name}
            </h2>
            {/* <div className="flex items-center justify-center space-x-2">
              <span
                className={`text-lg font-medium ${getConfidenceColor(
                  result.confidence
                )}`}
              >
                {result.confidence}%
              </span>
              <span className="text-sm text-muted-foreground">
                ({getConfidenceText(result.confidence)})
              </span>
            </div> */}

            {/* Optional top-N display */}
            {/* {modelType === "species" && result.top3?.length ? (
              <div className="mt-3 text-xs text-muted-foreground">
                Top 3:{" "}
                {result.top3
                  .map((p) => `${p.nameTh} (${p.confidence}%)`)
                  .join(" • ")}
              </div>
            ) : null}

            {modelType === "venomous" && result.top2?.length ? (
              <div className="mt-3 text-xs text-muted-foreground">
                ตัวเลือก:{" "}
                {result.top2
                  .map((p) => `${p.nameTh} (${p.confidence}%)`)
                  .join(" • ")}
              </div>
            ) : null} */}
          </div>

          {/* Confidence Bar */}
          <div className="mb-6">
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
              <span>ความมั่นใจในการระบุ</span>
              <span>{result.confidence}%</span>
            </div>
            <div className="w-full bg-muted rounded-full h-2">
              <div
                className="bg-secondary h-2 rounded-full transition-all duration-700 ease-out"
                style={{ width: `${result.confidence}%` }}
              />
            </div>
          </div>

          {/* Image Thumbnail */}
          <div className="flex items-center space-x-3 p-3 bg-muted/30 rounded-lg">
            <img
              src={imageUrl}
              alt="Analyzed image"
              className="w-16 h-16 object-cover rounded-lg flex-shrink-0"
            />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground truncate">
                รูปภาพที่วิเคราะห์
              </p>
              <p className="text-xs text-muted-foreground truncate">
                {fileName}
              </p>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="space-y-3">
          <Button
            variant="outline"
            className="w-full btn-retry bg-gray-200"
            onClick={() => navigate("/")}
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            ลองใหม่อีกครั้ง
          </Button>
        </div>

        {/* Additional Info (Advice) */}
        <div className="card-secondary p-4" ref={adviceRef}>
          <h3 className="text-sm font-medium text-foreground mb-2">คำแนะนำ</h3>
          <div className="space-y-2 text-xs text-muted-foreground">
            {(modelType === "venomous" && result.name === "มีพิษ") ||
            modelType === "species" ? (
              <img src="/venimous-medic.png" alt="" />
            ) : (
              <img src="/non-venomous-medic.png" alt="" />
            )}
            <p>• หากพบงูจริง ควรรักษาระยะห่างและติดต่อหน่วยงานที่เกี่ยวข้อง</p>
            <p>• แอปพลิเคชันนี้ใช้สำหรับการศึกษาเท่านั้น</p>
          </div>
        </div>
      </div>

      {/* Floating Action Button (FAB) - bottom-right */}
      <button
        type="button"
        onClick={isAtAdvice ? handleScrollToTop : handleScrollToAdvice}
        aria-label={isAtAdvice ? "Scroll to top" : "Scroll to advice"}
        className="
          fixed bottom-4 right-4 z-50
          inline-flex items-center justify-center
          h-12 w-12 rounded-full shadow-lg
          bg-secondary text-secondary-foreground
          hover:opacity-90 transition
          focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-secondary
        "
      >
        {isAtAdvice ? (
          <ChevronUp className="h-6 w-6" />
        ) : (
          <ChevronDown className="h-6 w-6" />
        )}
      </button>
    </div>
  );
};

export default Result;
