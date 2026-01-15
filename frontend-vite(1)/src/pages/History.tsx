import { useNavigate } from "react-router-dom";
import { ArrowLeft, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HistoryItem {
  id: string;
  snakeName: string;
  timestamp: string;
  modelType: "species" | "venomous";
}

const History = () => {
  const navigate = useNavigate();

  // Get history from localStorage
  const historyData = localStorage.getItem("snakeHistory");
  const history: HistoryItem[] = historyData ? JSON.parse(historyData) : [];

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString("th-TH", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  // helper: return Tailwind classes per model type
  const badgeClass = (modelType: "species" | "venomous") =>
    modelType === "species"
      ? "text-xs px-2 py-1 rounded-full ring-1 bg-emerald-600/10 text-emerald-700 dark:text-emerald-300 ring-emerald-500/30"
      : "text-xs px-2 py-1 rounded-full ring-1 bg-blue-600/10 text-blue-700 dark:text-blue-300 ring-blue-500/30";

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-md mx-auto space-y-6">
        {/* Header */}
        <div className="relative text-center pt-8 pb-4">
          <Button
            variant="outline"
            size="sm"
            className="absolute left-0 top-8"
            onClick={() => navigate("/")}
          >
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <h1 className="text-2xl font-semibold text-foreground mb-2">
            ประวัติการวิเคราะห์
          </h1>
          <p className="text-muted-foreground text-sm">
            รายการงูที่เคยวิเคราะห์
          </p>
        </div>

        {/* History List */}
        <div className="space-y-3">
          {history.length === 0 ? (
            <div className="card-primary p-8 text-center">
              <p className="text-muted-foreground">
                ยังไม่มีประวัติการวิเคราะห์
              </p>
            </div>
          ) : (
            history.map((item) => (
              <div
                key={item.id}
                className="card-primary p-4 hover:shadow-lg transition-shadow"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-foreground mb-1">
                      {item.snakeName}
                    </h3>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Clock className="h-3.5 w-3.5" />
                      <span>{formatTime(item.timestamp)}</span>
                    </div>
                  </div>
                  <span
                    className={badgeClass(item.modelType)}
                    title={
                      item.modelType === "species"
                        ? "วิเคราะห์ชนิดงู"
                        : "วิเคราะห์มีพิษ/ไม่มีพิษ"
                    }
                  >
                    {item.modelType === "species" ? "ชนิดงู" : "ประเภทงู"}
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default History;
