"use client";

type WaterResult = {
  is_water: boolean;
  label: string;
  confidence: number;
  interpretations?: Record<string, string>;
  errors?: string[];
};

type WaterBadgeProps = {
  state: "idle" | "loading" | "success" | "error";
  result?: WaterResult | null;
  error?: string | null;
};

const baseClasses =
  "inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium border transition";

function formatConfidence(confidence: number) {
  const pct = Math.round(confidence * 100);
  return `${pct}%`;
}

function renderInterpretationHint(interpretations?: Record<string, string>) {
  if (!interpretations) return null;
  const values = Object.values(interpretations).filter(Boolean);
  if (values.length === 0) return null;
  const hint = values.slice(0, 2).join(" • ");
  return <span className="text-[11px] text-zinc-600">{hint}</span>;
}

export function WaterBadge({ state, result, error }: WaterBadgeProps) {
  if (state === "idle") return null;

  if (state === "loading") {
    return (
      <span
        className={`${baseClasses} border-zinc-300 bg-white text-zinc-600`}
        aria-live="polite"
      >
        Проверяем воду…
      </span>
    );
  }

  if (state === "error") {
    return (
      <span
        className={`${baseClasses} border-amber-300 bg-amber-50 text-amber-800`}
        role="status"
        aria-live="polite"
      >
        {error ?? "Статус недоступен"}
      </span>
    );
  }

  if (!result) return null;

  const tone = result.is_water
    ? "border-blue-300 bg-blue-50 text-blue-900"
    : "border-emerald-300 bg-emerald-50 text-emerald-800";

  return (
    <span
      className={`${baseClasses} ${tone}`}
      role="status"
      aria-live="polite"
      aria-label={`Вода: ${result.is_water ? "да" : "нет"}`}
    >
      <span>{result.label || (result.is_water ? "ВОДА" : "НЕ ВОДА")}</span>
      <span className="text-[11px] text-zinc-600">
        {formatConfidence(result.confidence)}
      </span>
      {renderInterpretationHint(result.interpretations)}
    </span>
  );
}
