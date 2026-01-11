"use client";

type ClickbaitResult = {
  is_clickbait: boolean;
  score: number;
  label: string;
  confidence_note?: string | null;
};

type ClickbaitBadgeProps = {
  state: "idle" | "loading" | "success" | "error";
  result?: ClickbaitResult | null;
  error?: string | null;
};

const baseClasses =
  "inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium border transition";

export function ClickbaitBadge({ state, result, error }: ClickbaitBadgeProps) {
  if (state === "idle") return null;

  if (state === "loading") {
    return (
      <span
        className={`${baseClasses} border-zinc-300 bg-white text-zinc-600`}
        aria-live="polite"
      >
        Проверяем кликбейт…
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

  const tone = result.is_clickbait
    ? "border-red-300 bg-red-50 text-red-800"
    : "border-emerald-300 bg-emerald-50 text-emerald-800";

  return (
    <span
      className={`${baseClasses} ${tone}`}
      role="status"
      aria-live="polite"
      aria-label={`Кликбейт: ${result.is_clickbait ? "да" : "нет"}`}
    >
      <span>{result.is_clickbait ? "Кликбейт" : "Не кликбейт"}</span>
      <span className="text-[11px] text-zinc-600">
        {(result.score * 100).toFixed(0)}%
      </span>
      {result.confidence_note && (
        <span className="text-[11px] text-zinc-500">
          {result.confidence_note}
        </span>
      )}
    </span>
  );
}
