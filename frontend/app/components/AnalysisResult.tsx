import { ClickbaitBadge } from "./ClickbaitBadge";

type ClickbaitResult = {
  is_clickbait: boolean;
  score: number;
  label: string;
  confidence_note?: string | null;
};

interface AnalysisResultProps {
  data: any | null;
  error: string | null;
  clickbaitState?: "idle" | "loading" | "success" | "error";
  clickbaitResult?: ClickbaitResult | null;
  clickbaitError?: string | null;
}

export function AnalysisResult({
  data,
  error,
  clickbaitState = "idle",
  clickbaitResult,
  clickbaitError,
}: AnalysisResultProps) {
  if (error) {
    return (
      <div className="mt-6 rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900">
        {error}
      </div>
    );
  }

  if (!data) {
    return null;
  }

  const { article, sentiment, contract_version, model_version, seed } = data;
  const headline = article?.title;

  return (
    <div className="mt-6 w-full space-y-4">
      <div className="rounded-md border border-zinc-200 bg-zinc-50 px-4 py-3">
        <p className="text-xs text-zinc-500">
          contract_version: {contract_version} • model_version: {model_version} •
          seed: {seed}
        </p>
      </div>

      <div className="space-y-2">
        {headline && (
          <div className="flex items-center gap-3 flex-wrap">
            <h2 className="text-xl font-semibold text-zinc-900">{headline}</h2>
            <ClickbaitBadge
              state={clickbaitState}
              result={clickbaitResult ?? undefined}
              error={clickbaitError ?? undefined}
            />
          </div>
        )}
        <div className="text-sm text-zinc-600">
          {article?.author && <span>Автор: {article.author}</span>}
          {article?.published_at && (
            <span className="ml-3">
              Дата:{" "}
              {new Date(article.published_at).toLocaleString("ru-RU", {
                dateStyle: "short",
                timeStyle: "short",
              })}
            </span>
          )}
        </div>
      </div>

      <div className="rounded-md border border-zinc-200 bg-white px-4 py-3">
        <p className="text-sm font-medium text-zinc-800 mb-2">
          Эмоциональная окраска:{" "}
          <span className="uppercase">{sentiment?.label}</span> (
          {(sentiment?.score * 100).toFixed(1)}%)
        </p>
        {article?.content && (
          <p className="whitespace-pre-wrap text-sm text-zinc-700">
            {article.content}
          </p>
        )}
      </div>
    </div>
  );
}
