import { ClickbaitBadge } from "./ClickbaitBadge";
import { WaterBadge } from "./WaterBadge";

type FreshnessStatus = "today" | "yesterday" | "recent" | "stale" | "unknown";

type Freshness = {
  status: FreshnessStatus;
  age_days: number | null;
  reference_date: string;
  message: string;
  source_date?: string | null;
};

type SentimentSummary = {
  text: string;
  sentiment_label: "positive" | "neutral" | "negative";
  confidence: number;
};

type QuoteSentiment = {
  quote_text: string;
  sentiment_label: "positive" | "neutral" | "negative";
  confidence: number;
  position: number;
  author?: string | null;
};

type SentimentResult = {
  main_text: SentimentSummary;
  quotes: QuoteSentiment[];
  errors?: string[];
};

type ArticleContent = {
  title?: string | null;
  author?: string | null;
  published_at?: string | null;
  content: string;
};

type Meta = {
  contract_version: string;
  analysis_version: string;
  analyzed_at: string;
  seed: number;
};

type AnalysisResponse = {
  request_id?: string | null;
  article: ArticleContent;
  freshness: Freshness;
  sentiment: SentimentResult;
  meta: Meta;
  errors?: string[];
};

type ClickbaitResult = {
  is_clickbait: boolean;
  score: number;
  label: string;
  confidence_note?: string | null;
};

type WaterResult = {
  is_water: boolean;
  label: string;
  confidence: number;
  features?: Record<string, number>;
  interpretations?: Record<string, string>;
  errors?: string[];
};

interface AnalysisResultProps {
  data: AnalysisResponse | null;
  error: string | null;
  clickbaitState?: "idle" | "loading" | "success" | "error";
  clickbaitResult?: ClickbaitResult | null;
  clickbaitError?: string | null;
  waterState?: "idle" | "loading" | "success" | "error";
  waterResult?: WaterResult | null;
  waterError?: string | null;
}

function formatFreshnessLabel(freshness: Freshness) {
  if (freshness.status === "unknown") return "Дата недоступна";
  return freshness.message;
}

function formatPublishedDate(publishedAt?: string | null, fallback?: string) {
  if (!publishedAt) {
    return fallback ?? null;
  }
  const parsed = new Date(publishedAt);
  if (Number.isNaN(parsed.getTime())) {
    return fallback ?? publishedAt;
  }
  return parsed.toLocaleString("ru-RU", {
    dateStyle: "short",
    timeStyle: "short",
  });
}

export function AnalysisResult({
  data,
  error,
  clickbaitState = "idle",
  clickbaitResult,
  clickbaitError,
  waterState = "idle",
  waterResult,
  waterError,
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

  const { article, sentiment, freshness, meta, errors } = data;
  const headline = article?.title;

  return (
    <div className="mt-6 w-full space-y-4">
      <div className="rounded-md border border-zinc-200 bg-zinc-50 px-4 py-3">
        <p className="text-xs text-zinc-500">
          contract_version: {meta.contract_version} • analysis_version:{" "}
          {meta.analysis_version} • seed: {meta.seed}
        </p>
      </div>

      {errors && errors.length > 0 && (
        <div className="rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          {errors.map((err, idx) => (
            <div key={idx}>{err}</div>
          ))}
        </div>
      )}

      <div className="space-y-2">
        {headline && (
          <div className="flex items-center gap-3 flex-wrap">
            <h2 className="text-xl font-semibold text-zinc-900">{headline}</h2>
            <ClickbaitBadge
              state={clickbaitState}
              result={clickbaitResult ?? undefined}
              error={clickbaitError ?? undefined}
            />
            <WaterBadge
              state={waterState}
              result={waterResult ?? undefined}
              error={waterError ?? undefined}
            />
          </div>
        )}
        <div className="text-sm text-zinc-600 space-y-1">
          <div>
            {article?.author && <span>Автор: {article.author}</span>}
            <span className="ml-3">
              Дата:{" "}
              {formatPublishedDate(
                freshness.source_date ?? article?.published_at ?? null,
                article?.published_at ?? undefined,
              ) ?? "Не указана"}
            </span>
          </div>
          <div>
            Свежесть:{" "}
            <span className="font-medium text-zinc-900">
              {formatFreshnessLabel(freshness)}
            </span>
          </div>
        </div>
      </div>

      <div className="rounded-md border border-zinc-200 bg-white px-4 py-3 space-y-3">
        <p className="text-sm font-medium text-zinc-800">
          Основной текст:{" "}
          <span className="uppercase">{sentiment.main_text.sentiment_label}</span>{" "}
          ({(sentiment.main_text.confidence * 100).toFixed(1)}%)
        </p>
        {article?.content && (
          <p className="whitespace-pre-wrap text-sm text-zinc-700">
            {article.content}
          </p>
        )}
      </div>

      {waterState !== "idle" && (
        <div className="rounded-md border border-zinc-200 bg-white px-4 py-3 space-y-2">
          <p className="text-sm font-medium text-zinc-800">Водность текста</p>

          {waterState === "loading" && (
            <p className="text-sm text-zinc-600">Проверяем воду…</p>
          )}

          {waterState === "error" && (
            <p className="text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded px-3 py-2">
              {waterError ?? "Статус недоступен"}
            </p>
          )}

          {waterState === "success" && waterResult && (
            <div className="space-y-2">
              <p className="text-sm text-zinc-800 flex items-center gap-2">
                <span className="font-semibold">
                  {waterResult.label || (waterResult.is_water ? "ВОДА" : "НЕ ВОДА")}
                </span>
                <span className="text-xs text-zinc-600">
                  {(waterResult.confidence * 100).toFixed(0)}%
                </span>
              </p>
              {waterResult.interpretations && (
                <ul className="text-sm text-zinc-700 space-y-1">
                  {Object.entries(waterResult.interpretations).map(
                    ([key, value]) =>
                      value ? (
                        <li key={key} className="flex gap-2">
                          <span className="text-xs uppercase text-zinc-500">{key}:</span>
                          <span>{value}</span>
                        </li>
                      ) : null,
                  )}
                </ul>
              )}
              {waterResult.features && (
                <div className="text-xs text-zinc-600 grid grid-cols-2 gap-x-4 gap-y-1">
                  {Object.entries(waterResult.features).map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="uppercase">{k}</span>
                      <span className="font-medium text-zinc-800">
                        {typeof v === "number" ? v.toFixed(3) : String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <div className="rounded-md border border-zinc-200 bg-white px-4 py-3 space-y-2">
        <p className="text-sm font-medium text-zinc-800">
          Цитаты и их тональность
        </p>
        {sentiment.quotes.length === 0 ? (
          <p className="text-sm text-zinc-600">Цитаты отсутствуют.</p>
        ) : (
          <ul className="space-y-2">
            {sentiment.quotes.map((quote) => (
              <li
                key={quote.position}
                className="rounded border border-zinc-200 bg-zinc-50 px-3 py-2"
              >
                <p className="text-sm text-zinc-700 italic">
                  “{quote.quote_text}”
                </p>
                {quote.author && (
                  <p className="text-xs text-zinc-600 mt-1">
                    Автор: {quote.author}
                  </p>
                )}
                <p className="text-xs text-zinc-600 mt-1">
                  Тон: <span className="uppercase">{quote.sentiment_label}</span>{" "}
                  ({(quote.confidence * 100).toFixed(1)}%)
                </p>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
