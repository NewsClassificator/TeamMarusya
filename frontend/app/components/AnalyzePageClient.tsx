"use client";

import React, { useEffect, useState } from "react";
import { AnalyzeForm } from "./AnalyzeForm";
import { AnalysisResult } from "./AnalysisResult";

type ClickbaitResult = {
  is_clickbait: boolean;
  score: number;
  label: string;
  confidence_note?: string | null;
};

export function AnalyzePageClient() {
  const [data, setData] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clickbaitState, setClickbaitState] = useState<
    "idle" | "loading" | "success" | "error"
  >("idle");
  const [clickbaitResult, setClickbaitResult] = useState<ClickbaitResult | null>(
    null,
  );
  const [clickbaitError, setClickbaitError] = useState<string | null>(null);
  const [clickbaitCache, setClickbaitCache] = useState<
    Record<string, ClickbaitResult>
  >({});
  const [lastHeadline, setLastHeadline] = useState<string | null>(null);

  async function requestClickbait(headline: string) {
    const cached = clickbaitCache[headline];
    if (cached) {
      setClickbaitResult(cached);
      setClickbaitState("success");
      setClickbaitError(null);
      return;
    }

    setClickbaitState("loading");
    setClickbaitError(null);

    try {
      const res = await fetch("/api/clickbait", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ headline }),
      });

      const payload = await res.json();

      if (!res.ok) {
        const message =
          payload?.message ??
          payload?.detail?.message ??
          "Статус недоступен, попробуйте позже.";
        setClickbaitState("error");
        setClickbaitError(message);
        setClickbaitResult(null);
        return;
      }

      const mapped = {
        is_clickbait: !!payload?.is_clickbait,
        score: Number(payload?.score ?? 0),
        label: payload?.label ?? "",
        confidence_note: payload?.confidence_note ?? null,
      };

      setClickbaitResult(mapped);
      setClickbaitState("success");
      setClickbaitError(null);
      setClickbaitCache((prev) => ({ ...prev, [headline]: mapped }));
    } catch (err) {
      setClickbaitState("error");
      setClickbaitError("Не удалось получить статус кликбейта.");
      setClickbaitResult(null);
    }
  }

  useEffect(() => {
    if (!data?.article?.title) {
      setClickbaitState("idle");
      setClickbaitResult(null);
      setLastHeadline(null);
      return;
    }

    const headline = data.article.title.trim();
    if (!headline) return;

    if (headline === lastHeadline && clickbaitState === "success") {
      return;
    }

    setLastHeadline(headline);
    requestClickbait(headline);
  }, [data]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex min-h-screen w-full max-w-3xl flex-col items-start justify-start py-16 px-6 bg-white dark:bg-black gap-8">
        <header className="space-y-2">
          <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">
            Анализ новостных статей
          </h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            Вставьте ссылку или текст новости, чтобы получить структурированный
            вывод и оценку эмоциональной окраски.
          </p>
        </header>

        <AnalyzeForm
          onResult={(result, err) => {
            setData(result);
            setError(err);
            setClickbaitState("idle");
            setClickbaitResult(null);
            setClickbaitError(null);
          }}
        />

        <AnalysisResult
          data={data}
          error={error}
          clickbaitState={clickbaitState}
          clickbaitResult={clickbaitResult}
          clickbaitError={clickbaitError}
        />
      </main>
    </div>
  );
}
