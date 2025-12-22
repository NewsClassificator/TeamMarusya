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

type WaterResult = {
  is_water: boolean;
  label: string;
  confidence: number;
  features?: Record<string, number>;
  interpretations?: Record<string, string>;
  errors?: string[];
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
  const [waterState, setWaterState] = useState<
    "idle" | "loading" | "success" | "error"
  >("idle");
  const [waterResult, setWaterResult] = useState<WaterResult | null>(null);
  const [waterError, setWaterError] = useState<string | null>(null);
  const [waterCache, setWaterCache] = useState<Record<string, WaterResult>>({});
  const [lastWaterText, setLastWaterText] = useState<string | null>(null);

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

  async function requestWater(text: string) {
    const cached = waterCache[text];
    if (cached) {
      setWaterResult(cached);
      setWaterState("success");
      setWaterError(null);
      return;
    }

    setWaterState("loading");
    setWaterError(null);

    try {
      const res = await fetch("/api/water-detection", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text, include_features: true }),
      });

      const payload = await res.json();

      if (!res.ok) {
        const message =
          payload?.message ??
          payload?.detail?.message ??
          "Статус воды недоступен, попробуйте позже.";
        setWaterState("error");
        setWaterError(message);
        setWaterResult(null);
        return;
      }

      const mapped: WaterResult = {
        is_water: !!payload?.is_water,
        label: payload?.label ?? "",
        confidence: Number(payload?.confidence ?? 0),
        features: payload?.features ?? undefined,
        interpretations: payload?.interpretations ?? undefined,
        errors: payload?.errors ?? undefined,
      };

      setWaterResult(mapped);
      setWaterState("success");
      setWaterError(null);
      setWaterCache((prev) => ({ ...prev, [text]: mapped }));
    } catch (err) {
      setWaterState("error");
      setWaterError("Не удалось получить статус воды.");
      setWaterResult(null);
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

  useEffect(() => {
    const text = data?.article?.content?.trim?.();

    if (!text) {
      setWaterState("idle");
      setWaterResult(null);
      setWaterError(null);
      setLastWaterText(null);
      return;
    }

    if (text.length < 20) {
      setWaterState("error");
      setWaterError("Текст слишком короткий для оценки воды.");
      setWaterResult(null);
      setLastWaterText(text);
      return;
    }

    if (text === lastWaterText && waterState === "success") {
      return;
    }

    setLastWaterText(text);
    requestWater(text);
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
          waterState={waterState}
          waterResult={waterResult}
          waterError={waterError}
        />
      </main>
    </div>
  );
}
