"use client";

import { useState } from "react";

type InputType = "url" | "text";

export interface AnalyzeFormProps {
  onResult: (data: any | null, error: string | null) => void;
}

export function AnalyzeForm({ onResult }: AnalyzeFormProps) {
  const [inputType, setInputType] = useState<InputType>("url");
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (isSubmitting) return;

    const trimmedUrl = url.trim();
    const trimmedText = text.trim();

    const payload: any = {
      input_type: inputType,
      language: "ru",
    };

    if (inputType === "url") {
      payload.url = trimmedUrl;
    } else {
      payload.text = trimmedText;
    }

    setIsSubmitting(true);
    onResult(null, null);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        const message =
          data?.message ??
          data?.detail?.message ??
          "Произошла ошибка при анализе.";
        onResult(null, message);
      } else {
        onResult(data, null);
      }
    } catch (err) {
      onResult(null, "Не удалось связаться с сервером.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-full">
      <div className="flex gap-3">
        <button
          type="button"
          onClick={() => setInputType("url")}
          className={`rounded-full px-4 py-2 text-sm font-medium border ${
            inputType === "url"
              ? "bg-black text-white border-black"
              : "bg-white text-black border-zinc-300"
          }`}
        >
          URL
        </button>
        <button
          type="button"
          onClick={() => setInputType("text")}
          className={`rounded-full px-4 py-2 text-sm font-medium border ${
            inputType === "text"
              ? "bg-black text-white border-black"
              : "bg-white text-black border-zinc-300"
          }`}
        >
          Текст
        </button>
      </div>

      {inputType === "url" ? (
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Вставьте ссылку на новость"
          className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
          required
        />
      ) : (
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Вставьте текст новости"
          className="w-full min-h-[140px] rounded-md border border-zinc-300 px-3 py-2 text-sm"
          required
        />
      )}

      <button
        type="submit"
        disabled={isSubmitting}
        className="inline-flex items-center justify-center rounded-full bg-black px-5 py-2 text-sm font-medium text-white disabled:opacity-60"
      >
        {isSubmitting ? "Анализируем..." : "Анализировать"}
      </button>
    </form>
  );
}

