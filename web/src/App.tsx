import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { blurImage, fetchModelInfo, sameParams, type BlurParams, type ModelInfo } from "./api";
import { DetectionsTable } from "./components/DetectionsTable";
import { Dropzone } from "./components/Dropzone";
import { EmptyState } from "./components/EmptyState";
import { Filmstrip } from "./components/Filmstrip";
import { ModelCard } from "./components/ModelCard";
import { SettingsPanel } from "./components/SettingsPanel";
import { TopBar } from "./components/TopBar";
import { Viewer, type ViewMode } from "./components/Viewer";
import { useDebounced } from "./hooks/useDebounced";
import { labelColor } from "./labels";
import type { Item } from "./types";

const RETRY_MS = 4000;
const VIEW_MODES: { id: ViewMode; label: string }[] = [
  { id: "compare", label: "Compare" },
  { id: "blurred", label: "Blurred" },
  { id: "original", label: "Original" },
];

let nextId = 0;
const makeItem = (file: File): Item => ({ id: `img-${nextId++}`, file, url: URL.createObjectURL(file), status: "queued" });

const blurredName = (name: string) => `${name.replace(/\.[^.]+$/, "")}-blurred.jpg`;

/** An item needs (re)processing when it has no result for the current settings and didn't already fail with them. */
const needsWork = (item: Item, params: BlurParams) =>
  item.status !== "processing" &&
  !(item.result && sameParams(item.result.params, params)) &&
  !(item.errorParams && sameParams(item.errorParams, params));

export default function App() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [online, setOnline] = useState(false);
  const [params, setParams] = useState<BlurParams | null>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const [mode, setMode] = useState<ViewMode>("compare");
  const inflight = useRef<string | null>(null);
  const debouncedParams = useDebounced(params, 350);

  // Connect to the API, retrying until it is up (the model can take a few seconds to load).
  useEffect(() => {
    if (online) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const attempt = () =>
      fetchModelInfo()
        .then((m) => {
          if (cancelled) return;
          setInfo(m);
          setOnline(true);
          setParams((p) => p ?? m.defaults);
        })
        .catch(() => {
          if (!cancelled) timer = setTimeout(attempt, RETRY_MS);
        });
    attempt();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [online]);

  const addFiles = useCallback((files: File[]) => {
    const added = files.map(makeItem);
    setItems((prev) => [...prev, ...added]);
    setSelectedId((cur) => cur ?? added[0]?.id ?? null);
  }, []);

  // Paste screenshots / copied images straight into the page.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const files = Array.from(e.clipboardData?.files ?? []).filter((f) => f.type.startsWith("image/"));
      if (files.length) addFiles(files);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [addFiles]);

  // One request at a time: the selected image first, then the rest of the queue.
  useEffect(() => {
    if (!online || !debouncedParams || inflight.current) return;
    const target =
      items.find((it) => it.id === selectedId && needsWork(it, debouncedParams)) ??
      items.find((it) => needsWork(it, debouncedParams));
    if (!target) return;

    const p = debouncedParams;
    inflight.current = target.id;
    const update = (patch: Partial<Item>) =>
      setItems((prev) => prev.map((it) => (it.id === target.id ? { ...it, ...patch } : it)));
    update({ status: "processing" });

    blurImage(target.file, p)
      .then((result) => {
        inflight.current = null;
        update({ status: "done", result, error: undefined, errorParams: undefined });
      })
      .catch((err: Error) => {
        inflight.current = null;
        if (err instanceof TypeError) setOnline(false); // network failure: API went away
        update({ status: "error", error: err.message, errorParams: p });
      });
  }, [items, selectedId, debouncedParams, online]);

  const removeItem = (id: string) => {
    const gone = items.find((it) => it.id === id);
    if (gone) URL.revokeObjectURL(gone.url);
    if (id === selectedId) setSelectedId(items.find((it) => it.id !== id)?.id ?? null);
    setItems((prev) => prev.filter((it) => it.id !== id)); // functional: keeps results that land meanwhile
  };

  const clearAll = () => {
    items.forEach((it) => URL.revokeObjectURL(it.url));
    setItems([]);
    setSelectedId(null);
  };

  const selected = items.find((it) => it.id === selectedId) ?? null;
  const result = selected?.result;
  const stale = !!(result && params && !sameParams(result.params, params));

  const totals = useMemo(() => {
    const done = items.filter((it) => it.result);
    const regions = done.reduce((n, it) => n + (it.result?.detections.length ?? 0), 0);
    return { done: done.length, regions };
  }, [items]);

  return (
    <div className="app">
      <TopBar info={info} online={online} />

      <div className="layout">
        <aside className="sidebar">
          {params && (
            <SettingsPanel
              params={params}
              defaults={info?.defaults ?? null}
              onChange={setParams}
              showBoxes={showBoxes}
              onShowBoxes={setShowBoxes}
            />
          )}
          <ModelCard info={info} />
        </aside>

        <main className="main">
          {!selected ? (
            <EmptyState onFiles={addFiles} online={online} />
          ) : (
            <>
              <div className="toolbar">
                <div className="segmented">
                  {VIEW_MODES.map((m) => (
                    <button key={m.id} className={mode === m.id ? "active" : ""} onClick={() => setMode(m.id)}>
                      {m.label}
                    </button>
                  ))}
                </div>
                <div className="toolbar-stats">
                  {result &&
                    Object.entries(result.counts).map(([label, n]) => (
                      <span key={label} className="stat">
                        <span className="swatch" style={{ background: labelColor(label) }} />
                        {n} {label}
                        {n === 1 ? "" : "s"}
                      </span>
                    ))}
                  {result && <span className="stat muted mono">{Math.round(result.elapsed_ms)} ms</span>}
                  {stale && <span className="stat warn">updating…</span>}
                </div>
                <a
                  className={`btn primary ${result ? "" : "disabled"}`}
                  href={result?.image}
                  download={selected ? blurredName(selected.file.name) : undefined}
                  aria-disabled={!result}
                >
                  <svg viewBox="0 0 24 24" aria-hidden>
                    <path d="M12 4v12m0 0 4-4m-4 4-4-4M4 20h16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Download
                </a>
              </div>

              <Viewer
                originalUrl={selected.url}
                result={result}
                busy={selected.status === "processing" || (selected.status === "queued" && !result)}
                error={selected.status === "error" ? selected.error : undefined}
                showBoxes={showBoxes}
                mode={mode}
              />

              <div className="below">
                <div className="file-meta">
                  <span className="filename" title={selected.file.name}>
                    {selected.file.name}
                  </span>
                  {result && (
                    <span className="muted mono small">
                      {result.original_width}×{result.original_height}
                      {result.width !== result.original_width && ` → ${result.width}×${result.height}`}
                    </span>
                  )}
                </div>
                {result && <DetectionsTable result={result} />}
              </div>

              <div className="strip-row">
                <Filmstrip items={items} selectedId={selectedId} onSelect={setSelectedId} onRemove={removeItem} />
                <div className="strip-side">
                  <Dropzone onFiles={addFiles} compact />
                  <div className="strip-meta muted small">
                    {totals.done}/{items.length} processed · {totals.regions} regions blurred
                    <button className="link" onClick={clearAll}>
                      Clear all
                    </button>
                  </div>
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
