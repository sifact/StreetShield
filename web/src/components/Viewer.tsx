import { useCallback, useRef, useState, type PointerEvent } from "react";
import type { BlurResult } from "../api";
import { labelColor } from "../labels";

export type ViewMode = "compare" | "blurred" | "original";

interface Props {
  originalUrl: string;
  result: BlurResult | undefined;
  busy: boolean;
  error: string | undefined;
  showBoxes: boolean;
  mode: ViewMode;
}

/** Boxes positioned in % of the image, so labels keep a fixed, readable pixel size at any zoom. */
function BoxOverlay({ result }: { result: BlurResult }) {
  return (
    <div className="boxes" aria-hidden>
      {result.detections.map((d, i) => {
        const [x, y, w, h] = d.box;
        return (
          <div
            key={i}
            className="box"
            style={{
              left: `${(x / result.width) * 100}%`,
              top: `${(y / result.height) * 100}%`,
              width: `${(w / result.width) * 100}%`,
              height: `${(h / result.height) * 100}%`,
              ["--c" as string]: labelColor(d.label),
            }}
          >
            <span className="box-label">
              {d.label} {Math.round(d.score * 100)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function Viewer({ originalUrl, result, busy, error, showBoxes, mode }: Props) {
  const [split, setSplit] = useState(50);
  const frame = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const move = useCallback((e: PointerEvent) => {
    const rect = frame.current?.getBoundingClientRect();
    if (!rect) return;
    setSplit(Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100)));
  }, []);

  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  // Size the frame from the image's real proportions, capped to 62% of the viewport height.
  const dims = result ? { w: result.width, h: result.height } : natural;
  const frameStyle = dims
    ? { aspectRatio: `${dims.w} / ${dims.h}`, width: `min(100%, calc(62vh * ${dims.w / dims.h}))` }
    : undefined;
  const showBlurred = result && mode !== "original";

  return (
    <div className="viewer">
      <div
        ref={frame}
        className={`frame ${mode === "compare" && result ? "comparing" : ""}`}
        style={frameStyle}
        onPointerDown={(e) => {
          if (mode !== "compare" || !result) return;
          dragging.current = true;
          (e.target as Element).setPointerCapture?.(e.pointerId);
          move(e);
        }}
        onPointerMove={(e) => dragging.current && move(e)}
        onPointerUp={() => (dragging.current = false)}
      >
        <img
          className="layer"
          src={originalUrl}
          alt="Original upload"
          draggable={false}
          onLoad={(e) => setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        />
        {showBlurred && (
          <img
            className="layer top"
            src={result.image}
            alt="Blurred result"
            draggable={false}
            style={mode === "compare" ? { clipPath: `inset(0 0 0 ${split}%)` } : undefined}
          />
        )}
        {result && showBoxes && <BoxOverlay result={result} />}
        {mode === "compare" && result && (
          <div className="divider" style={{ left: `${split}%` }}>
            <div className="handle">
              <svg viewBox="0 0 24 24" aria-hidden>
                <path d="m9 7-5 5 5 5M15 7l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <span className="tag left">Original</span>
            <span className="tag right">Blurred</span>
          </div>
        )}
        {busy && (
          <div className="veil">
            <div className="spinner" />
            <span>Detecting faces &amp; plates…</span>
          </div>
        )}
        {error && !busy && (
          <div className="veil error">
            <strong>Couldn’t process this image</strong>
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  );
}
