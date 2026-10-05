import { useRef, useState, type DragEvent } from "react";

interface Props {
  onFiles: (files: File[]) => void;
  compact?: boolean;
}

const isImage = (f: File) => f.type.startsWith("image/");

export function Dropzone({ onFiles, compact = false }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const accept = (list: FileList | null) => {
    const files = Array.from(list ?? []).filter(isImage);
    if (files.length) onFiles(files);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    accept(e.dataTransfer.files);
  };

  return (
    <div
      className={`dropzone ${over ? "over" : ""} ${compact ? "compact" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      onClick={() => input.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
    >
      <input
        ref={input}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => {
          accept(e.target.files);
          e.target.value = "";
        }}
      />
      <svg viewBox="0 0 24 24" className="dz-icon" aria-hidden>
        <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div className="dz-text">
        <strong>Drop images</strong> or click to browse
        {!compact && <div className="dz-hint">JPG, PNG, WebP · up to 20 MB · paste with ⌘V</div>}
      </div>
    </div>
  );
}
