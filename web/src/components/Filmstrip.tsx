import type { Item } from "../types";

interface Props {
  items: Item[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
}

export function Filmstrip({ items, selectedId, onSelect, onRemove }: Props) {
  return (
    <div className="filmstrip" role="listbox" aria-label="Uploaded images">
      {items.map((item) => {
        const total = item.result ? item.result.detections.length : null;
        return (
          <div
            key={item.id}
            role="option"
            aria-selected={item.id === selectedId}
            className={`thumb ${item.id === selectedId ? "selected" : ""} ${item.status}`}
            onClick={() => onSelect(item.id)}
            title={item.file.name}
          >
            <img src={item.result?.image ?? item.url} alt={item.file.name} />
            <span className="thumb-badge">
              {item.status === "processing" ? <span className="mini-spinner" /> : item.status === "error" ? "!" : total ?? "…"}
            </span>
            <button
              className="thumb-remove"
              aria-label={`Remove ${item.file.name}`}
              onClick={(e) => {
                e.stopPropagation();
                onRemove(item.id);
              }}
            >
              ×
            </button>
          </div>
        );
      })}
    </div>
  );
}
