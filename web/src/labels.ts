// One place for class colours so boxes, badges and tables agree.
export const LABEL_COLORS: Record<string, string> = {
  face: "var(--face)",
  plate: "var(--plate)",
};

export const labelColor = (label: string) => LABEL_COLORS[label] ?? "var(--accent)";

export const pct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(digits)}%`;
