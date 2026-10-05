import type { BlurMethod, BlurParams } from "../api";

interface Props {
  params: BlurParams;
  defaults: BlurParams | null;
  onChange: (p: BlurParams) => void;
  showBoxes: boolean;
  onShowBoxes: (v: boolean) => void;
}

interface SliderProps {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
  onChange: (v: number) => void;
}

function Slider({ label, hint, value, min, max, step, format = (v) => v.toFixed(2), onChange }: SliderProps) {
  const fill = ((value - min) / (max - min)) * 100;
  return (
    <label className="field">
      <div className="field-head">
        <span>{label}</span>
        <span className="mono value">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ ["--fill" as string]: `${fill}%` }}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="field-hint">{hint}</div>
    </label>
  );
}

const METHODS: { id: BlurMethod; label: string }[] = [
  { id: "gaussian", label: "Gaussian" },
  { id: "pixelate", label: "Pixelate" },
];

export function SettingsPanel({ params, defaults, onChange, showBoxes, onShowBoxes }: Props) {
  const set = <K extends keyof BlurParams>(key: K, value: BlurParams[K]) => onChange({ ...params, [key]: value });

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Settings</h2>
        {defaults && (
          <button className="link" onClick={() => onChange(defaults)}>
            Reset
          </button>
        )}
      </div>

      <div className="field">
        <div className="field-head">
          <span>Blur style</span>
        </div>
        <div className="segmented" role="radiogroup">
          {METHODS.map((m) => (
            <button
              key={m.id}
              role="radio"
              aria-checked={params.method === m.id}
              className={params.method === m.id ? "active" : ""}
              onClick={() => set("method", m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <Slider
        label="Confidence threshold"
        hint="Lower blurs more. Safer for privacy, more false positives."
        value={params.score_thresh}
        min={0.05}
        max={0.95}
        step={0.05}
        onChange={(v) => set("score_thresh", v)}
      />
      <Slider
        label="Box padding"
        hint="Grows each box so hair and plate edges are covered."
        value={params.box_padding}
        min={0}
        max={0.5}
        step={0.05}
        format={(v) => `${Math.round(v * 100)}%`}
        onChange={(v) => set("box_padding", v)}
      />
      <Slider
        label="NMS overlap (IoU)"
        hint="Boxes overlapping more than this are merged into one."
        value={params.nms_iou}
        min={0.1}
        max={0.9}
        step={0.05}
        onChange={(v) => set("nms_iou", v)}
      />

      <label className="toggle">
        <input type="checkbox" checked={showBoxes} onChange={(e) => onShowBoxes(e.target.checked)} />
        <span className="track" />
        <span>Show detection boxes</span>
      </label>
    </section>
  );
}
