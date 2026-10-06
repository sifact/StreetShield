import type { ModelInfo } from "../api";
import { labelColor, pct } from "../labels";

export function ModelCard({ info }: { info: ModelInfo | null }) {
  if (!info) return null;
  const test = info.test;
  const history = info.history;
  const maxMap = Math.max(...history.map((h) => h.val.mAP50), 1e-6);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Model</h2>
        <span className="muted mono small ellipsis" title={info.arch}>
          {info.arch.replace("fasterrcnn_", "frcnn_")}
        </span>
      </div>
      {test ? (
        <table className="metrics">
          <thead>
            <tr>
              <th />
              <th title="Average precision at IoU 0.5">AP50</th>
              <th title="Fraction of objects found (IoU 0.5, score ≥ 0.5)">Recall</th>
              <th title="Fraction of detections that are correct">Prec.</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(test.per_class).map(([name, m]) => (
              <tr key={name}>
                <td>
                  <span className="swatch" style={{ background: labelColor(name) }} />
                  {name}
                </td>
                <td className="mono">{pct(m.AP50)}</td>
                <td className="mono">{pct(m["recall@0.5"])}</td>
                <td className="mono">{pct(m["precision@0.5"])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="muted small">No test metrics yet. Run `streetshield evaluate`.</p>
      )}
      {history.length > 1 && (
        <div className="spark" title="Validation mAP@0.5 per epoch">
          {history.map((h) => (
            <div
              key={h.epoch}
              className={`bar ${h.epoch === info.checkpoint_epoch ? "best" : ""}`}
              style={{ height: `${(h.val.mAP50 / maxMap) * 100}%` }}
              title={`epoch ${h.epoch}: ${pct(h.val.mAP50)}`}
            />
          ))}
        </div>
      )}
      {test && <p className="muted small">Held-out test set · {test.num_images} images · val mAP per epoch above</p>}
    </section>
  );
}
