import type { BlurResult } from "../api";
import { labelColor } from "../labels";

export function DetectionsTable({ result }: { result: BlurResult }) {
  if (!result.detections.length) {
    return (
      <div className="empty-dets">
        No faces or plates found above the {Math.round(result.params.score_thresh * 100)}% threshold. Lower it if
        something was missed.
      </div>
    );
  }
  return (
    <div className="table-wrap">
      <table className="dets">
        <thead>
          <tr>
            <th>#</th>
            <th>Class</th>
            <th>Confidence</th>
            <th className="right col-box">Box (x, y, w, h)</th>
          </tr>
        </thead>
        <tbody>
          {result.detections.map((d, i) => (
            <tr key={i}>
              <td className="muted mono">{i + 1}</td>
              <td>
                <span className="pill" style={{ color: labelColor(d.label), borderColor: labelColor(d.label) }}>
                  {d.label}
                </span>
              </td>
              <td>
                <div className="score">
                  <div className="score-bar">
                    <div style={{ width: `${d.score * 100}%`, background: labelColor(d.label) }} />
                  </div>
                  <span className="mono">{(d.score * 100).toFixed(1)}%</span>
                </div>
              </td>
              <td className="mono right muted col-box">{d.box.map((v) => Math.round(v)).join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
