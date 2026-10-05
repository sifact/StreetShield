import type { ModelInfo } from "../api";
import { pct } from "../labels";

interface Props {
  info: ModelInfo | null;
  online: boolean;
}

export function TopBar({ info, online }: Props) {
  return (
    <header className="topbar">
      <div className="brand">
        <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden>
          <circle cx="16" cy="16" r="11" fill="none" stroke="currentColor" strokeWidth="2.5" strokeDasharray="4 3" />
          <circle cx="16" cy="16" r="4" fill="currentColor" />
        </svg>
        <div>
          <div className="brand-name">Blur Studio</div>
          <div className="brand-sub">Face &amp; license-plate anonymiser</div>
        </div>
      </div>
      <div className="topbar-meta">
        {info?.test && (
          <span className="chip" title="Test-set mean average precision at IoU 0.5">
            test mAP@0.5 <strong>{pct(info.test.mAP50)}</strong>
          </span>
        )}
        {info && (
          <span className="chip mono" title={info.arch}>
            {info.device} · epoch {info.checkpoint_epoch ?? "?"}
          </span>
        )}
        <span className={`status ${online ? "ok" : "down"}`}>
          <span className="dot" />
          {online ? "API online" : "API offline"}
        </span>
      </div>
    </header>
  );
}
