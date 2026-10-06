import { Dropzone } from "./Dropzone";

export function EmptyState({ onFiles, online }: { onFiles: (files: File[]) => void; online: boolean }) {
  return (
    <div className="empty">
      <div className="empty-art" aria-hidden>
        <div className="blob face" />
        <div className="blob plate" />
      </div>
      <h1>Anonymise street photos</h1>
      <p className="muted">
        A two-stage detector finds every face and license plate, then blurs them before the image is shared. Everything
        runs on your machine.
      </p>
      <Dropzone onFiles={onFiles} />
      {!online && (
        <p className="warn small">
          The API isn’t reachable. Start it with <code>uv run streetshield serve</code>.
        </p>
      )}
    </div>
  );
}
