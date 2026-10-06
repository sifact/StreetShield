// Typed client for the FastAPI backend (src/streetshield/serving). Shapes mirror serving/schemas.py.

export type BlurMethod = "gaussian" | "pixelate";

export interface BlurParams {
  score_thresh: number;
  nms_iou: number;
  box_padding: number;
  method: BlurMethod;
}

export interface Detection {
  box: [number, number, number, number]; // x, y, w, h in result-image pixels
  label: string;
  score: number;
}

export interface BlurResult {
  width: number;
  height: number;
  original_width: number;
  original_height: number;
  detections: Detection[];
  counts: Record<string, number>;
  image: string; // data URL
  elapsed_ms: number;
  params: BlurParams;
}

export interface ClassMetrics {
  AP50: number;
  AP75: number;
  AP: number;
  "precision@0.5": number;
  "recall@0.5": number;
  num_gt: number;
}

export interface ModelInfo {
  status: string;
  device: string;
  arch: string;
  classes: string[];
  checkpoint_epoch: number | null;
  val: Record<string, number> | null;
  test: { mAP50: number; mAP75: number; mAP: number; num_images: number; per_class: Record<string, ClassMetrics> } | null;
  history: { epoch: number; val: { mAP50: number; mAP: number } }[];
  defaults: BlurParams;
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    /* fall through */
  }
  return `${res.status} ${res.statusText}`;
}

export async function fetchModelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  const res = await fetch("/api/model", { signal });
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

export async function blurImage(file: File, params: BlurParams, signal?: AbortSignal): Promise<BlurResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("score_thresh", String(params.score_thresh));
  form.append("nms_iou", String(params.nms_iou));
  form.append("box_padding", String(params.box_padding));
  form.append("method", params.method);
  const res = await fetch("/api/blur", { method: "POST", body: form, signal });
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

export const sameParams = (a: BlurParams, b: BlurParams) =>
  a.score_thresh === b.score_thresh && a.nms_iou === b.nms_iou && a.box_padding === b.box_padding && a.method === b.method;
