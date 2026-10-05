import type { BlurParams, BlurResult } from "./api";

export type ItemStatus = "queued" | "processing" | "done" | "error";

/** One uploaded image and the latest blur result for it. */
export interface Item {
  id: string;
  file: File;
  url: string; // object URL of the original
  status: ItemStatus;
  result?: BlurResult;
  error?: string;
  errorParams?: BlurParams; // params that produced the error, so we don't retry the same failing request
}
