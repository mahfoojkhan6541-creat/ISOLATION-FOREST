import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const PROJECT_ROOT = process.cwd();
const ARTIFACTS = path.join(PROJECT_ROOT, "artifacts");
const PYTHON = process.env.PYTHON_BIN ?? "python3";

export type ValidationPreview = {
  summary: Record<string, unknown>;
  cleanPreview: Record<string, unknown>[];
  quarantinePreview: Record<string, unknown>[];
};

export type TrainedUpload = ValidationPreview & {
  artifact: Record<string, any>;
  metrics: Record<string, unknown>;
  scores: Array<Record<string, string>>;
  fileBytes: Buffer;
  contentHash: string;
  inputExtension: string;
};

function ensureSupportedName(fileName: string) {
  const extension = path.extname(fileName).toLowerCase();
  if (![".csv", ".xlsx", ".xls", ".json"].includes(extension)) {
    throw new Error("Upload a CSV, XLSX, XLS, or JSON measurement file.");
  }
  return extension;
}

function parseJsonOutput(stdout: string) {
  const start = stdout.indexOf("{");
  if (start === -1) throw new Error("The screening pipeline did not return a JSON summary.");
  return JSON.parse(stdout.slice(start));
}

async function runPython(args: string[], timeout = 110_000) {
  try {
    return await execFileAsync(PYTHON, ["ml/cli.py", ...args], {
      cwd: PROJECT_ROOT,
      timeout,
      maxBuffer: 8 * 1024 * 1024,
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Unknown pipeline failure";
    throw new Error(`The screening pipeline could not complete: ${detail}`);
  }
}

function parseCsvRow(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];
    if (char === '"' && quoted && next === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      cells.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  cells.push(current);
  return cells;
}

async function readCsvPreview(filePath: string, limit = 12) {
  const raw = await fs.readFile(filePath, "utf8");
  const [headerLine, ...rows] = raw.trim().split(/\r?\n/);
  if (!headerLine) return [];
  const headers = parseCsvRow(headerLine);
  return rows.slice(0, limit).map(row => {
    const cells = parseCsvRow(row);
    return Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""]));
  });
}

export async function validateUploadedFile(fileName: string, contentBase64: string): Promise<ValidationPreview> {
  const extension = ensureSupportedName(fileName);
  const bytes = Buffer.from(contentBase64, "base64");
  if (!bytes.length || bytes.length > 3 * 1024 * 1024) {
    throw new Error("The upload must be between 1 byte and 3 MB for interactive validation.");
  }
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), "sih26170-"));
  const input = path.join(temporary, `input${extension}`);
  const clean = path.join(temporary, "clean.csv");
  const quarantine = path.join(temporary, "quarantine.csv");
  try {
    await fs.writeFile(input, bytes);
    const { stdout } = await runPython(["validate-data", "--input", input, "--clean-output", clean, "--quarantine-output", quarantine]);
    return {
      summary: parseJsonOutput(stdout),
      cleanPreview: await readCsvPreview(clean),
      quarantinePreview: await readCsvPreview(quarantine),
    };
  } finally {
    await fs.rm(temporary, { recursive: true, force: true });
  }
}

export async function trainUploadedFile(fileName: string, contentBase64: string): Promise<TrainedUpload> {
  const inputExtension = ensureSupportedName(fileName);
  const fileBytes = Buffer.from(contentBase64, "base64");
  if (!fileBytes.length || fileBytes.length > 3 * 1024 * 1024) {
    throw new Error("The upload must be between 1 byte and 3 MB for interactive training.");
  }
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), "sih26170-train-"));
  const input = path.join(temporary, `input${inputExtension}`);
  const clean = path.join(temporary, "clean.csv");
  const quarantine = path.join(temporary, "quarantine.csv");
  const model = path.join(temporary, "model.json");
  const metrics = path.join(temporary, "metrics.json");
  const scores = path.join(temporary, "scores.csv");
  try {
    await fs.writeFile(input, fileBytes);
    const validationResult = await runPython(["validate-data", "--input", input, "--clean-output", clean, "--quarantine-output", quarantine]);
    const summary = parseJsonOutput(validationResult.stdout) as Record<string, unknown>;
    if (!summary.has_168h_target || Number(summary.component_parameter_series ?? 0) < 12) {
      throw new Error("Training needs at least 12 valid component/parameter series with 0h, 24h, and 168h measurements.");
    }
    await runPython(["train", "--input", input, "--model-output", model]);
    await runPython(["evaluate", "--input", input, "--model", model, "--output", metrics]);
    await runPython(["predict", "--input", input, "--model", model, "--output", scores]);
    const [artifactText, metricsText, scoreText] = await Promise.all([fs.readFile(model, "utf8"), fs.readFile(metrics, "utf8"), fs.readFile(scores, "utf8")]);
    const lines = scoreText.trim().split(/\r?\n/);
    const headers = parseCsvRow(lines.shift() ?? "");
    return {
      summary,
      cleanPreview: await readCsvPreview(clean),
      quarantinePreview: await readCsvPreview(quarantine),
      artifact: JSON.parse(artifactText),
      metrics: JSON.parse(metricsText),
      scores: lines.map(line => {
        const values = parseCsvRow(line);
        return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
      }),
      fileBytes,
      contentHash: createHash("sha256").update(fileBytes).digest("hex"),
      inputExtension,
    };
  } finally {
    await fs.rm(temporary, { recursive: true, force: true });
  }
}

export async function getDemoWorkspace() {
  const [artifactText, metricsText, sourceText] = await Promise.all([
    fs.readFile(path.join(ARTIFACTS, "demo_model.json"), "utf8"),
    fs.readFile(path.join(ARTIFACTS, "demo_metrics.json"), "utf8"),
    fs.readFile(path.join(ARTIFACTS, "demo_scores.csv"), "utf8"),
  ]);
  const lines = sourceText.trim().split(/\r?\n/);
  const headers = parseCsvRow(lines.shift() ?? "");
  const scores = lines.slice(0, 280).map(line => {
    const values = parseCsvRow(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
  return {
    artifact: JSON.parse(artifactText),
    metrics: JSON.parse(metricsText),
    scores,
    dataOrigin: "SYNTHETIC_DEMONSTRATION_ONLY",
  };
}
