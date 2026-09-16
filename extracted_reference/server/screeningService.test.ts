import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";
import { getDemoWorkspace, trainUploadedFile, validateUploadedFile } from "./screeningService";

describe("screeningService", () => {
  it("loads a deterministic synthetic demo workspace with auditable decisions", async () => {
    const workspace = await getDemoWorkspace();
    expect(workspace.dataOrigin).toBe("SYNTHETIC_DEMONSTRATION_ONLY");
    expect(workspace.scores.length).toBeGreaterThan(20);
    expect(workspace.scores[0]).toHaveProperty("action");
    expect(workspace.artifact).toHaveProperty("policy");
    expect(workspace.metrics).toHaveProperty("early_decision_contract");
  });

  it("trains and scores a validated upload through the same Python service used by the UI", async () => {
    const source = await readFile(path.join(process.cwd(), "artifacts", "synthetic_burnin_demo.csv"));
    const result = await trainUploadedFile("synthetic_burnin_demo.csv", source.toString("base64"));
    expect(result.summary.accepted_rows).toBeGreaterThan(100);
    expect(result.artifact).toHaveProperty("policy");
    expect(result.metrics).toHaveProperty("early_decision_contract");
    expect(result.scores.some(score => score.action === "REVIEW")).toBe(true);
    const demo = await getDemoWorkspace();
    const directScores = new Map(result.scores.map(score => [`${score.component_id}:${score.parameter}`, score]));
    expect(demo.scores).toHaveLength(result.scores.length);
    expect(demo.scores.every(score => {
      const direct = directScores.get(`${score.component_id}:${score.parameter}`);
      return direct?.action === score.action
        && direct?.risk_score === score.risk_score
        && direct?.predicted_168h === score.predicted_168h
        && direct?.prediction_interval_90 === score.prediction_interval_90
        && direct?.reason_codes === score.reason_codes
        && direct?.explanation === score.explanation;
    })).toBe(true);
  }, 120_000);

  it("rejects malformed upload paths before any model execution", async () => {
    await expect(validateUploadedFile("burnin.exe", Buffer.from("not data").toString("base64"))).rejects.toThrow("CSV, XLSX, XLS, or JSON");
    await expect(validateUploadedFile("empty.csv", "")).rejects.toThrow("between 1 byte and 3 MB");
    await expect(validateUploadedFile("invalid.csv", Buffer.from("not,a,valid,measurement\n1,2,3,4").toString("base64"))).rejects.toThrow("screening pipeline could not complete");
  });

  it("serves the demo workspace through the authenticated screening API", async () => {
    const ctx = {
      user: { id: 1, openId: "test-owner", name: "Test Owner", email: "owner@example.test", loginMethod: "test", role: "admin", createdAt: new Date(), updatedAt: new Date(), lastSignedIn: new Date() },
      req: { protocol: "https", headers: {} },
      res: { clearCookie: () => undefined },
    } as unknown as TrpcContext;
    const result = await appRouter.createCaller(ctx).screening.demoWorkspace();
    expect(result.dataOrigin).toBe("SYNTHETIC_DEMONSTRATION_ONLY");
    expect(result.scores.length).toBeGreaterThan(20);
  });
});
