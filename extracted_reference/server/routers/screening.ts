import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { createModelVersion, createScreeningDataset, createScreeningResults, createScreeningRun, listScreeningDatasets } from "../db";
import { getDemoWorkspace, trainUploadedFile, validateUploadedFile } from "../screeningService";
import { protectedProcedure, router } from "../_core/trpc";
import { storagePut } from "../storage";

const uploadInput = z.object({ fileName: z.string().min(1).max(160), contentBase64: z.string().min(4).max(4_200_000) });

function contentType(extension: string) {
  return extension === ".csv" ? "text/csv" : extension === ".json" ? "application/json" : "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
}

export const screeningRouter = router({
  demoWorkspace: protectedProcedure.query(async () => {
    try {
      return await getDemoWorkspace();
    } catch (error) {
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: error instanceof Error ? error.message : "Unable to load the demonstration workspace.",
      });
    }
  }),
  validateUpload: protectedProcedure
    .input(uploadInput)
    .mutation(async ({ input }) => {
      try {
        return await validateUploadedFile(input.fileName, input.contentBase64);
      } catch (error) {
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: error instanceof Error ? error.message : "The measurement file could not be validated.",
        });
      }
    }),
  trainUpload: protectedProcedure.input(uploadInput).mutation(async ({ ctx, input }) => {
    try {
      const trained = await trainUploadedFile(input.fileName, input.contentBase64);
      const sourceObject = await storagePut(`burnin/${ctx.user.id}/datasets/${input.fileName}`, trained.fileBytes, contentType(trained.inputExtension));
      const datasetId = await createScreeningDataset({
        ownerId: ctx.user.id,
        fileName: input.fileName,
        fileKey: sourceObject.key,
        contentHash: trained.contentHash,
        sourceKind: "USER_UPLOAD",
        validationSummary: trained.summary,
      });
      let modelVersionId: string | null = null;
      let runId: string | null = null;
      if (datasetId) {
        const artifactObject = await storagePut(`burnin/${ctx.user.id}/models/${datasetId}.json`, JSON.stringify(trained.artifact, null, 2), "application/json");
        modelVersionId = await createModelVersion({
          datasetId,
          ownerId: ctx.user.id,
          modelKind: String(trained.artifact.model?.kind ?? "unknown"),
          artifactKey: artifactObject.key,
          trainingDataHash: String(trained.artifact.training_data_hash ?? trained.contentHash),
          metrics: trained.metrics,
          policy: trained.artifact.policy ?? {},
        });
        if (modelVersionId) {
          runId = await createScreeningRun({
            datasetId,
            modelVersionId,
            ownerId: ctx.user.id,
            dataOrigin: "USER_UPLOAD",
            summary: { actionCounts: trained.metrics.action_counts ?? {}, screenedSeries: trained.metrics.screened_series ?? trained.scores.length },
          });
          if (runId) {
            await createScreeningResults(runId, trained.scores.map((score: Record<string, string>) => ({
              componentId: score.component_id,
              lotId: score.lot_id,
              parameter: score.parameter,
              action: score.action as "PASS" | "REVIEW" | "REJECT",
              riskScore: Number(score.risk_score),
              predicted168h: Number(score.predicted_168h),
              interval90: Number(score.prediction_interval_90),
              reasonCodes: (() => { try { return JSON.parse(score.reason_codes); } catch { return [score.reason_codes]; } })(),
              explanation: score.explanation,
            })));
          }
        }
      }
      return { ...trained, fileBytes: undefined, inputExtension: undefined, datasetId, modelVersionId, runId };
    } catch (error) {
      throw new TRPCError({ code: "BAD_REQUEST", message: error instanceof Error ? error.message : "Training could not complete." });
    }
  }),
  recentDatasets: protectedProcedure.query(async ({ ctx }) => listScreeningDatasets(ctx.user.id)),
});
