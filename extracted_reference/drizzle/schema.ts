import { double, int, json, mysqlEnum, mysqlTable, text, timestamp, varchar } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 * Extend this file with additional tables as your product grows.
 * Columns use camelCase to match both database fields and generated types.
 */
export const users = mysqlTable("users", {
  /**
   * Surrogate primary key. Auto-incremented numeric value managed by the database.
   * Use this for relations between tables.
   */
  id: int("id").autoincrement().primaryKey(),
  /** Manus OAuth identifier (openId) returned from the OAuth callback. Unique per user. */
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/** Uploaded source data is held in managed object storage; this table keeps traceable metadata only. */
export const screeningDatasets = mysqlTable("screeningDatasets", {
  id: varchar("id", { length: 40 }).primaryKey(),
  ownerId: int("ownerId").notNull().references(() => users.id),
  fileName: varchar("fileName", { length: 255 }).notNull(),
  fileKey: varchar("fileKey", { length: 512 }).notNull(),
  contentHash: varchar("contentHash", { length: 64 }).notNull(),
  sourceKind: varchar("sourceKind", { length: 48 }).notNull().default("USER_UPLOAD"),
  validationSummary: json("validationSummary").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

/** Versioned model evidence prevents unexplained policy or coefficient changes. */
export const screeningModelVersions = mysqlTable("screeningModelVersions", {
  id: varchar("id", { length: 40 }).primaryKey(),
  datasetId: varchar("datasetId", { length: 40 }).notNull().references(() => screeningDatasets.id),
  ownerId: int("ownerId").notNull().references(() => users.id),
  modelKind: varchar("modelKind", { length: 64 }).notNull(),
  artifactKey: varchar("artifactKey", { length: 512 }).notNull(),
  trainingDataHash: varchar("trainingDataHash", { length: 64 }).notNull(),
  metrics: json("metrics").notNull(),
  policy: json("policy").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

/** A batch screening run retains the model version and source dataset that created it. */
export const screeningRuns = mysqlTable("screeningRuns", {
  id: varchar("id", { length: 40 }).primaryKey(),
  datasetId: varchar("datasetId", { length: 40 }).notNull().references(() => screeningDatasets.id),
  modelVersionId: varchar("modelVersionId", { length: 40 }).notNull().references(() => screeningModelVersions.id),
  ownerId: int("ownerId").notNull().references(() => users.id),
  dataOrigin: varchar("dataOrigin", { length: 64 }).notNull(),
  summary: json("summary").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

/** Structured, auditable component decisions; measurement files themselves remain in storage. */
export const screeningResults = mysqlTable("screeningResults", {
  id: varchar("id", { length: 40 }).primaryKey(),
  runId: varchar("runId", { length: 40 }).notNull().references(() => screeningRuns.id),
  componentId: varchar("componentId", { length: 128 }).notNull(),
  lotId: varchar("lotId", { length: 128 }).notNull(),
  parameter: varchar("parameter", { length: 128 }).notNull(),
  action: mysqlEnum("action", ["PASS", "REVIEW", "REJECT"]).notNull(),
  riskScore: double("riskScore").notNull(),
  predicted168h: double("predicted168h").notNull(),
  interval90: double("interval90").notNull(),
  reasonCodes: json("reasonCodes").notNull(),
  explanation: text("explanation").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});
