import { desc, eq } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { InsertUser, screeningDatasets, screeningModelVersions, screeningResults, screeningRuns, users } from "../drizzle/schema";
import { ENV } from './_core/env';
import { nanoid } from "nanoid";

let _db: ReturnType<typeof drizzle> | null = null;

// Lazily create the drizzle instance so local tooling can run without a DB.
export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot upsert user: database not available");
    return;
  }

  try {
    const values: InsertUser = {
      openId: user.openId,
    };
    const updateSet: Record<string, unknown> = {};

    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];

    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };

    textFields.forEach(assignNullable);

    if (user.lastSignedIn !== undefined) {
      values.lastSignedIn = user.lastSignedIn;
      updateSet.lastSignedIn = user.lastSignedIn;
    }
    if (user.role !== undefined) {
      values.role = user.role;
      updateSet.role = user.role;
    } else if (user.openId === ENV.ownerOpenId) {
      values.role = 'admin';
      updateSet.role = 'admin';
    }

    if (!values.lastSignedIn) {
      values.lastSignedIn = new Date();
    }

    if (Object.keys(updateSet).length === 0) {
      updateSet.lastSignedIn = new Date();
    }

    await db.insert(users).values(values).onDuplicateKeyUpdate({
      set: updateSet,
    });
  } catch (error) {
    console.error("[Database] Failed to upsert user:", error);
    throw error;
  }
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get user: database not available");
    return undefined;
  }

  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);

  return result.length > 0 ? result[0] : undefined;
}

export async function createScreeningDataset(input: {
  ownerId: number;
  fileName: string;
  fileKey: string;
  contentHash: string;
  sourceKind: string;
  validationSummary: Record<string, unknown>;
}) {
  const db = await getDb();
  if (!db) return null;
  const id = nanoid();
  await db.insert(screeningDatasets).values({ ...input, id });
  return id;
}

export async function createModelVersion(input: {
  datasetId: string;
  ownerId: number;
  modelKind: string;
  artifactKey: string;
  trainingDataHash: string;
  metrics: Record<string, unknown>;
  policy: Record<string, unknown>;
}) {
  const db = await getDb();
  if (!db) return null;
  const id = nanoid();
  await db.insert(screeningModelVersions).values({ ...input, id });
  return id;
}

export async function createScreeningRun(input: {
  datasetId: string;
  modelVersionId: string;
  ownerId: number;
  dataOrigin: string;
  summary: Record<string, unknown>;
}) {
  const db = await getDb();
  if (!db) return null;
  const id = nanoid();
  await db.insert(screeningRuns).values({ ...input, id });
  return id;
}

export async function createScreeningResults(runId: string, records: Array<{
  componentId: string;
  lotId: string;
  parameter: string;
  action: "PASS" | "REVIEW" | "REJECT";
  riskScore: number;
  predicted168h: number;
  interval90: number;
  reasonCodes: unknown;
  explanation: string;
}>) {
  const db = await getDb();
  if (!db || records.length === 0) return;
  await db.insert(screeningResults).values(records.map(record => ({ ...record, id: nanoid(), runId })));
}

export async function listScreeningDatasets(ownerId: number) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(screeningDatasets).where(eq(screeningDatasets.ownerId, ownerId)).orderBy(desc(screeningDatasets.createdAt)).limit(20);
}
