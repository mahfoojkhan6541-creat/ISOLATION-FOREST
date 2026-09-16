CREATE TABLE `screeningDatasets` (
	`id` varchar(40) NOT NULL,
	`ownerId` int NOT NULL,
	`fileName` varchar(255) NOT NULL,
	`fileKey` varchar(512) NOT NULL,
	`contentHash` varchar(64) NOT NULL,
	`sourceKind` varchar(48) NOT NULL DEFAULT 'USER_UPLOAD',
	`validationSummary` json NOT NULL,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `screeningDatasets_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `screeningModelVersions` (
	`id` varchar(40) NOT NULL,
	`datasetId` varchar(40) NOT NULL,
	`ownerId` int NOT NULL,
	`modelKind` varchar(64) NOT NULL,
	`artifactKey` varchar(512) NOT NULL,
	`trainingDataHash` varchar(64) NOT NULL,
	`metrics` json NOT NULL,
	`policy` json NOT NULL,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `screeningModelVersions_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `screeningResults` (
	`id` varchar(40) NOT NULL,
	`runId` varchar(40) NOT NULL,
	`componentId` varchar(128) NOT NULL,
	`lotId` varchar(128) NOT NULL,
	`parameter` varchar(128) NOT NULL,
	`action` enum('PASS','REVIEW','REJECT') NOT NULL,
	`riskScore` double NOT NULL,
	`predicted168h` double NOT NULL,
	`interval90` double NOT NULL,
	`reasonCodes` json NOT NULL,
	`explanation` text NOT NULL,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `screeningResults_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `screeningRuns` (
	`id` varchar(40) NOT NULL,
	`datasetId` varchar(40) NOT NULL,
	`modelVersionId` varchar(40) NOT NULL,
	`ownerId` int NOT NULL,
	`dataOrigin` varchar(64) NOT NULL,
	`summary` json NOT NULL,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `screeningRuns_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
ALTER TABLE `screeningDatasets` ADD CONSTRAINT `screeningDatasets_ownerId_users_id_fk` FOREIGN KEY (`ownerId`) REFERENCES `users`(`id`) ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `screeningModelVersions` ADD CONSTRAINT `screeningModelVersions_datasetId_screeningDatasets_id_fk` FOREIGN KEY (`datasetId`) REFERENCES `screeningDatasets`(`id`) ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `screeningModelVersions` ADD CONSTRAINT `screeningModelVersions_ownerId_users_id_fk` FOREIGN KEY (`ownerId`) REFERENCES `users`(`id`) ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `screeningResults` ADD CONSTRAINT `screeningResults_runId_screeningRuns_id_fk` FOREIGN KEY (`runId`) REFERENCES `screeningRuns`(`id`) ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `screeningRuns` ADD CONSTRAINT `screeningRuns_datasetId_screeningDatasets_id_fk` FOREIGN KEY (`datasetId`) REFERENCES `screeningDatasets`(`id`) ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `screeningRuns` ADD CONSTRAINT `screeningRuns_modelVersionId_screeningModelVersions_id_fk` FOREIGN KEY (`modelVersionId`) REFERENCES `screeningModelVersions`(`id`) ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `screeningRuns` ADD CONSTRAINT `screeningRuns_ownerId_users_id_fk` FOREIGN KEY (`ownerId`) REFERENCES `users`(`id`) ON DELETE no action ON UPDATE no action;