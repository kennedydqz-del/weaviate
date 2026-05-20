//                           _       _
// __      _____  __ ___   ___  __ _| |_ ___
// \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
//  \ V  V /  __/ (_| |\ V /| | (_| | ||  __/
//   \_/\_/ \___|\__,_| \_/ |_|\__,_|\__\___|
//
//  Copyright © 2016 - 2026 Weaviate B.V. All rights reserved.
//
//  CONTACT: hello@weaviate.io
//

package db

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// KnownReindexTaskLookup reports whether the given (taskID, taskVersion)
// tuple is currently known to the DTM scheduler. The implementation is
// supplied by the caller — typically a closure that queries the DTM
// manager's `ListDistributedTasks` snapshot — so this package stays
// free of a direct cluster/distributedtask dependency. Consulted once
// per tracker dir; the audit is not a query-path, so the per-call cost
// is acceptable.
type KnownReindexTaskLookup func(taskID string, taskVersion uint64) bool

// SetReindexAuditDeps installs the dependencies the
// [DB.AuditOrphanReindexTrackersIfReady] no-arg helper needs to run an
// audit on-demand. The setter is intended to be called once, from
// the goroutine that owns Scheduler bootstrap, the moment the lookup
// closure becomes meaningful.
//
// Threading: a single mutex guards the (lookup, logger) tuple.
// Concurrent calls from later code paths (e.g. a hypothetical
// post-restore second wiring) overwrite atomically; readers see one
// or the other tuple, never a half-set one.
func (db *DB) SetReindexAuditDeps(lookup KnownReindexTaskLookup, logger logrus.FieldLogger) {
	db.reindexAuditMu.Lock()
	defer db.reindexAuditMu.Unlock()
	db.reindexAuditLookup = lookup
	db.reindexAuditLogger = logger
}

// AuditOrphanReindexTrackersIfReady is the no-arg wrapper around
// [DB.AuditOrphanReindexTrackers] used from places that need to
// trigger an audit but were not present at startup to capture the
// lookup closure (the canonical example: the schema executor's
// post-restore RestoreClassDir hook, which fires from a RAFT FSM
// apply on a goroutine that does not have direct DTM access).
//
// Returns nil — never an error — when the audit dependencies have
// not been set yet. This is the correct behavior: the startup-time
// audit goroutine has not finished wiring deps, and the orphans (if
// any) will be picked up by the startup audit once deps land.
func (db *DB) AuditOrphanReindexTrackersIfReady(ctx context.Context) error {
	db.reindexAuditMu.RLock()
	lookup := db.reindexAuditLookup
	logger := db.reindexAuditLogger
	db.reindexAuditMu.RUnlock()
	if lookup == nil {
		return nil
	}
	return db.AuditOrphanReindexTrackers(ctx, lookup, logger)
}

// orphanReindexTracker describes one tracker directory the audit has
// classified as an orphan (DTM does not know about the owning task).
// The struct exists only to give the cleanup loop a single
// place to thread the contextual fields a WARN log needs.
type orphanReindexTracker struct {
	collection  string
	shardName   string
	dirName     string
	prefix      string
	generation  int
	taskID      string
	taskVersion uint64
	unitID      string
	properties  []string
	indexTypes  []string
}

// String builds the one-line WARN payload the audit uses for each
// orphan it cleans up. The format is keyed so log queries can grep on
// any field (taskID, dir, collection) without parsing.
func (o *orphanReindexTracker) String() string {
	return fmt.Sprintf(
		"collection=%q shard=%q tracker=%q gen=%d taskID=%q taskVersion=%d unitID=%q properties=%v indexTypes=%v",
		o.collection, o.shardName, o.dirName, o.generation,
		o.taskID, o.taskVersion, o.unitID, o.properties, o.indexTypes)
}

// AuditOrphanReindexTrackers walks every loaded shard's
// `<lsm>/.migrations/` directory and quarantines tracker directories
// whose owning DTM task is unknown to the scheduler. An orphan
// tracker is one whose `payload.mig` references a (TaskID, TaskVersion)
// that DTM does not know about — the canonical case is a restored
// cluster whose backup payload captured the tracker but not the DTM
// unit driving it (see 0-weaviate-issues#215 B3).
//
// For each orphan tracker the audit:
//
//  1. Logs a structured WARN naming the tracker, owning task, and
//     affected properties / indexes.
//  2. For each (property, indexType) the orphan touches, calls
//     [Shard.CleanStalePartialReindexState], which:
//     - shuts down any orphan sidecar bucket loaded under the
//     strategy's prefix (`property_<p>_<index>__…_<gen>`);
//     - removes the sidecar dir + the GlobalBucketRegistry entry;
//     - removes the `.migrations/<tracker>/` directory itself.
//
// The canonical main bucket is never touched: it serves pre-migration
// data on a restored cluster because the cluster-wide schema flip
// never committed (the DTM task was never finalized post-restore).
// After the audit the restored cluster reaches a self-consistent
// state: schema reflects pre-migration tokenization / index state,
// queries hit the canonical bucket, and disk space leaked by the
// orphan sidecar is reclaimed.
//
// Pre-conditions:
//
//   - DTM scheduler bootstrap has run on this node, i.e. the closure
//     returns a stable answer for every (taskID, version) it sees.
//   - Shards have completed initial load. Lazy-loaded MT shards that
//     are still cold are skipped — they'll be re-evaluated when next
//     activated (see the existing pre-load CleanStalePartialReindexState
//     hooks in the cancel / submit flow).
//
// Returns nil even when individual shards / trackers fail; per-orphan
// errors are logged. The audit is best-effort and a single bad
// tracker must not abort cleanup of the others. The return value
// exists for future composition; today, callers ignore it after the
// logger has captured everything actionable.
func (db *DB) AuditOrphanReindexTrackers(ctx context.Context, knownTask KnownReindexTaskLookup, logger logrus.FieldLogger) error {
	if logger == nil {
		logger = logrus.New()
	}
	if knownTask == nil {
		// Defensive: a nil lookup means "every task is unknown", which
		// would auto-quarantine in-flight migrations during a normal
		// restart. Refuse loudly rather than wreck production.
		logger.Error("reindex orphan audit: KnownReindexTaskLookup is nil; skipping audit (every legitimate in-flight reindex would be misclassified as an orphan)")
		return fmt.Errorf("reindex orphan audit: KnownReindexTaskLookup is nil")
	}

	auditLogger := logger.WithField("action", "reindex_orphan_audit")

	rootPath := db.config.RootPath
	if rootPath == "" {
		return nil
	}

	indexEntries, err := os.ReadDir(rootPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		auditLogger.WithField("path", rootPath).
			Warnf("reindex orphan audit: cannot read root path; skipping audit: %v", err)
		return nil
	}

	// Build a lookup from on-disk indexID directory name back to the
	// loaded Index (when present) so we can route per-shard cleanup
	// through the in-memory state where it exists.
	db.indexLock.RLock()
	loadedByID := make(map[string]*Index, len(db.indices))
	for id, idx := range db.indices {
		loadedByID[id] = idx
	}
	db.indexLock.RUnlock()

	var orphanCount int
	for _, indexEntry := range indexEntries {
		if !indexEntry.IsDir() {
			continue
		}
		indexDir := indexEntry.Name()
		indexPath := filepath.Join(rootPath, indexDir)
		shardEntries, shardErr := os.ReadDir(indexPath)
		if shardErr != nil {
			continue
		}
		idx := loadedByID[indexDir]
		// Best-effort class name for logging — the on-disk directory
		// name and the in-memory class name can diverge only via the
		// indexID transformation in [indexID], which is irreversible
		// without consulting the schema. The loaded-index path uses
		// the real class name; the unloaded path uses the on-disk
		// directory name as a best-effort label.
		collection := indexDir
		if idx != nil {
			collection = idx.Config.ClassName.String()
		}
		for _, shardEntry := range shardEntries {
			if !shardEntry.IsDir() {
				continue
			}
			shardName := shardEntry.Name()
			lsmPath := filepath.Join(indexPath, shardName, "lsm")
			orphans := collectOrphanTrackers(lsmPath, collection, shardName, knownTask, auditLogger)
			if len(orphans) == 0 {
				continue
			}
			// Route to the loaded-shard path when possible — it handles
			// in-memory bucket pointers and the GlobalBucketRegistry
			// entries. Fall through to disk-only cleanup when the shard
			// is not loaded (post-restore-before-class-load OR cold MT
			// tenant); the disk dirs are removed directly and the
			// loaded-shard hooks (CleanStalePartialReindexState's
			// submit/cancel callers) handle any stray state if/when the
			// shard later activates.
			var shard *Shard
			if idx != nil {
				if sl := idx.shards.Load(shardName); sl != nil {
					if s, ok := sl.(*Shard); ok {
						shard = s
					}
				}
			}
			if shard != nil {
				orphanCount += db.cleanLoadedShardOrphans(ctx, shard, orphans, auditLogger)
			} else {
				orphanCount += cleanUnloadedShardOrphans(lsmPath, orphans, auditLogger)
			}
		}
	}

	if orphanCount > 0 {
		auditLogger.WithField("orphan_count", orphanCount).
			Warn("reindex orphan audit: cleanup complete; restored cluster reached self-consistent state (canonical buckets retained, orphan sidecars and tracker dirs removed)")
	} else {
		auditLogger.Debug("reindex orphan audit: no orphan trackers found")
	}
	return nil
}

// collectOrphanTrackers walks `<lsmPath>/.migrations/` and returns
// every tracker dir classified as an orphan (started.mig present,
// tidied.mig/merged.mig absent, payload.mig parseable, and the
// referenced (taskID, version) NOT known to DTM). The function does
// NOT modify any state — that's the caller's job, since the cleanup
// path depends on whether the shard is loaded.
func collectOrphanTrackers(lsmPath, collection, shardName string, knownTask KnownReindexTaskLookup, logger logrus.FieldLogger) []orphanReindexTracker {
	migsDir := filepath.Join(lsmPath, ".migrations")
	entries, err := os.ReadDir(migsDir)
	if err != nil {
		return nil
	}
	var orphans []orphanReindexTracker
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dirName := entry.Name()
		prefix, generation, ok := parseMigrationDirName(dirName)
		if !ok {
			continue
		}
		trackerPath := filepath.Join(migsDir, dirName)
		if fileExistsInDir(trackerPath, "tidied.mig") || fileExistsInDir(trackerPath, "merged.mig") {
			continue
		}
		if !fileExistsInDir(trackerPath, "started.mig") {
			continue
		}
		rec, recOK := loadAuditRecord(trackerPath)
		if !recOK {
			logger.WithField("collection", collection).WithField("shard", shardName).
				WithField("tracker", dirName).
				Warn("reindex orphan audit: tracker missing payload.mig; manual cleanup may be needed")
			continue
		}
		if knownTask(rec.TaskID, rec.TaskVersion) {
			continue
		}
		orphans = append(orphans, orphanReindexTracker{
			collection:  collection,
			shardName:   shardName,
			dirName:     dirName,
			prefix:      prefix,
			generation:  generation,
			taskID:      rec.TaskID,
			taskVersion: rec.TaskVersion,
			unitID:      rec.UnitID,
			properties:  append([]string(nil), rec.Payload.Properties...),
			indexTypes:  semanticMigrationIndexTypesForAudit(rec.Payload.MigrationType),
		})
	}
	return orphans
}

// cleanLoadedShardOrphans cleans every orphan on the shard under a
// single PauseCompaction window. Per-orphan pause/resume previously
// raced: the defer re-enabled compaction between orphans, a fresh
// compaction started on the next orphan's sidecar bucket, and the
// next pause timed out trying to drain it.
//
// The audit's pause path does NOT coordinate with backup's
// [Index.HaltForTransfer] via haltForTransferMux. A concurrent backup
// is gated upstream by [Backupable] / [Index.refuseIfReindexInFlight]
// — it refuses on the not-yet-removed tracker dirs before reaching
// Deactivate.
func (db *DB) cleanLoadedShardOrphans(ctx context.Context, shard *Shard, orphans []orphanReindexTracker, logger logrus.FieldLogger) int {
	if len(orphans) == 0 {
		return 0
	}
	pauseCtx, cancelPause := context.WithTimeout(ctx, orphanCleanupPauseTimeout)
	defer cancelPause()
	if err := shard.store.PauseCompaction(pauseCtx); err != nil {
		logger.WithField("collection", orphans[0].collection).WithField("shard", orphans[0].shardName).
			Warnf("reindex orphan audit: failed to pause compaction on shard; skipping all orphan cleanups on this shard (next restart will retry): %v", err)
		return 0
	}
	// Resume must fire even if the audit ctx was cancelled.
	defer func() {
		if err := shard.store.ResumeCompaction(context.Background()); err != nil {
			logger.WithField("shard", orphans[0].shardName).
				Warnf("reindex orphan audit: failed to resume compaction after orphan cleanup; the next restart will resume it naturally: %v", err)
		}
	}()

	cleaned := 0
	for i := range orphans {
		o := &orphans[i]
		logger.WithField("orphan", o.String()).
			Warn("reindex orphan audit: found tracker for unknown task (typically backup-restore of a pre-#215-fix payload); quarantining sidecar bucket + tracker dir")
		if err := db.cleanupOrphanTrackerCompactionPaused(ctx, shard, o, logger); err != nil {
			logger.WithField("orphan", o.String()).
				Warnf("reindex orphan audit: cleanup failed for tracker; manual intervention may be required to reclaim the disk space: %v", err)
			continue
		}
		cleaned++
	}
	return cleaned
}

// cleanUnloadedShardOrphans removes orphan tracker dirs + their
// matching sidecar bucket dirs directly from disk. Used when the
// shard has not been loaded into the live DB yet — the typical
// post-restore-before-FSM-apply window. No in-memory bucket pointers
// or GlobalBucketRegistry entries exist for the orphan, so plain
// `os.RemoveAll` is sufficient.
//
// On the post-restore path the FSM has not yet applied the schema and
// the *Shard struct does not exist — so there is no live state to
// disturb. Direct disk removal is the correct cleanup primitive
// here; when the FSM later loads the class for the first time, the
// shard init walks a clean `.migrations/` and `lsm/` directory.
func cleanUnloadedShardOrphans(lsmPath string, orphans []orphanReindexTracker, logger logrus.FieldLogger) int {
	cleaned := 0
	for i := range orphans {
		o := &orphans[i]
		logger.WithField("orphan", o.String()).
			Warn("reindex orphan audit: found tracker for unknown task on unloaded shard (post-restore window); removing tracker dir + sidecar dirs from disk")
		trackerPath := filepath.Join(lsmPath, ".migrations", o.dirName)
		if err := os.RemoveAll(trackerPath); err != nil {
			logger.WithField("orphan", o.String()).
				Warnf("reindex orphan audit: failed to remove orphan tracker dir; manual intervention may be required: %v", err)
			continue
		}
		removeUnloadedSidecarsForOrphan(lsmPath, o, logger)
		cleaned++
	}
	return cleaned
}

// removeUnloadedSidecarsForOrphan removes per-property sidecar bucket
// directories that match the orphan's per-property prefix and
// generation. Called only from the unloaded-shard cleanup path; no
// in-memory state is involved.
//
// We can't reproduce the exact strategy-specific suffix names without
// the strategy instance, so we scan the lsm dir and match by:
//
//   - canonical main-bucket prefix (e.g. `property_body__`,
//     `property_body_searchable__`, `property_body_rangeable__`); and
//   - the gen-suffix tail `_<N>` that the orphan tracker carries.
//
// This matches every sidecar variant the runtime-reindex code path
// produces (`__retokenize_reindex_<N>`, `__filt_retokenize_ingest_<N>`,
// `__blockmax_<N>`, etc.) without hard-coding the strategy suffix
// vocabulary.
func removeUnloadedSidecarsForOrphan(lsmPath string, o *orphanReindexTracker, logger logrus.FieldLogger) {
	entries, err := os.ReadDir(lsmPath)
	if err != nil {
		return
	}
	genSuffixStr := genSuffix(o.generation)
	for _, propName := range o.properties {
		prefixes := []string{
			"property_" + propName + "__",
			"property_" + propName + "_searchable__",
			"property_" + propName + "_rangeable__",
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			name := entry.Name()
			matched := false
			for _, p := range prefixes {
				if strings.HasPrefix(name, p) {
					matched = true
					break
				}
			}
			if !matched {
				continue
			}
			if !strings.HasSuffix(name, genSuffixStr) {
				continue
			}
			path := filepath.Join(lsmPath, name)
			if err := os.RemoveAll(path); err != nil {
				logger.WithField("path", path).
					Warnf("reindex orphan audit: failed to remove orphan sidecar dir; manual intervention may be required: %v", err)
			}
		}
	}
}

// orphanCleanupPauseTimeout caps how long the audit waits for an
// in-flight compaction on the orphan sidecar bucket to drain before
// proceeding with cleanup. The default lsmkv compaction cycle target
// is in the low minutes for the segment sizes orphan sidecars usually
// carry; 5 minutes leaves headroom for a slow CI runner while still
// bounding worst case so a misbehaving compaction does not wedge the
// audit forever. On timeout the audit defers cleanup of that one
// tracker to the next process restart — that path picks up the same
// orphan because the tracker dir is still on disk, but with a
// different in-flight compaction state.
const orphanCleanupPauseTimeout = 5 * time.Minute

// cleanupOrphanTrackerCompactionPaused invokes
// CleanStalePartialReindexState for every (property, indexType) the
// orphan claims, which is the existing shutdown+remove+registry-clear
// path the cancel-handler uses.
//
// PRE-CONDITION: the caller (auditShardForOrphans) has already issued
// [Store.PauseCompaction] on this shard and holds the pause for the
// duration of every orphan cleanup on the shard. Pausing per-orphan
// would race the cycle manager: the resume in the defer between two
// orphans would allow a fresh compaction to start on a different
// orphan sidecar bucket, and the next pause would time out trying to
// drain it.
//
// Safe to call on a loaded shard concurrent with normal traffic: the
// inner function acquires the shard-local locks the rest of the
// runtime-reindex machinery already coordinates on, and the
// pause/resume primitives are the same ones [Shard.HaltForTransfer]
// uses on the backup path.
func (db *DB) cleanupOrphanTrackerCompactionPaused(ctx context.Context, shard *Shard, o *orphanReindexTracker, logger logrus.FieldLogger) error {
	if len(o.properties) == 0 || len(o.indexTypes) == 0 {
		// No (prop, indexType) pair to act on — likely a class-level
		// migration (Map→Blockmax) whose properties live inside the
		// strategy's per-property bookkeeping rather than the payload.
		// Fall back to a direct tracker-dir removal so disk usage is
		// reclaimed even when CleanStalePartialReindexState wouldn't
		// match anything.
		trackerPath := filepath.Join(shard.pathLSM(), ".migrations", o.dirName)
		if err := os.RemoveAll(trackerPath); err != nil {
			return fmt.Errorf("remove orphan tracker dir %q: %w", trackerPath, err)
		}
		logger.WithField("orphan", o.String()).
			Info("reindex orphan audit: removed class-level tracker dir (no property/indexType to clean via CleanStalePartialReindexState)")
		return nil
	}

	for _, propName := range o.properties {
		for _, indexType := range o.indexTypes {
			if err := shard.CleanStalePartialReindexState(ctx, propName, indexType); err != nil {
				return fmt.Errorf("clean stale partial reindex state for (prop=%q,indexType=%q): %w", propName, indexType, err)
			}
		}
	}
	return nil
}

// loadAuditRecord reads the on-disk recovery record for a tracker dir
// using the same payload.mig contract as
// [loadReindexRecoveryRecord], but with the looser sentinel-set gate
// the audit needs: payload.mig must be present and parseable; the
// started.mig / reindexed.mig / tidied.mig presence is the caller's
// responsibility (already filtered above).
func loadAuditRecord(trackerPath string) (reindexRecoveryRecord, bool) {
	var rec reindexRecoveryRecord
	data, err := os.ReadFile(filepath.Join(trackerPath, reindexRecoveryPayloadFile))
	if err != nil {
		return rec, false
	}
	if err := json.Unmarshal(data, &rec); err != nil {
		return rec, false
	}
	return rec, true
}

// semanticMigrationIndexTypesForAudit returns the (property, indexType)
// fan-out the audit's CleanStalePartialReindexState loop iterates over.
// It extends [semanticMigrationIndexTypes] with the format-only
// strategies so the audit handles ALL migration types — semantic and
// format-only alike. The classic helper restricts itself to semantic
// migrations because the LocalCallbacksDone recovery check it backs
// only fires for the swap-barrier family; the audit needs the full
// set because orphans can come from any migration shape.
//
// Class-level migrations (Map→Blockmax, RoaringSet refresh) return nil
// — they don't carry per-property index types in their payload, and
// the audit falls back to direct tracker-dir removal for them via
// [DB.cleanupOrphanTrackerCompactionPaused]'s nil-len branch.
func semanticMigrationIndexTypesForAudit(mt ReindexMigrationType) []string {
	switch mt {
	case ReindexTypeChangeTokenization:
		return []string{"searchable", "filterable"}
	case ReindexTypeChangeTokenizationFilterable:
		return []string{"filterable"}
	case ReindexTypeEnableSearchable:
		return []string{"searchable"}
	case ReindexTypeEnableFilterable:
		return []string{"filterable"}
	case ReindexTypeEnableRangeable, ReindexTypeRepairRangeable:
		return []string{"rangeable"}
	case ReindexTypeRepairSearchable:
		// Map→Blockmax is class-level — no per-property indexType
		// mapping. Return nil to trigger the direct-removal branch.
		return nil
	case ReindexTypeRepairFilterable:
		// RoaringSet refresh is class-level — same as above.
		return nil
	}
	// Defensive: a future MigrationType added without a mapping here
	// must still be handled (direct removal is safe — orphan sidecar
	// dirs the canonical bucket doesn't reference cannot leak query
	// data). Class-level cleanup via os.RemoveAll on the tracker dir
	// reclaims the disk space; the per-property sidecars remain only
	// when a known prefix/indexType is added.
	return nil
}
