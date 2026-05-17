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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ErrBackupBlockedByInFlightReindex is returned by [Shard.HaltForTransfer]
// (and the inactive-shard backup paths in [Index.backupShardWithHardlinks]
// / [Index.backupShardWithoutHardlinks]) when at least one runtime-reindex
// task is in flight on the shard. It is wrapped with a structured message
// that includes every active tracker on the shard so an operator can see
// which migrations are blocking the backup.
//
// Why we refuse rather than silently snapshot the in-flight state: the
// backup file walk + hardlink loop races against the reindex iteration in
// two distinct ways (see 0-weaviate-issues#215):
//
//  1. WAL close race — [Shard.HaltForTransfer] flushes memtables; the
//     reindex iteration writes concurrently to the source bucket via its
//     double-write callbacks, and the two paths race to close the same
//     commit log file. We have observed ~30% backup failures with
//     "file already closed" on the source bucket's WAL.
//
//  2. Hardlink ENOENT race — [Store.ListFiles] enumerates segments at one
//     instant; the reindex iteration creates / rotates / deletes segment
//     files in the next few hundred milliseconds. The hardlink loop then
//     attempts to link a file that no longer exists. We have observed
//     ~10% backup failures with "no such file or directory".
//
// Even when neither race fires and the backup completes, the captured
// state is not a self-consistent point in time: it mixes pre-swap bucket
// registrations with post-swap sentinel files, producing an on-disk
// state the restored cluster cannot reach via any sequence of FSM
// transitions. The orphan tracker + sidecar bucket persist forever on
// the restored cluster because the DTM unit was never part of the backup
// payload, so no scheduler ever drives the half-migration to completion.
//
// The proper architectural fix — pause the iteration during
// HaltForTransfer, or capture DTM state in the backup payload — is a
// larger change tracked separately. Until then, the safest behavior is
// to refuse the backup with a clear, actionable error and let the
// operator retry once the migration finishes (typically tens of seconds
// to a few minutes on a class-sized corpus).
var ErrBackupBlockedByInFlightReindex = errors.New("backup blocked: runtime-reindex in flight on this shard")

// InFlightReindexTracker describes one runtime-reindex tracker directory
// that is currently between [markStarted] and [markTidied]. The struct
// carries enough context for a structured error message that names the
// blocking migration without exposing internal sentinel mechanics.
//
// Returned by [inFlightReindexTrackers]. Pure value type — safe to copy
// and to include in error chains.
type InFlightReindexTracker struct {
	// DirName is the bare `.migrations/<...>` entry, e.g.
	// "searchable_retokenize_body_1".
	DirName string

	// Prefix is the strategy prefix without the `_<gen>` suffix, e.g.
	// "searchable_retokenize_body".
	Prefix string

	// Generation is the per-node `_<N>` suffix; >= 1 for any directory
	// produced by the runtime-reindex code path.
	Generation int

	// Started, Reindexed, Tidied are the sentinel-file presence flags
	// at the moment [inFlightReindexTrackers] scanned the directory.
	// Started is always true for entries returned (the caller filters
	// on it); Tidied is always false. Reindexed indicates whether the
	// per-shard reindex iteration has reached a terminal point.
	Started   bool
	Reindexed bool
	Tidied    bool
}

// String produces a one-line summary suitable for log fields / error
// messages: "<dir> [started=true reindexed=false tidied=false]".
func (t InFlightReindexTracker) String() string {
	return fmt.Sprintf("%s [started=%v reindexed=%v tidied=%v]",
		t.DirName, t.Started, t.Reindexed, t.Tidied)
}

// inFlightReindexTrackers scans `<lsmPath>/.migrations/` for tracker
// directories that have `started.mig` present and neither `tidied.mig`
// nor `merged.mig` — i.e. directories whose owning reindex iteration
// has not finished swapping the new data into the canonical bucket.
//
// `merged.mig` is treated as a completion signal alongside `tidied.mig`
// because [FinalizeCompletedMigrations]'s recovery path knows how to
// promote a merged-but-not-tidied directory on the next startup without
// any in-process state. A backup that captures a merged directory and
// gets restored will run the same finalize path on the restored cluster
// — that case is safe and does not need to be rejected.
//
// Directories with `started.mig` absent are pre-iteration scratch space
// that doesn't affect the on-disk state of any bucket. Including them
// in the in-flight set would surface every pre-create as a backup
// blocker, which is unnecessary and noisy.
//
// Returned entries are sorted by DirName for stable error messages and
// deterministic tests.
//
// Missing `.migrations/` directory returns (nil, nil) — the common case
// for a shard that has never had a runtime-reindex.
func inFlightReindexTrackers(lsmPath string) ([]InFlightReindexTracker, error) {
	if lsmPath == "" {
		return nil, nil
	}
	migsDir := filepath.Join(lsmPath, ".migrations")
	entries, err := os.ReadDir(migsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read migrations dir %q: %w", migsDir, err)
	}

	out := make([]InFlightReindexTracker, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		prefix, gen, ok := parseMigrationDirName(name)
		if !ok {
			// Pre-generation legacy state (shouldn't exist on this branch)
			// or a stray directory left by manual operator surgery. Either
			// way it does not match the in-flight contract; defensive skip.
			continue
		}
		dirPath := filepath.Join(migsDir, name)
		started := fileExistsInDir(dirPath, "started.mig")
		if !started {
			continue
		}
		tidied := fileExistsInDir(dirPath, "tidied.mig")
		if tidied {
			continue
		}
		// `merged.mig` is also a completion signal: the reindex iteration
		// finished and the ingest dir holds the target-tokenization data
		// in full. FinalizeCompletedMigrations on the restored cluster
		// would promote it cleanly on next startup; no race risk for the
		// backup module to capture.
		if fileExistsInDir(dirPath, "merged.mig") {
			continue
		}
		out = append(out, InFlightReindexTracker{
			DirName:    name,
			Prefix:     prefix,
			Generation: gen,
			Started:    true,
			Reindexed:  fileExistsInDir(dirPath, "reindexed.mig"),
			Tidied:     false,
		})
	}

	sort.Slice(out, func(i, j int) bool { return out[i].DirName < out[j].DirName })
	return out, nil
}

// refuseIfReindexInFlight is the inactive-shard counterpart to the
// in-flight check inside [Shard.HaltForTransfer]. It looks at the
// on-disk `<lsm>/.migrations/` directory of the named shard and
// returns the same structured [ErrBackupBlockedByInFlightReindex]
// error if any tracker dir is between started and tidied/merged.
//
// Used by [Index.backupInactiveShardWithHardlinks] and
// [Index.backupInactiveShardWithoutHardlinks] so a COLD/INACTIVE
// shard that was deactivated mid-migration cannot quietly slip into
// the backup payload — even though no iteration goroutine is running
// on it, the DTM unit driving the migration is not part of the
// backup, so a restored cluster would still inherit the orphan
// tracker + sidecar bucket.
func (i *Index) refuseIfReindexInFlight(shardName string) error {
	lsmPath := shardPathLSM(i.path(), shardName)
	trackers, err := inFlightReindexTrackers(lsmPath)
	if err != nil {
		return fmt.Errorf("check in-flight reindex state for shard %q: %w", shardName, err)
	}
	return reindexInFlightError(shardName, trackers)
}

// reindexInFlightError returns an error that wraps
// [ErrBackupBlockedByInFlightReindex] with a human-readable summary of
// the active trackers. The wrapping preserves `errors.Is` on the
// sentinel so REST handlers can map the failure to a structured HTTP
// status without string matching.
//
// Returns nil when `trackers` is empty so callers can use the helper as
// a one-shot "is there anything to refuse?" gate:
//
//	if err := reindexInFlightError(shardName, trackers); err != nil {
//	    return err
//	}
func reindexInFlightError(shardName string, trackers []InFlightReindexTracker) error {
	if len(trackers) == 0 {
		return nil
	}
	parts := make([]string, len(trackers))
	for i, t := range trackers {
		parts[i] = t.String()
	}
	return fmt.Errorf(
		"%w: shard %q has %d active tracker(s): %s; retry after the migration finishes (poll GET /v1/schema/<class> until indexes are 'ready') or cancel it via PUT /v1/schema/<class>/indexes/<prop> {\"<indexType>\":{\"cancel\":true}}",
		ErrBackupBlockedByInFlightReindex,
		shardName,
		len(trackers),
		strings.Join(parts, ", "),
	)
}
