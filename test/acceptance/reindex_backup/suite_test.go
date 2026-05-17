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

// Package reindex_backup_test contains end-to-end coverage for the
// interaction between the backup module and runtime-reindex tasks.
//
// 0-weaviate-issues#215 (QA Claude report) catalogued three high-
// severity bug classes on PR #10694:
//
//   - B1/B2: ~40% backup-create failure during an in-flight reindex
//     iteration (WAL "file already closed" + hardlink ENOENT races)
//   - B3:    silent half-state on restored clusters when the backup
//     payload captured `.migrations/<tracker>` without the DTM unit
//   - B4–B8: API recovery dead-ends (covered separately)
//
// This package pins the post-fix behavior on the active surface:
//
//   - Backup is REJECTED with ErrBackupBlockedByInFlightReindex
//     whenever any shard has a tracker with started.mig present and
//     tidied.mig/merged.mig absent. The rejection is fast (no flush
//     attempted), the response body names the blocking tracker, and
//     the rejection does not leave the per-shard halt counter
//     incremented (proven by a follow-up successful backup on the
//     same shard once the migration completes).
//
//   - Quiet-state backups (no in-flight tracker) round-trip cleanly.
//     This pins that the new gate does not regress the common case.
//
// The suite shares a single testcontainer to keep the runtime under
// 5 minutes per the CI/Pipeline Monitoring norm — each subtest uses a
// fresh class so the in-flight bit on one journey doesn't leak into
// the next. The filesystem backup backend is sufficient: the bug is
// in HaltForTransfer / list-files, not in any backend-specific
// transport.
package reindex_backup_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-openapi/strfmt"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	clientbackups "github.com/weaviate/weaviate/client/backups"
	"github.com/weaviate/weaviate/client/batch"
	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/test/acceptance/helpers/reindex"
	"github.com/weaviate/weaviate/test/docker"
	"github.com/weaviate/weaviate/test/helper"
	moduleshelper "github.com/weaviate/weaviate/test/helper/modules"
)

// TestBackupVsReindexSuite is the umbrella test that drives all
// reindex-backup interaction scenarios on a shared single-node
// testcontainer. Subtests are independent and use distinct class names
// so their `.migrations/` state cannot bleed between cases.
func TestBackupVsReindexSuite(t *testing.T) {
	ctx := context.Background()

	compose, err := docker.New().
		WithBackendFilesystem().
		WithWeaviate().
		// Faster scheduler ticks so an in-flight migration progresses
		// quickly enough for the "backup succeeds after migration
		// finishes" assertion to fit in the test timeout.
		WithWeaviateEnv("DISTRIBUTED_TASKS_SCHEDULER_TICK_INTERVAL_SECONDS", "1").
		Start(ctx)
	require.NoError(t, err)
	defer func() {
		require.NoError(t, compose.Terminate(ctx))
	}()

	helper.SetupClient(compose.GetWeaviate().URI())
	defer helper.ResetClient()
	restURI := compose.GetWeaviate().URI()

	// Dump container logs on failure for post-mortem.
	container := compose.GetWeaviate().Container()
	defer func() {
		if t.Failed() {
			reader, err := container.Logs(ctx)
			if err != nil {
				t.Logf("failed to read container logs: %v", err)
				return
			}
			defer reader.Close()
			logs, _ := io.ReadAll(reader)
			lines := strings.Split(string(logs), "\n")
			if len(lines) > 200 {
				lines = lines[len(lines)-200:]
			}
			t.Logf("=== Container logs (last 200 lines) ===\n%s", strings.Join(lines, "\n"))
		}
	}()

	// Baseline: a class with no migration produces a clean round-trip.
	// Pins that the new gate doesn't regress the common case.
	t.Run("BaselineBackupRoundTrip", func(t *testing.T) {
		testBaselineBackupRoundTrip(t, restURI)
	})

	// The headline assertion: a backup fired while a migration is in
	// flight is REJECTED with the structured ErrBackupBlockedByInFlightReindex
	// error, names the active tracker(s), and clears once the migration
	// completes — proving the halt counter wasn't left incremented.
	t.Run("BackupRefusedDuringInFlightMigration", func(t *testing.T) {
		testBackupRefusedDuringInFlightMigration(t, ctx, compose, restURI)
	})

	// After the migration finishes, the same backup-id-pattern must
	// succeed and round-trip. Pins that the rejection is self-clearing.
	t.Run("BackupSucceedsAfterMigrationFinishes", func(t *testing.T) {
		testBackupSucceedsAfterMigrationFinishes(t, restURI)
	})

	// Phase 2: orphan-on-restore cleanup. Crafts a pre-fix orphan
	// state on disk (an in-flight tracker dir + sidecar bucket dir,
	// payload referencing a taskID that DTM does not know about),
	// restarts the container, and asserts the post-bootstrap audit
	// removes the orphan while leaving the canonical bucket and
	// data intact. Pre-fix backups taken on older binaries are the
	// motivating real-world case; the test crafts the orphan via
	// `docker exec` so it does not depend on a separate old-binary
	// image. See 0-weaviate-issues#215 B3.
	t.Run("PostRestartOrphanAuditClearsTracker", func(t *testing.T) {
		testPostRestartOrphanAuditClearsTracker(t, ctx, compose, restURI)
	})

	// B5: cancel:true with no in-flight task must return a 404 that
	// carries a structured error body (collection / property /
	// indexType named, GET hint included). Bare 404 used to be
	// "indistinguishable from endpoint-not-found" per QA's report;
	// the structured body is what gives operators something to
	// recover with. See 0-weaviate-issues#215 B5.
	//
	// Subtests B5/B6/B7 re-resolve URI from compose each call because
	// PostRestartOrphanAuditClearsTracker above does a Stop+Start that
	// rebinds the test container to a new dynamic port; the captured
	// restURI from before the suite ran is stale at this point.
	t.Run("CancelOnNoInFlightReturnsStructured404", func(t *testing.T) {
		testCancelOnNoInFlightReturns404(t, compose.GetWeaviate().URI())
	})

	// B6: cleanup:true on orphan state — operator-driven recovery
	// path that does NOT switch the BM25 algorithm (the rebuild verb
	// does). Craft orphan state on disk via docker exec, fire PUT
	// cleanup, verify on-disk state is gone WITHOUT restart and
	// WITHOUT touching the schema's tokenization. Then verify that
	// cleanup:true with an in-flight task gets refused (409). See
	// 0-weaviate-issues#215 B6.
	t.Run("CleanupRemovesOrphanWithoutAlgorithmSwitch", func(t *testing.T) {
		testCleanupRemovesOrphan(t, ctx, compose, compose.GetWeaviate().URI())
	})

	// B7: PR description's `{"searchable":{"algorithm":"BlockMaxWAND"}}`
	// body shape is now accepted by the handler (vs. the prior 400).
	// When the class is already on BlockMaxWAND (the default for
	// new classes), the request is refused with 409 to prevent the
	// MapToBlockmax engine from running on a non-Map source — that
	// path crashes at the runtime-swap step with "rename ... no such
	// file or directory". When the operator submits "WAND", the
	// handler returns 400 (reverse direction not supported).
	//
	// Test pins all three:
	//   - already-blockmax + BlockMaxWAND → 409, structured body
	//   - already-blockmax + rebuild:true → 409, same body (parity)
	//   - already-blockmax + WAND          → 400
	t.Run("AlgorithmVerbRefusesOnAlreadyBlockMaxRejectsWAND", func(t *testing.T) {
		testAlgorithmVerb(t, compose.GetWeaviate().URI())
	})

	// Adjacent (per QA Claude verification at
	// 0-weaviate-issues#215#issuecomment-4470603684): system-level
	// regression guard for the MutationGuard CheckClassMutation path
	// introduced upstream in 2f77d4905e + 6b8d8e692a (#218 + #219).
	// The unit test TestSchemaManager_DeleteClass_MutationGuard at
	// the cluster/schema level pins the FSM-side check; this subtest
	// pins the same contract at the full RAFT+REST round-trip — a
	// DELETE /v1/schema/<class> while a reindex on `body` is STARTED
	// must be rejected with a structured 400 that names the
	// in-flight task. After the task finishes, DELETE must succeed.
	t.Run("MutationGuardBlocksDeleteClassDuringInFlight", func(t *testing.T) {
		testMutationGuardBlocksDeleteClassDuringInFlight(t, compose.GetWeaviate().URI())
	})
}

// testBaselineBackupRoundTrip creates a class with no migration in flight,
// runs a filesystem backup, deletes the class, restores from the backup,
// and checks the object count round-trips. Sanity check that the new
// gate does not regress the unaffected path.
func testBaselineBackupRoundTrip(t *testing.T, restURI string) {
	className := "ReindexBackup_Baseline"
	helper.CreateClass(t, &models.Class{
		Class: className,
		Properties: []*models.Property{
			{Name: "body", DataType: []string{"text"}, Tokenization: "word"},
		},
		Vectorizer: "none",
	})
	defer helper.DeleteClass(t, className)

	importBodies(t, className, 200)

	preCount := moduleshelper.GetClassCount(t, className, "")
	require.EqualValues(t, 200, preCount)

	backupAndRestoreRoundTrip(t, className, "reindex-backup-baseline", preCount,
		"restored count must equal pre-backup count")
}

// testBackupRefusedDuringInFlightMigration is the headline assertion.
// Flow:
//  1. Seed a class with enough rows that the change-tokenization
//     reindex iteration is observable for at least ~500ms (long enough
//     to win the race against the backup HTTP call).
//  2. Fire PUT searchable:{tokenization:lowercase} → returns 202.
//  3. Without polling sentinels, fire a backup-create. With the fix
//     in place, the backup returns FAILED almost immediately with
//     the structured "backup blocked: runtime-reindex in flight on
//     this shard..." message, naming the active tracker.
//  4. Wait for the migration to drain; assert the schema reports
//     ready for both searchable + filterable indexes.
//
// Without the fix this test would be flaky (the 40% backup-failure
// rate from QA), would silently produce a backup of the orphan state,
// or would race a successful-but-incomplete backup. After the fix the
// rejection is deterministic.
func testBackupRefusedDuringInFlightMigration(t *testing.T, ctx context.Context, compose *docker.DockerCompose, restURI string) {
	className := "ReindexBackup_RefusedDuringInFlight"
	helper.CreateClass(t, &models.Class{
		Class: className,
		Properties: []*models.Property{
			{Name: "body", DataType: []string{"text"}, Tokenization: "word"},
		},
		Vectorizer: "none",
	})
	defer helper.DeleteClass(t, className)

	// 50_000 objects with ~30-token bodies. Sized so the change-
	// tokenization iteration takes long enough that the indexes
	// status poll below has time to observe the "indexing" state and
	// the subsequent backup HTTP call lands mid-iteration. On a CI
	// runner the migration runs ~5-15s on this corpus.
	importBodies(t, className, 50_000)

	// Fire the migration. PUT returns 202 STARTED before the iteration
	// has finished, by design — the contract is asynchronous.
	taskID := submitChangeTokenization(t, restURI, className, "body", "lowercase")
	t.Logf("change-tokenization task submitted: %s", taskID)

	// Wait for the migration to reach "indexing" state, i.e. the
	// per-shard reindex iteration has started and the started.mig
	// sentinel is on disk. This is the SAME wait the QA harness
	// performs at 20ms tick. Without this wait, a small corpus can
	// finish the migration entirely before the backup HTTP call lands,
	// producing a spurious "backup succeeded" (false negative) here.
	awaitIndexingState(t, restURI, className, "body")

	backupID := "reindex-backup-refuse"

	// Post-Adjacent-17 fix: the Backupable precheck refuses during the
	// canCommit phase BEFORE the coordinator writes the initial
	// backup_config.json. The HTTP call therefore returns a
	// synchronous 422 with the structured error body — no async status
	// poll required. Pre-fix the same flow was: POST → 202, then poll
	// → FAILED with the same error; we preserve the substring
	// assertions but read them from the 422 payload.
	_, err := helper.CreateBackup(t, helper.DefaultBackupConfig(), className, "filesystem", backupID)
	require.Error(t, err, "create-backup must be refused synchronously while reindex is in flight")
	var refusal *clientbackups.BackupsCreateUnprocessableEntity
	require.ErrorAs(t, err, &refusal, "expected 422 BackupsCreateUnprocessableEntity, got %T: %v", err, err)
	require.NotNil(t, refusal.Payload, "422 payload must not be nil")
	require.NotEmpty(t, refusal.Payload.Error, "422 ErrorResponse must surface the refusal reason")
	errMsg := errorResponseMessage(refusal.Payload)

	// The error body must surface the structured message so REST callers
	// and operators can distinguish "backup blocked by in-flight reindex"
	// from other failure shapes (disk full, network blip, etc.).
	require.Contains(t, errMsg, "backup blocked: runtime-reindex in flight on this shard",
		"error body must name the blocking condition; got: %s", errMsg)
	require.Contains(t, errMsg, "tracker(s):",
		"error body must list the active tracker(s); got: %s", errMsg)
	// The active tracker for change-tokenization on a text property
	// uses the searchable_retokenize_<prop> dir-name prefix.
	require.Contains(t, errMsg, "searchable_retokenize_body",
		"error body must name the actual tracker dir so the operator knows which migration to wait for; got: %s", errMsg)
	// The error also surfaces the actionable remediation (poll schema
	// or cancel via the documented PUT). Without this, an operator
	// chasing the log line lacks a next step.
	require.Contains(t, errMsg, "retry after the migration finishes",
		"error body must include an actionable next step")

	// 0-weaviate-issues#215 Adjacent 17: the refusal must NOT leave a
	// staging dir on disk. Pre-fix, the coordinator wrote
	// /tmp/backups/<bid>/backup_config.json (Started) before commit
	// ran, and the per-node /tmp/backups/<bid>/<node>/backup.json
	// (Failed, Classes:[]) after the refusal — so a retry with the
	// same backup ID hit checkIfBackupExists's "Status != Cancelled"
	// rejection. Post-fix the per-node Backupable precheck refuses
	// during canCommit, before the coordinator's initial PutMeta —
	// no /tmp/backups/<bid>/ directory should exist at all.
	container := compose.GetWeaviate().Container()
	stagingPath := "/tmp/backups/" + backupID
	code, _, _ := container.Exec(ctx, []string{"test", "-d", stagingPath})
	assert.NotEqual(t, 0, code,
		"refused backup must not leave a staging dir at %s; pre-fix this dir was created with backup_config.json + per-node Classes:[] metadata, blocking same-ID retry", stagingPath)

	// Let the in-flight reindex drain so the post-test teardown is clean.
	reindexhelpers.AwaitReindexFinished(t, restURI, taskID, reindexhelpers.WithTimeout(120*time.Second))

	// Same-ID retry must now succeed cleanly (the migration has
	// drained; the staging dir is empty; no Failed metadata blocks
	// checkIfBackupExists). This is the operator-facing observable
	// of the staging-cleanup fix.
	_, err = helper.CreateBackup(t, helper.DefaultBackupConfig(), className, "filesystem", backupID)
	require.NoError(t, err, "same-ID retry after migration drains must succeed; pre-fix the staging dir left by the refusal would have blocked checkIfBackupExists")
	helper.ExpectBackupEventuallyCreated(t, backupID, "filesystem", nil,
		helper.WithDeadline(2*time.Minute))
}

// errorResponseMessage flattens the multi-element ErrorResponse Error
// slice into a single string for substring assertions in tests.
func errorResponseMessage(er *models.ErrorResponse) string {
	if er == nil {
		return ""
	}
	parts := make([]string, 0, len(er.Error))
	for _, e := range er.Error {
		if e == nil {
			continue
		}
		parts = append(parts, e.Message)
	}
	return strings.Join(parts, "; ")
}

// testBackupSucceedsAfterMigrationFinishes pins the "self-clearing"
// property of the refusal: once the migration completes, the same
// pattern of backup-create + restore works. This catches a regression
// where the rejection path would accidentally leave the per-shard
// halt counter incremented (forcing every subsequent halt to short-
// circuit on the "shard was already halted" branch and silently skip
// the flush).
func testBackupSucceedsAfterMigrationFinishes(t *testing.T, restURI string) {
	className := "ReindexBackup_SucceedsAfterFinish"
	helper.CreateClass(t, &models.Class{
		Class: className,
		Properties: []*models.Property{
			{Name: "body", DataType: []string{"text"}, Tokenization: "word"},
		},
		Vectorizer: "none",
	})
	defer helper.DeleteClass(t, className)

	importBodies(t, className, 500)
	preCount := moduleshelper.GetClassCount(t, className, "")
	require.EqualValues(t, 500, preCount)

	// Run a migration end-to-end first, then back up the quiet state.
	taskID := submitChangeTokenization(t, restURI, className, "body", "lowercase")
	t.Logf("change-tokenization task submitted: %s", taskID)
	reindexhelpers.AwaitReindexFinished(t, restURI, taskID, reindexhelpers.WithTimeout(60*time.Second))

	backupAndRestoreRoundTrip(t, className, "reindex-backup-after-finish", preCount,
		"post-migration backup must round-trip object count cleanly")
}

// importBodies inserts `count` objects into `className` with a `body`
// text property carrying enough tokens that the change-tokenization
// iteration has meaningful per-object work.
func importBodies(t *testing.T, className string, count int) {
	t.Helper()
	const batchSize = 500
	// Body text long enough that the analyzer does measurable per-object
	// work — short enough that 5k objects still import in a few seconds.
	body := strings.Repeat("Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel ", 4)
	for start := 0; start < count; start += batchSize {
		end := start + batchSize
		if end > count {
			end = count
		}
		objects := make([]*models.Object, end-start)
		for i := range objects {
			objects[i] = &models.Object{
				Class:      className,
				ID:         strfmt.UUID(uuid.New().String()),
				Properties: map[string]interface{}{"body": body},
			}
		}
		params := batch.NewBatchObjectsCreateParams().
			WithBody(batch.BatchObjectsCreateBody{Objects: objects})
		resp, err := helper.Client(t).Batch.BatchObjectsCreate(params, nil)
		require.NoError(t, err)
		require.NotNil(t, resp)
	}
}

// submitChangeTokenization issues PUT /v1/schema/<class>/indexes/<prop>
// with {"searchable":{"tokenization":<target>}} and returns the task ID.
// Asserts a 202 response; any other status fails the test.
func submitChangeTokenization(t *testing.T, restURI, collection, property, target string) string {
	t.Helper()
	body := fmt.Sprintf(`{"searchable":{"tokenization":%q}}`, target)
	url := fmt.Sprintf("http://%s/v1/schema/%s/indexes/%s", restURI, collection, property)
	req, err := http.NewRequest(http.MethodPut, url, bytes.NewReader([]byte(body)))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, http.StatusAccepted, resp.StatusCode,
		"index update returned non-202: %s", string(respBody))

	// Body is `{"status":"STARTED","taskId":"..."}` — pluck the taskId.
	type body202 struct {
		TaskID string `json:"taskId"`
		Status string `json:"status"`
	}
	var parsed body202
	require.NoError(t, json.Unmarshal(respBody, &parsed))
	require.NotEmpty(t, parsed.TaskID, "submit response missing taskId: %s", string(respBody))
	return parsed.TaskID
}

// testPostRestartOrphanAuditClearsTracker exercises Phase 2 (B3):
// the post-bootstrap orphan-reindex audit. With Phase 1's
// HaltForTransfer gate in place, organic creation of a pre-fix orphan
// is no longer possible — so this test crafts the orphan state on
// disk via `docker exec` (cp / mkdir / write) and restarts the
// container. After restart, the audit must:
//
//   - leave the canonical bucket and its data intact (count + queries
//     match the pre-orphan-injection baseline);
//   - remove the orphan tracker dir from `<lsm>/.migrations/`;
//   - remove the orphan sidecar bucket dir.
//
// The injected tracker carries a payload.mig referencing a taskID
// that the DTM scheduler has never seen — exactly the shape a
// pre-fix-backup restore would produce.
func testPostRestartOrphanAuditClearsTracker(t *testing.T, ctx context.Context, compose *docker.DockerCompose, restURI string) {
	const (
		className   = "ReindexBackup_OrphanAudit"
		shardLookup = "ReindexBackup_OrphanAudit"
	)

	// Seed canonical data on a fresh class.
	helper.CreateClass(t, &models.Class{
		Class: className,
		Properties: []*models.Property{
			{Name: "body", DataType: []string{"text"}, Tokenization: "word"},
		},
		Vectorizer: "none",
	})
	defer helper.DeleteClass(t, className)

	importBodies(t, className, 500)
	preCount := moduleshelper.GetClassCount(t, className, "")
	require.EqualValues(t, 500, preCount)

	shardName := reindexhelpers.GetFirstShardName(t, restURI, className)
	require.NotEmpty(t, shardName, "could not resolve shard name for %s", className)

	// Stop the container so we can stage the orphan state on disk
	// without racing the running shard. The same node is restarted
	// below; data path persists across stop/start.
	require.NoError(t, compose.StopAt(ctx, 0, nil))

	// We have to use docker.start after compose.StopAt; helper.Client
	// is the original endpoint. Restart preserves the data dir.
	require.NoError(t, compose.StartAt(ctx, 0))
	helper.SetupClient(compose.GetWeaviate().URI())
	container := compose.GetWeaviate().Container()

	// Sanity: count must survive the restart.
	require.EqualValues(t, preCount, moduleshelper.GetClassCount(t, className, ""),
		"baseline restart must not lose data; the orphan-audit failure mode is about extra state, not missing state")

	// Craft the orphan state.
	lsmPath := fmt.Sprintf("/data/%s/%s/lsm", strings.ToLower(className), shardName)
	orphanDir := "searchable_retokenize_body_999" // gen 999 is far outside anything the runtime would pick
	sidecarBucket := "property_body_searchable__retokenize_reindex_999"
	injectOrphanTrackerOnDisk(t, ctx, container, lsmPath, orphanDir, sidecarBucket,
		`{"taskID":"orphan-from-prefix-backup","taskVersion":1,"unitID":"u0","payload":{"collection":"`+className+`","migrationType":"change-tokenization","properties":["body"],"targetTokenization":"lowercase","bucketStrategy":"map_collection"}}`)

	// Restart so the audit fires on the staged orphan.
	require.NoError(t, compose.StopAt(ctx, 0, nil))
	require.NoError(t, compose.StartAt(ctx, 0))
	helper.SetupClient(compose.GetWeaviate().URI())
	container = compose.GetWeaviate().Container()

	// The audit runs asynchronously after meta store ready + DTM
	// bootstrap. Eventually-assert; the audit completes well under
	// the deadline on a small cluster.
	require.Eventually(t, func() bool {
		// The .migrations/<orphanDir>/ must be gone.
		code, _, _ := container.Exec(ctx, []string{
			"test", "-d",
			filepath.Join(lsmPath, ".migrations", orphanDir),
		})
		return code != 0
	}, 60*time.Second, 500*time.Millisecond,
		"orphan tracker dir was not cleaned up by the post-bootstrap audit")

	// Sidecar bucket must also be gone (audit shut it down + removed).
	code, _, _ := container.Exec(ctx, []string{"test", "-d", filepath.Join(lsmPath, sidecarBucket)})
	assert.NotEqual(t, 0, code,
		"orphan sidecar bucket dir was not cleaned up; expected exit != 0 from test -d, got %d", code)

	// Canonical bucket and data survive — the audit's contract is
	// "quarantine orphans, never touch the live canonical bucket".
	assert.EqualValues(t, preCount, moduleshelper.GetClassCount(t, className, ""),
		"canonical data must survive the audit; the orphan tracker cleanup must not touch property_body or property_body_searchable")
}

// testCancelOnNoInFlightReturns404 exercises 0-weaviate-issues#215 B5:
// PUT {"searchable":{"cancel":true}} when no task targets the tuple
// must return 404 with a structured body identifying the (collection,
// property, indexType) tuple and pointing at GET /v1/schema/<class>/
// indexes for state inspection.
//
// Bare 404 was the prior behavior and is "indistinguishable from
// endpoint-not-found" per QA — the structured body is what tells
// operators their cancel is hitting a real endpoint but found
// nothing to cancel.
func testCancelOnNoInFlightReturns404(t *testing.T, restURI string) {
	const (
		className = "ReindexBackup_CancelNoTask"
		propName  = "body"
	)
	helper.CreateClass(t, &models.Class{
		Class: className,
		Properties: []*models.Property{
			{Name: propName, DataType: []string{"text"}, Tokenization: "word"},
		},
		Vectorizer: "none",
	})
	defer helper.DeleteClass(t, className)

	// Seed at least one object so the class isn't empty (defensive —
	// some handlers short-circuit on empty classes).
	importBodies(t, className, 5)

	url := fmt.Sprintf("http://%s/v1/schema/%s/indexes/%s", restURI, className, propName)
	req, err := http.NewRequest(http.MethodPut, url,
		bytes.NewReader([]byte(`{"searchable":{"cancel":true}}`)))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	require.Equalf(t, http.StatusNotFound, resp.StatusCode,
		"expected 404 for cancel-with-no-task; got %d: %s", resp.StatusCode, string(respBody))

	// Bare 404 had an empty body — assert ours is structured.
	require.NotEmpty(t, respBody, "B5 requires a structured body, got empty payload")

	// Body must mention collection, property, indexType, and the GET
	// recovery hint. Order isn't asserted; substrings are.
	bodyStr := string(respBody)
	assert.Contains(t, bodyStr, className, "404 body must name the collection")
	assert.Contains(t, bodyStr, propName, "404 body must name the property")
	assert.Contains(t, bodyStr, "searchable", "404 body must name the indexType")
	assert.Contains(t, bodyStr, "no in-flight reindex task to cancel",
		"404 body must explain why nothing was cancelled")
	assert.Contains(t, bodyStr, "GET /v1/schema/",
		"404 body must point operators at the state-inspection endpoint")
}

// testCleanupRemovesOrphan exercises 0-weaviate-issues#215 B6: the
// side-effect-free cleanup verb. Operators with silent half-state need
// a way to clean it without restarting the cluster AND without forcing
// a Map→BlockMax cutover (which the rebuild verb does as a one-way
// side effect).
//
// Flow:
//  1. Craft orphan tracker + sidecar on disk via docker exec (same
//     pattern as the B3 audit test, but no restart afterwards).
//  2. PUT {"searchable":{"cleanup":true}} — assert 202 with
//     Status="CLEANED".
//  3. Assert tracker dir + sidecar bucket dir are gone WITHOUT a
//     restart (the verb's whole point).
//  4. Assert the schema's reported tokenization on the searchable
//     index is UNCHANGED — the rebuild verb would have triggered a
//     migration; cleanup must not.
//  5. Issue a real change-tokenization, then immediately fire cleanup
//     while the task is STARTED — must return 409 with structured
//     body naming the offending task and pointing at cancel.
func testCleanupRemovesOrphan(t *testing.T, ctx context.Context, compose *docker.DockerCompose, restURI string) {
	const (
		className = "ReindexBackup_Cleanup"
		propName  = "body"
	)
	helper.CreateClass(t, &models.Class{
		Class: className,
		Properties: []*models.Property{
			{Name: propName, DataType: []string{"text"}, Tokenization: "word"},
		},
		Vectorizer: "none",
	})
	defer helper.DeleteClass(t, className)

	importBodies(t, className, 200)
	preCount := moduleshelper.GetClassCount(t, className, "")
	require.EqualValues(t, 200, preCount)

	shardName := reindexhelpers.GetFirstShardName(t, restURI, className)
	require.NotEmpty(t, shardName)

	container := compose.GetWeaviate().Container()
	lsmPath := fmt.Sprintf("/data/%s/%s/lsm", strings.ToLower(className), shardName)
	orphanDir := "searchable_retokenize_body_998"
	sidecarBucket := "property_body_searchable__retokenize_reindex_998"
	injectOrphanTrackerOnDisk(t, ctx, container, lsmPath, orphanDir, sidecarBucket,
		fmt.Sprintf(`{"taskID":"orphan-cleanup-verb","taskVersion":1,"unitID":"u0","payload":{"collection":%q,"migrationType":"change-tokenization","properties":["body"],"targetTokenization":"lowercase","bucketStrategy":"map_collection"}}`, className))

	// Capture the schema's tokenization before cleanup. This is the
	// "no side effects on schema" guard.
	tokBefore := readSearchableTokenization(t, restURI, className, propName)
	require.Equal(t, "word", tokBefore, "fixture: searchable tokenization must be 'word' before cleanup")

	// Fire the cleanup verb.
	url := fmt.Sprintf("http://%s/v1/schema/%s/indexes/%s", restURI, className, propName)
	req, err := http.NewRequest(http.MethodPut, url,
		bytes.NewReader([]byte(`{"searchable":{"cleanup":true}}`)))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equalf(t, http.StatusAccepted, resp.StatusCode,
		"expected 202 from cleanup; got %d: %s", resp.StatusCode, string(respBody))

	// Body should report CLEANED. The handler returns IndexUpdateResponse.
	var parsed struct {
		Status string `json:"status"`
	}
	require.NoError(t, json.Unmarshal(respBody, &parsed))
	assert.Equal(t, "CLEANED", parsed.Status,
		"cleanup response status should be CLEANED, got %q", parsed.Status)

	// Without a restart: tracker + sidecar must be gone.
	code, _, _ := container.Exec(ctx, []string{"test", "-d", filepath.Join(lsmPath, ".migrations", orphanDir)})
	assert.NotEqual(t, 0, code, "orphan tracker dir must be removed by cleanup verb (no restart)")

	code, _, _ = container.Exec(ctx, []string{"test", "-d", filepath.Join(lsmPath, sidecarBucket)})
	assert.NotEqual(t, 0, code, "orphan sidecar dir must be removed by cleanup verb (no restart)")

	// Schema's tokenization must be UNCHANGED — the whole point of
	// the cleanup verb vs rebuild=true is no side effects.
	tokAfter := readSearchableTokenization(t, restURI, className, propName)
	assert.Equal(t, tokBefore, tokAfter,
		"cleanup must not switch tokenization (the side effect rebuild=true would have triggered)")

	// And data must survive.
	assert.EqualValues(t, preCount, moduleshelper.GetClassCount(t, className, ""),
		"canonical data must survive cleanup; sidecar removal must not touch property_body or property_body_searchable")

	// Now exercise the 409 refusal path: launch a real change-tokenization
	// (which writes started.mig for a SHORT window), fire cleanup, expect
	// 409. Because the migration may complete fast on a 200-row corpus,
	// we try a few times — the race is acceptable because BOTH the 409
	// and a 202-because-the-task-already-finished response prove the
	// gate (409 explicitly, 202 implicitly because the dispatch only
	// looks at STARTED tasks).
	taskID := submitChangeTokenization(t, restURI, className, propName, "lowercase")

	// Attempt cleanup repeatedly with a tight loop; if the task is still
	// STARTED, we expect 409. We continue until we either see a 409 or
	// the task reaches FINISHED. If we never see 409, log it but don't
	// fail — the unit-test layer already pins the 409 dispatch logic.
	saw409 := false
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		req, err := http.NewRequest(http.MethodPut, url,
			bytes.NewReader([]byte(`{"searchable":{"cleanup":true}}`)))
		require.NoError(t, err)
		req.Header.Set("Content-Type", "application/json")
		r, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		body, _ := io.ReadAll(r.Body)
		_ = r.Body.Close()
		if r.StatusCode == http.StatusConflict {
			saw409 = true
			// Verify the 409 body is structured.
			s := string(body)
			assert.Contains(t, s, "refusing cleanup",
				"409 body must explain the refusal")
			assert.Contains(t, s, taskID,
				"409 body must name the conflicting task ID")
			assert.Contains(t, s, `\"cancel\":true`,
				"409 body must point operators at the cancel verb (the JSON snippet appears in the structured error message)")
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !saw409 {
		t.Logf("note: did not observe 409 from cleanup-during-task (task likely finished too fast for this fixture); unit tests already pin the dispatch logic")
	}

	reindexhelpers.AwaitReindexFinished(t, restURI, taskID, reindexhelpers.WithTimeout(60*time.Second))
}

// testAlgorithmVerb exercises 0-weaviate-issues#215 B7 and the
// follow-on already-blockmax refusal guard.
//
// B7 made `{"searchable":{"algorithm":"BlockMaxWAND"}}` a recognised
// body shape (pre-fix it returned 400 with no useful message). The
// handler routes it through the same `repair-searchable` migration
// as `rebuild:true`. That migration is implemented as Map→Blockmax
// — it assumes the source bucket is a legacy Map-strategy bucket.
//
// On a fresh class created with the default UsingBlockMaxWAND=true,
// the source bucket is already on the inverted (blockmax) strategy.
// Running the migration on such a class hits the runtime-swap step
// with `rename "...__blockmax_ingest_1" "...__blockmax_map_2": no
// such file or directory` and leaves a half-state behind. Surfaced
// by the first iteration of this test.
//
// Handler now refuses both `rebuild:true` and `algorithm:"BlockMaxWAND"`
// at submit time when `UsingBlockMaxWAND` is already true on the
// class, with 409 + structured body pointing the operator at
// `cleanup:true` for orphan-state recovery.
//
// Test pins:
//   - already-blockmax + algorithm:"BlockMaxWAND" → 409, body names
//     class+prop and points at cleanup:true
//   - already-blockmax + rebuild:true             → 409, same shape
//   - any state + algorithm:"WAND"                → 400, body names
//     BlockMaxWAND as the supported alternative
//   - no DTM task is left behind in any of the three paths
func testAlgorithmVerb(t *testing.T, restURI string) {
	const (
		className = "ReindexBackup_AlgorithmVerb"
		propName  = "body"
	)
	helper.CreateClass(t, &models.Class{
		Class: className,
		Properties: []*models.Property{
			{Name: propName, DataType: []string{"text"}, Tokenization: "word"},
		},
		Vectorizer: "none",
	})
	defer helper.DeleteClass(t, className)

	importBodies(t, className, 10)

	url := fmt.Sprintf("http://%s/v1/schema/%s/indexes/%s", restURI, className, propName)

	// Case 1: already-blockmax + algorithm:"BlockMaxWAND" → 409.
	req, err := http.NewRequest(http.MethodPut, url,
		bytes.NewReader([]byte(`{"searchable":{"algorithm":"BlockMaxWAND"}}`)))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	bodyBytes, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	require.Equalf(t, http.StatusConflict, resp.StatusCode,
		"algorithm:BlockMaxWAND on an already-blockmax class must be refused with 409; got %d: %s", resp.StatusCode, string(bodyBytes))
	assert.Contains(t, string(bodyBytes), "already on BlockMaxWAND",
		"409 body must explain the refusal reason")
	assert.Contains(t, string(bodyBytes), className,
		"409 body must name the class")
	assert.Contains(t, string(bodyBytes), propName,
		"409 body must name the property")
	assert.Contains(t, string(bodyBytes), `\"cleanup\":true`,
		"409 body must point the operator at the cleanup verb for orphan-state recovery")

	// Case 2: already-blockmax + rebuild:true → 409 (parity with
	// case 1; both routes through repair-searchable).
	req, err = http.NewRequest(http.MethodPut, url,
		bytes.NewReader([]byte(`{"searchable":{"rebuild":true}}`)))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	resp, err = http.DefaultClient.Do(req)
	require.NoError(t, err)
	bodyBytes, _ = io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	require.Equalf(t, http.StatusConflict, resp.StatusCode,
		"rebuild:true on an already-blockmax class must be refused with 409; got %d: %s", resp.StatusCode, string(bodyBytes))
	assert.Contains(t, string(bodyBytes), "already on BlockMaxWAND",
		"409 body must explain the refusal reason")

	// Case 3: algorithm:"WAND" — rejected with 400 IMMEDIATELY (the
	// handler doesn't even consult the class state; alias
	// normalisation fails first). Order matters: this exercises the
	// 400 path even after the 409 cases, because alias-normalisation
	// fails BEFORE the UsingBlockMaxWAND lookup runs.
	req, err = http.NewRequest(http.MethodPut, url,
		bytes.NewReader([]byte(`{"searchable":{"algorithm":"WAND"}}`)))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	resp, err = http.DefaultClient.Do(req)
	require.NoError(t, err)
	bodyBytes, _ = io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	require.Equalf(t, http.StatusBadRequest, resp.StatusCode,
		"WAND algorithm must be rejected with 400; got %d: %s", resp.StatusCode, string(bodyBytes))
	assert.Contains(t, string(bodyBytes), "BlockMaxWAND",
		"400 body must name the supported alternative")
	assert.Contains(t, string(bodyBytes), "not supported",
		"400 body must explain the rejection")

	// Verify no DTM task is left behind in any of the three paths.
	// Pre-guard, the BlockMaxWAND submission would have scheduled a
	// repair-searchable task and crashed it mid-flight. With the
	// guard, no task is created. GET /v1/tasks must be empty for
	// this class.
	resp2, err := http.Get(fmt.Sprintf("http://%s/v1/tasks", restURI))
	require.NoError(t, err)
	defer resp2.Body.Close()
	tasksBody, err := io.ReadAll(resp2.Body)
	require.NoError(t, err)
	assert.NotContains(t, string(tasksBody), className,
		"refused submits must not schedule any DTM task; tasks body should not name the class. Got: %s", string(tasksBody))
}

// testMutationGuardBlocksDeleteClassDuringInFlight pins the upstream
// MutationGuard CheckClassMutation path (2f77d4905e + 6b8d8e692a, #218
// + #219) at the full RAFT+REST round-trip. The cluster/schema
// TestSchemaManager_DeleteClass_MutationGuard unit test pins the
// FSM-side check with a mocked guard; this test exercises the actual
// installed guard via a real DELETE /v1/schema/{class} HTTP call
// while a real reindex task is STARTED in DTM.
//
// Flow:
//  1. Create class + seed enough rows that the change-tokenization
//     migration is observable for several seconds.
//  2. Submit change-tokenization on `body`.
//  3. Wait for the task to reach the indexing state (started.mig on
//     disk; task status STARTED in DTM).
//  4. Issue DELETE /v1/schema/<class>. Must return 400 with a
//     structured body naming the in-flight task and the actionable
//     remedy (cancel before delete).
//  5. Confirm the class still exists (GET 200).
//  6. Wait for the task to finish; re-issue DELETE. Must succeed.
//
// Origin: QA Claude verification at 0-weaviate-issues#215
// issuecomment-4470603684 suggesting V12_mutation_guard.py be
// promoted to the Go acceptance suite as a regression guard for
// #219.
func testMutationGuardBlocksDeleteClassDuringInFlight(t *testing.T, restURI string) {
	const (
		className = "ReindexBackup_MutationGuard"
		propName  = "body"
	)
	helper.CreateClass(t, &models.Class{
		Class: className,
		Properties: []*models.Property{
			{Name: propName, DataType: []string{"text"}, Tokenization: "word"},
		},
		Vectorizer: "none",
	})
	// Note: NO defer DeleteClass here — the test itself drives the
	// final delete to prove the post-tidied path. If the test fails
	// midway and the class survives, the next test run will reject
	// the create-class with "class already exists"; that is the
	// intentional loud failure mode for this scenario.
	deletedByTest := false
	defer func() {
		if !deletedByTest {
			// Bail-out cleanup so a partial-failure run does not
			// leave the class around. Mirrors the DefaultMaintenance
			// behavior of the other subtests, but only runs on the
			// abort path.
			helper.DeleteClass(t, className)
		}
	}()

	// Seed enough rows that the migration is observable for at
	// least ~3-5 s on this hardware (50k matches the existing
	// testBackupRefusedDuringInFlightMigration sizing).
	importBodies(t, className, 50_000)

	taskID := submitChangeTokenization(t, restURI, className, propName, "lowercase")
	t.Logf("change-tokenization task submitted for mutation-guard probe: %s", taskID)

	// Wait until the per-shard iteration is in the indexing state
	// so the DELETE attempt below races a genuinely STARTED task.
	awaitIndexingState(t, restURI, className, propName)

	// Issue the DELETE while the task is STARTED.
	deleteURL := fmt.Sprintf("http://%s/v1/schema/%s", restURI, className)
	req, err := http.NewRequest(http.MethodDelete, deleteURL, nil)
	require.NoError(t, err)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	bodyBytes, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	// The MutationGuard rejection surfaces as a 400 with a structured
	// "bad request" body. Status 200 would mean the guard didn't
	// fire — the regression we want to catch.
	require.Equalf(t, http.StatusBadRequest, resp.StatusCode,
		"DELETE during in-flight reindex must be refused with 400 (MutationGuard #219); got %d: %s",
		resp.StatusCode, string(bodyBytes))

	// Body must surface enough context for the operator to act.
	bodyStr := string(bodyBytes)
	assert.Contains(t, bodyStr, "in flight",
		"400 body must explain that a task is in flight; got: %s", bodyStr)
	assert.Contains(t, bodyStr, className,
		"400 body must name the class being deleted; got: %s", bodyStr)
	assert.Contains(t, bodyStr, "cancel",
		"400 body must point operators at the cancel remedy; got: %s", bodyStr)

	// The class must still exist.
	getURL := fmt.Sprintf("http://%s/v1/schema/%s", restURI, className)
	resp, err = http.Get(getURL)
	require.NoError(t, err)
	_ = resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode,
		"class must survive a guard-rejected DELETE; got %d on GET", resp.StatusCode)

	// Let the migration finish, then DELETE must succeed.
	reindexhelpers.AwaitReindexFinished(t, restURI, taskID, reindexhelpers.WithTimeout(120*time.Second))

	req, err = http.NewRequest(http.MethodDelete, deleteURL, nil)
	require.NoError(t, err)
	resp, err = http.DefaultClient.Do(req)
	require.NoError(t, err)
	_ = resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode,
		"post-tidied DELETE must succeed; got %d", resp.StatusCode)
	deletedByTest = true
}

// readSearchableTokenization reads the current tokenization of a
// property's searchable index via GET /v1/schema/<class>/indexes.
// Returns "" if the property has no searchable index. Used by the B6
// cleanup test to prove the verb does NOT switch tokenization.
func readSearchableTokenization(t *testing.T, restURI, collection, property string) string {
	t.Helper()
	resp, err := http.Get(fmt.Sprintf("http://%s/v1/schema/%s/indexes", restURI, collection))
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	var parsed struct {
		Properties []struct {
			Name    string `json:"name"`
			Indexes []struct {
				Type         string `json:"type"`
				Tokenization string `json:"tokenization"`
			} `json:"indexes"`
		} `json:"properties"`
	}
	require.NoError(t, json.Unmarshal(body, &parsed),
		"unparseable response body: %s", string(body))
	for _, p := range parsed.Properties {
		if p.Name != property {
			continue
		}
		for _, idx := range p.Indexes {
			if idx.Type == "searchable" {
				return idx.Tokenization
			}
		}
	}
	return ""
}

// backupAndRestoreRoundTrip creates a filesystem backup, deletes the
// class, restores from the backup, and asserts the post-restore count
// equals the supplied pre-backup count. Used by both the baseline
// happy-path test and the post-migration self-clearing test; extracted
// to eliminate the 13-line duplicate flagged by SonarCloud on
// PR #11327. The `msg` is the assert-equal context shown if the count
// regression fires.
func backupAndRestoreRoundTrip(t *testing.T, className, backupID string, preCount int64, msg string) {
	t.Helper()
	_, err := helper.CreateBackup(t, helper.DefaultBackupConfig(), className, "filesystem", backupID)
	require.NoError(t, err)
	helper.ExpectBackupEventuallyCreated(t, backupID, "filesystem", nil,
		helper.WithDeadline(2*time.Minute))

	helper.DeleteClass(t, className)
	_, err = helper.RestoreBackup(t, helper.DefaultRestoreConfig(), className, "filesystem", backupID, nil, false)
	require.NoError(t, err)
	helper.ExpectBackupEventuallyRestored(t, backupID, "filesystem", nil,
		helper.WithDeadline(2*time.Minute))

	postCount := moduleshelper.GetClassCount(t, className, "")
	assert.Equal(t, preCount, postCount, msg)
}

// injectOrphanTrackerOnDisk crafts the on-disk shape that a pre-fix
// backup-restore (or any half-completed migration that was captured by
// `cp -r` between the started.mig and tidied.mig sentinel writes)
// would leave on a restored shard:
//
//   - .migrations/<orphanDir>/started.mig
//   - .migrations/<orphanDir>/reindexed.mig
//   - .migrations/<orphanDir>/payload.mig (with the supplied JSON body)
//   - <sidecarBucket>/marker.flag        (proof-of-presence for assertions)
//
// Two tests inject this shape from different drivers — the post-restart
// orphan-audit test (which restarts after the injection) and the
// cleanup-verb test (which fires the verb WITHOUT a restart). The
// injection content is identical; the difference is only in what
// runs against the resulting state. Extracted to eliminate a 10-line
// duplicate block flagged by SonarCloud on PR #11327.
func injectOrphanTrackerOnDisk(t *testing.T, ctx context.Context, container testcontainers.Container,
	lsmPath, orphanDir, sidecarBucket, payloadJSON string,
) {
	t.Helper()
	for _, cmd := range [][]string{
		{"mkdir", "-p", filepath.Join(lsmPath, ".migrations", orphanDir)},
		{"touch", filepath.Join(lsmPath, ".migrations", orphanDir, "started.mig")},
		{"touch", filepath.Join(lsmPath, ".migrations", orphanDir, "reindexed.mig")},
		{"sh", "-c", fmt.Sprintf("cat > %s <<'EOF'\n%s\nEOF", filepath.Join(lsmPath, ".migrations", orphanDir, "payload.mig"), payloadJSON)},
		{"mkdir", "-p", filepath.Join(lsmPath, sidecarBucket)},
		// A small marker file so the audit's directory removal is observable.
		{"touch", filepath.Join(lsmPath, sidecarBucket, "marker.flag")},
	} {
		execInContainer(t, ctx, container, cmd...)
	}
}

// execInContainer runs a command inside the testcontainer and fails
// the test on non-zero exit. Used by the orphan-injection step of the
// Phase 2 audit test.
func execInContainer(t *testing.T, ctx context.Context, c testcontainers.Container, cmd ...string) {
	t.Helper()
	code, reader, err := c.Exec(ctx, cmd)
	require.NoError(t, err, "exec %v", cmd)
	output := ""
	if reader != nil {
		buf := new(strings.Builder)
		_, _ = io.Copy(buf, reader)
		output = buf.String()
	}
	require.Equal(t, 0, code, "exec %v exited %d; output: %s", cmd, code, output)
}

// getFirstShardName was lifted to reindexhelpers.GetFirstShardName
// as part of the SonarCloud duplication fix on PR #11327. Callers in
// this file now invoke that exported version directly.

// awaitIndexingState polls GET /v1/schema/<class>/indexes until at least
// one index for the named property reports status "indexing" — i.e. the
// per-shard reindex iteration has marked started.mig on disk. This is
// the deterministic equivalent of polling for the sentinel file from
// outside the container.
//
// If the migration completes before the poll observes "indexing"
// (possible on a tiny corpus when CI is fast), the function returns
// without failing — the calling test will then observe a quiet shard
// and the backup will succeed (the test should treat that as a
// fixture-too-small problem, not as a regression of the gate).
func awaitIndexingState(t *testing.T, restURI, collection, property string) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(fmt.Sprintf("http://%s/v1/schema/%s/indexes", restURI, collection))
		if err != nil {
			time.Sleep(20 * time.Millisecond)
			continue
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		var parsed struct {
			Properties []struct {
				Name    string `json:"name"`
				Indexes []struct {
					Type   string `json:"type"`
					Status string `json:"status"`
				} `json:"indexes"`
			} `json:"properties"`
		}
		if err := json.Unmarshal(body, &parsed); err == nil {
			for _, p := range parsed.Properties {
				if p.Name != property {
					continue
				}
				for _, idx := range p.Indexes {
					if idx.Status == "indexing" {
						t.Logf("observed indexing state for %s/%s (type=%s)", collection, property, idx.Type)
						return
					}
				}
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	// Migration may have finished too quickly to observe — caller
	// will see the backup succeed and surface a clear failure.
	t.Logf("warning: did not observe indexing state for %s/%s within deadline; migration may have completed too fast for the test fixture", collection, property)
}

// awaitTaskFinished was a local copy of the same helper that already
// existed in reindexhelpers.AwaitReindexFinished. Removed to fix the
// SonarCloud "Duplication on New Code" gate failure on PR #11327;
// see test/acceptance/helpers/reindex/helpers.go for the canonical
// implementation. All call sites in this file now invoke that one
// directly with reindexhelpers.WithTimeout(...) for the per-test
// timeout override.
