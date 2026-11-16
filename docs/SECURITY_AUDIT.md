# Security audit

Pre-publication security and privacy review of Blood Cell VisionOps, run before
the repository's first public release. This report contains no secret values.

## 1. Executive summary

**Recommendation: BLOCKED — SECURITY FIXES REQUIRED.** The repository itself is in good shape, but one
Critical issue outside the code must be resolved first.

- **No credentials were found in tracked files or in any reachable Git history** (Gitleaks plus manual
  checks over all refs).
- **One Critical issue outside the repository:** a Gemini API key was exposed in a chat transcript
  during development and is still active. It was never committed, but it must be revoked and replaced
  before release.
- **Fixed in this change:** 5 High/Medium application issues (cross-site access to the local API, LLM
  output able to carry links/HTML into the dashboard, decompression bombs and oversized images,
  event-loop blocking, unpickling of unverified model files) and several hardening items.
- **Documented, not fixed:** demo-grade deployment (no authentication, default database password,
  localhost-only), unpatched OS packages in the Debian and Postgres base images (no upstream fix yet), and
  local paths in model files that remain in the existing private Git history.

## 2. Scope and limitations

**In scope:** the full repository at the audited commit and its reachable history; the private GitHub
repository (branches, tags, releases, artifacts, Actions settings); the local working tree, including
ignored files; the three running project containers and their images; Python dependencies; the CI
workflow; the Gemini integration.

**Not in scope / not possible:**
- No penetration test of a deployed service, no load testing, no fuzzing campaign.
- GitHub secret scanning and Dependabot alerts are not enabled on the private repository, so no
  GitHub-side scan results exist.
- The exposed key's usage history cannot be inspected from here (needs the Google account).
- Findings are point-in-time. New CVEs will appear in the base images and dependencies.

## 3. Findings by severity

| ID | Severity | Finding | Status |
|---|---|---|---|
| S1 | CRITICAL | Exposed Gemini API key still active | **Open: owner must rotate** |
| S2 | HIGH | Any website could read and write the local API through the browser (CORS `*`) | Fixed |
| S3 | HIGH | LLM narrative could carry links, images, HTML or `javascript:` URLs into the dashboard; the user-supplied filename was sent to the LLM unsanitised | Fixed |
| S4 | MEDIUM | Decompression bomb caused HTTP 500; 81-megapixel images were decoded; uploads were fully read before the size check | Fixed |
| S5 | MEDIUM | Inference and the Gemini call (up to 45 s) ran on the event loop, stalling all requests, including `/health` | Fixed |
| S6 | MEDIUM | `.pt` checkpoints are unpickled by Ultralytics (`weights_only=False`) with no integrity check | Fixed |
| S7 | MEDIUM | Model files embedded the developer's absolute home-directory path (OS username) and local git info | Fixed in current files; **remains in existing private history** |
| S8 | MEDIUM | Outdated pip/setuptools/zlib in our images (fixable CVEs) | Fixed |
| S9 | MEDIUM | Base-image OS packages with no upstream fix (Debian 13, Postgres Alpine Go runtime) | Documented |
| S10 | MEDIUM | Demo deployment: no auth or rate limiting, default DB password, no container resource limits | Documented (localhost-only by design) |
| S11 | LOW | CI: no explicit token permissions, actions pinned by tag, checkout kept credentials, no Dependabot | Fixed |
| S12 | LOW | Dashboard displayed raw database/HTTP exception text; QC status inserted into raw HTML unvalidated | Fixed |
| S13 | LOW | Unpinned Python dependencies and floating base-image tags (image ≠ audited versions) | Documented; Dependabot added |
| S14 | LOW | Local-only Git refs still contain the old personal commit email (never pushed) | **Open: owner approval to delete** |
| S15 | LOW | `xml.etree` parsing of dataset annotations (Bandit B314) | Accepted: trusted, pinned dataset |
| S16 | INFO | Plain-text prompt injection via filename (e.g. "ignore previous instructions") cannot be removed by sanitising | Mitigated: no LLM tools, claim check, code-rendered tables |

### Details

**S1: exposed Gemini key (CRITICAL).** *Where:* outside the repository. The key was pasted into a chat
log, and it is also in the git-ignored local `.env` file. *Evidence:* a model-list request with that key
succeeded during the audit. Gitleaks found it only in `.env` (ignored, never committed), and 0 times in
history. *Impact:* anyone with the transcript can spend the account's quota. *Remediation:* revoke it in
Google AI Studio, create a new key, and put it only in `.env`. *Verify:* the old key is rejected and the
new key lists models.

**S2: cross-site access to the local API (HIGH).** *Where:* `bloodcell/api.py` (was
`CORSMiddleware(allow_origins=["*"])`). *Evidence:* a preflight from `https://evil.example` returned
`Access-Control-Allow-Origin: *`, and a cross-site POST was accepted. *Impact:* any web page open in the
user's browser could read stored analyses and images, create records, and spend the Gemini quota through
`localhost`. *Fix:* the CORS middleware was removed (the dashboard calls the API server-side), and any
request with a foreign `Origin` header now gets 403. Same-origin requests (Swagger UI), server-side
requests and `BCV_ALLOWED_ORIGINS` still work. *Verify:* `tests/test_security.py`, plus a live check
(cross-origin 403, same-origin 200).

**S3: markup and prompt injection through the LLM (HIGH).** *Where:* `bloodcell/report_validation.py`,
`bloodcell/api.py`. *Evidence:* narratives containing `[x](https://…)`, `![](…)`, `<img onerror=…>`,
`javascript:` URLs and bare URLs all passed the claim check, and those reports are rendered as Markdown in
the dashboard. Filenames like `<img src=x onerror=alert(1)>.jpg` were stored and sent to Gemini
unchanged. *Fix:* the checker now rejects links, images, URLs and HTML. Filenames are reduced to a safe
character set, stripped of directories and capped at 100 characters. *Verify:* 7 markup cases and 7
filename cases in `tests/test_security.py`.

**S4: image-based denial of service (MEDIUM).** *Evidence:* a 312-byte PNG declaring 900 megapixels
returned 500, and an 81-megapixel image was decoded (about 240 MB of RAM). *Fix:* reads stop at the
upload limit plus 1 byte; a decoded-size cap applies (`BCV_MAX_PIXELS`, default 40 MP); Pillow's bomb
error maps to 413. *Verify:* tests, plus the live probes now returning 413.

**S5: event-loop blocking (MEDIUM).** *Fix:* inference, the Gemini call and the database write run in
a worker thread (`run_in_threadpool`). *Verify:* a test runs a 1 s inference and confirms `/health`
still answers in under 0.5 s.

**S6: unverified checkpoint deserialisation (MEDIUM).** *Fix:* before loading, `Detector` checks the
file's SHA-256 against `models/model_info.json` and refuses a mismatch. The Docker API serves the ONNX
file, which is not pickled. *Verify:* tests cover a tampered file (refused) and the served files (pass).

**S7: personal paths in model files (MEDIUM).** *Evidence:* the `.pt` training args, the pickled
module's args and `pt_path`, the git info, and the ONNX `description` metadata held the developer's
absolute project path, including the OS username. *Fix:* `scripts/promote_model.py` now scrubs these on
every promotion, and the current files were scrubbed. Predictions are byte-identical before and after,
and the hashes in `model_info.json` were updated. *Remaining:* earlier commits on the private remote
still contain the old files. Removing them needs a history rewrite and a force-push (owner decision).
*Verify:* a test asserts no `/Users/` or `/home/` bytes in either model file.

**S8/S9: container vulnerabilities.** Trivy 0.76.2, CRITICAL/HIGH/MEDIUM, after the fixes:

| Image | Critical | High | Medium | Fixable |
|---|---|---|---|---|
| API (Debian 13 + Python 3.11) | 1 | 122 | 155 | 0 C / 1 H / 1 M |
| Dashboard (Debian 13 + Python 3.11) | 0 | 48 | 93 | 0 C / 1 H / 1 M |
| `postgres:16-alpine` (upstream, latest digest) | 1 | 25 | 31 | all, upstream only |

The API image's Critical finding (`libxml2`) and nearly all High findings are Debian packages with no
fixed version published yet. Fixed by this change: Debian security updates are applied at build time,
and pip/setuptools/wheel are upgraded (which removes the vendored `wheel` and `jaraco.context` CVEs).
The remaining "fixable" item is a `setuptools 70.3.0` record that Trivy reports without a file path. The
image actually has setuptools 84.2.1 installed, plus a 79.0.1 wheel bundled with Python's `ensurepip`,
so it is treated as a probable false positive. The Postgres findings are in the Go runtime of a helper
binary inside the official image, and the local image already matches the newest published digest. All
three containers bind to `127.0.0.1` only.

**S10: demo deployment (MEDIUM, documented).** Verified on the running containers: ports bind to
127.0.0.1, the app containers run as a non-root `appuser`, nothing is privileged, no capabilities are
added, there is no Docker socket mount, and host mounts are read-only. There is no authentication, rate
limiting or resource limit, and the database password is a documented default. This is acceptable only
for a localhost demo, and the README says so.

**S11: CI hardening (LOW).** *Evidence:* the repository's default token was already read-only and no
secrets are used. The workflow had no untrusted `${{ }}` interpolation, no `pull_request_target` and no
artifacts. *Fix:* explicit `permissions: contents: read`, actions pinned to commit SHAs,
`persist-credentials: false`, job timeouts, `concurrency`, and `.github/dependabot.yml` covering pip,
Docker and Actions.

**S12: dashboard (LOW).** *Fix:* database and HTTP errors show a generic message, and the QC badge only
renders known status values. The LLM report goes through `st.markdown` without `unsafe_allow_html`, so
raw HTML is escaped; with S3, links and images are rejected upstream as well.

## 4. Fixes implemented

`bloodcell/api.py`, `bloodcell/config.py`, `bloodcell/report_validation.py`, `bloodcell/inference.py`,
`dashboard/app.py`, `scripts/promote_model.py`, `models/best.pt`, `models/best.onnx` (metadata only),
`models/model_info.json`, `docker/api.Dockerfile`, `docker/dashboard.Dockerfile`,
`.github/workflows/ci.yml`, `.github/dependabot.yml`, `.env.example`, `README.md`, and new regression
tests in `tests/test_security.py`. No tests were weakened or removed.

## 5. Tests and scans

| Check | Tool | Result |
|---|---|---|
| Secret scan, all reachable history (all refs) | Gitleaks 8.30.1 | PASS: 0 findings |
| Secret scan, working tree | Gitleaks 8.30.1 | PASS for tracked files. Flags only the git-ignored `.env` (S1) and two false positives in third-party source maps inside ignored virtualenvs. |
| Manual: credential URLs, passwords, tokens, keys in text, model binaries, `.npz` arrays | grep / strings / numpy | PASS: only the documented local-dev default DB password and test placeholders |
| Personal data in pushed history | git log | PASS for email (GitHub no-reply only). Paths: see S7. |
| Image metadata (EXIF and text chunks) | Pillow | PASS: no EXIF; only Matplotlib "Software" tags |
| Python dependencies (local, API image, dashboard image) | pip-audit 2.10.1 | PASS: no known vulnerabilities. NOT VERIFIED for `torch`/`torchvision` CPU builds (not on PyPI; covered by Trivy). |
| Container images | Trivy 0.76.2 | FAIL with no fix available upstream (see S9) |
| Static analysis | Bandit 1.9.4 | PASS with 2 accepted Medium: S15 (XML), and B614 in `scripts/promote_model.py`, which unpickles the checkpoint that training just produced on the same machine, before any hash exists for it. 2 Low (one false positive). |
| Lint | ruff | PASS |
| Automated tests | pytest | PASS: 98 passed |
| Live API probes (bomb, oversize, spoofed type, truncated image, traversal, markup, cross-origin) | local, non-destructive | PASS after fixes |
| Database preserved across rebuild | row count, max id, checksum | PASS: identical |
| GitHub-side alerts | secret scanning / Dependabot | NOT VERIFIED: not enabled on the private repo |

## 6. Remaining risks

- **S1** until the key is rotated.
- **S7** in the existing private history (local paths in old model files) until the history is rewritten,
  or until the owner accepts it.
- **S9:** unpatched base-image packages. Rebuild regularly; Dependabot will propose base updates.
- **S10:** no authentication. Do not expose the stack beyond localhost.
- **S16:** an LLM can still be steered by plain text. Mitigated: it has no tools, no file or database
  access, its numbers are checked, and the tables are rendered by code.
- The claim check has documented semantic blind spots (see README).
- Dependency versions are not locked, so a future build can differ from what was audited.

## 7. Git history and secret-scan status

All reachable history (main plus local backup refs): 0 secrets found by Gitleaks. The remote has one
branch (`main`), no tags, no releases and no Actions artifacts. Pushed commits use a GitHub no-reply
identity. Two **local-only** refs (`refs/tags/pre-email-rewrite`, `refs/original/...`) still hold the
pre-rewrite commits with the old personal email (S14). They were never pushed; deleting them needs owner
approval.

## 8. Dependency and container-scan status

pip-audit: PASS. Trivy: open findings with no available fix in the Debian and Postgres bases (S9). Version
drift between the image and the audited environment is small and documented (ultralytics 8.4.215 vs
8.4.212, fastapi 0.163.1 vs 0.163.0, python-multipart 0.0.34 vs 0.0.33).

## 9. GitHub Actions status

PASS after S11: read-only token, SHA-pinned actions, no secrets, no privileged triggers, no artifacts,
Dependabot configured. CI is verified on GitHub after each push.

## 10. Publication recommendation

**BLOCKED — SECURITY FIXES REQUIRED** until:

1. **S1:** the exposed Gemini key is revoked and replaced.
2. **S7:** the owner either approves a history rewrite that removes the old model files' local paths
   (and force-pushes to the still-private remote), or explicitly accepts that the OS username and
   project path stay visible in old commits.

Once both are done, with Medium and Low items as documented above, the recommendation becomes
**APPROVED FOR PUBLIC RELEASE**. No configuration is ever risk-free; the remaining risks are listed in
section 6.
