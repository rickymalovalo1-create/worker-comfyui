---
"worker-comfyui": minor
---

Bump the pinned ComfyUI version from 0.29.0 to 0.34.0 (current upstream latest). Workflows exported from recent ComfyUI versions use core nodes that don't exist in 0.29-era images and fail at runtime with "missing node" errors — notably every image consumed as a base by comfyui-wizard. ComfyUI's dependency floors are unchanged between 0.29 and 0.34, so the existing torch 2.11.0+cu128 and transformers<5 pins still apply, and the build-time smoke test gates startup.

Also fixes the release workflow's Docker Hub description sync: the previous action authenticated via Docker Hub's legacy user-login API, which rejects organization access tokens unconditionally — it 401'd on every release since the org token rotation and aborted the job before the GitHub release was created (5.9.0 had to be backfilled by hand). The step is now a direct PATCH to the modern namespaces endpoint using the same token, and runs after the GitHub release step so a description failure can never block a release again.
