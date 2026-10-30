"""
RemoteDockerEnvironment — thin wrapper around DockerEnvironment for remote
Docker daemons accessed via DOCKER_HOST (tcp:// or ssh://).

Differs from local DockerEnvironment:

1. ``is_mounted = False`` — bind-mount log volumes land on the *remote* host,
   not the training host. The trial runner uses ``docker compose cp`` to pull
   logs back (already implemented in DockerEnvironment).

2. ``_DOCKER_COMPOSE_BASE_PATH`` points to a compose fragment *without*
   host-volume mounts, avoiding orphan directories on the remote host.

3. ``start()`` creates /logs/{agent,verifier,artifacts} inside the container
   (remote compose base has no bind mounts to create them implicitly).

4. **Agent runtime image mount** — Docker has no ImageVolume (K8s 1.31+).
   When ``agent_runtime_image`` is passed (via environment.kwargs in the
   agent loop config), the runtime image is extracted once to a staging
   directory on the remote Docker host, then bind-mounted read-only into
   every task container via compose volume override.

Usage (agent_loop_config yaml):
    environment:
      import_path: harbor_patch.environments.remote_docker:RemoteDockerEnvironment
      kwargs:
        agent_runtime_image: docker.io/jierun/c-cc-2.1.118:v0.1
        agent_runtime_mount_path: /opt/custom-agent-runtime/claude-code
        agent_runtime_image_subpath: opt/custom-agent-runtime/claude-code
"""

import asyncio
import json
import logging
import os
import shlex
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.trial.paths import EnvironmentPaths

_DIR = Path(__file__).parent
_COMPOSE_BASE_REMOTE = _DIR / "docker-compose-base-remote.yaml"
_COMPOSE_BUILD_PROXY = _DIR / "docker-compose-build-proxy.yaml"

logger = logging.getLogger(__name__)


def _force_real_docker_cli() -> None:
    """Put the REAL docker CLI ahead of the proot-dind shim on PATH.

    On proot-dind dev pods, `docker` resolves to a `docker-proot` wrapper that
    passes pull/load/info/images through to the real client but intercepts
    `docker compose build/up` with a LOCAL proot build. That proot build is a
    half-implementation: it ignores COPY and its RUN steps silently fail, so the
    task image comes out empty/broken and every trial dies (KeyError / empty
    trajectory / position_ids crash downstream).

    RemoteDockerEnvironment always targets a real remote daemon via DOCKER_HOST,
    where a normal `docker build` handles COPY/RUN/caching correctly. So every
    docker call from this backend must use the real binary. Prepend a dir that
    symlinks it; harbor's compose subprocesses inherit this PATH.
    """
    real = "/usr/bin/docker"
    if not os.path.exists(real):
        return
    bindir = "/tmp/harbor-real-docker-bin"
    try:
        os.makedirs(bindir, exist_ok=True)
        link = os.path.join(bindir, "docker")
        if not os.path.lexists(link):
            os.symlink(real, link)
        parts = os.environ.get("PATH", "").split(":")
        if bindir not in parts:
            os.environ["PATH"] = bindir + ":" + os.environ.get("PATH", "")
        logger.info("RemoteDocker: real docker CLI (%s) fronted ahead of proot shim", real)
    except OSError as e:
        logger.warning("RemoteDocker: could not front real docker CLI: %s", e)


_force_real_docker_cli()

_RUNTIME_STAGE_BASE = "/opt/harbor-agent-runtime-cache"

_runtime_staged: dict[str, str] = {}
_runtime_stage_lock = asyncio.Lock()


class RemoteDockerEnvironment(DockerEnvironment):
    _DOCKER_COMPOSE_BASE_PATH = _COMPOSE_BASE_REMOTE

    def __init__(
        self,
        *args,
        agent_runtime_image: str | None = None,
        agent_runtime_mount_path: str = "/opt/custom-agent-runtime/claude-code",
        agent_runtime_image_subpath: str | None = None,
        agent_runtime_stage_helper_image: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._agent_runtime_image = (agent_runtime_image or "").strip() or None
        self._agent_runtime_mount_path = agent_runtime_mount_path
        self._agent_runtime_image_subpath = (
            (agent_runtime_image_subpath or "").strip()
            or agent_runtime_mount_path.lstrip("/")
        )
        self._agent_runtime_stage_helper_image = (
            (agent_runtime_stage_helper_image or "").strip() or None
        )

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def _docker_compose_paths(self) -> list[Path]:
        """Append an opt-in BuildKit proxy override for task-image builds.

        Docker daemon proxy settings cover image pulls, but they are not
        automatically present in Dockerfile RUN steps. The compose override
        passes Docker's pre-defined proxy build args so apt/pip/curl can reach
        the host-side Clash listener without exposing the proxy at task runtime.
        """
        paths = super()._docker_compose_paths
        if (
            os.environ.get("HARBOR_DOCKER_BUILD_PROXY", "").strip()
            and _COMPOSE_BUILD_PROXY not in paths
        ):
            paths.append(_COMPOSE_BUILD_PROXY)
        return paths

    @staticmethod
    async def _run_cmd(cmd: str) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        return proc.returncode, (out or b"").decode(errors="replace").strip()

    async def _stage_runtime_without_shell(
        self,
        image: str,
        subpath: str,
        safe_tag: str,
        uniq: str,
    ) -> tuple[int, str]:
        """Stage a scratch-based runtime through a disposable helper image.

        Docker cannot bind-mount an image filesystem. The Codex runtime is
        FROM scratch, so it has no shell that the normal staging command can
        execute. Stream only the runtime subtree with ``docker cp`` into a
        helper container, then atomically install the marked cache directory.
        """
        helper = self._agent_runtime_stage_helper_image
        if not helper:
            return 1, "no runtime staging helper image configured"

        cache_arg = shlex.quote(f"{_RUNTIME_STAGE_BASE}:/base")
        helper_q = shlex.quote(helper)
        image_q = shlex.quote(image)
        src_path_q = shlex.quote(f"/{subpath.strip('/')}/.")
        dest_q = shlex.quote(f"/base/{safe_tag}")
        tmp_q = shlex.quote(f"/base/{uniq}")

        rc, out = await self._run_cmd(
            f"docker run --rm -v {cache_arg} {helper_q} "
            f"sh -c 'test -f /base/{safe_tag}/.staged_ok'"
        )
        if rc == 0:
            return 0, "runtime already staged"

        rc, out = await self._run_cmd(f"docker pull -q {helper_q}")
        if rc != 0:
            return rc, out

        script = (
            "set -eu; "
            f"tmp={tmp_q}; dest={dest_q}; "
            'rm -rf "$tmp"; mkdir -p "$tmp"; '
            'tar -xf - -C "$tmp"; '
            'touch "$tmp/.staged_ok"; '
            'if ! mv -T "$tmp" "$dest" 2>/dev/null; then '
            'test -f "$dest/.staged_ok"; '
            "fi; "
            'rm -rf "$tmp"'
        )
        script_q = shlex.quote(script)
        marker_q = shlex.quote(f"{_RUNTIME_STAGE_BASE}/{safe_tag}/.staged_ok")

        # The source image only needs to be created, not started. `docker cp`
        # emits a tar stream which is extracted by the helper container.
        cmd = (
            "set -eu; "
            f"cid=$(docker create {image_q} /codex-placeholder-never-run); "
            "trap 'docker rm -f \"$cid\" >/dev/null 2>&1 || true' 0 1 2 3 15; "
            f'docker cp "$cid":{src_path_q} - | '
            f"docker run --rm -i -v {cache_arg} {helper_q} sh -c {script_q}; "
            f"test -f {marker_q}"
        )
        return await self._run_cmd(cmd)

    async def _ensure_runtime_staged(self) -> str | None:
        """Extract the runtime image to a directory on the remote Docker host
        (once per image, cached across trials). Returns the remote path."""
        if not self._agent_runtime_image:
            return None

        image = self._agent_runtime_image
        subpath = self._agent_runtime_image_subpath

        async with _runtime_stage_lock:
            cached = _runtime_staged.get(image)
            if cached:
                logger.info("Agent runtime already staged: %s", cached)
                return cached

            safe_tag = image.replace("/", "_").replace(":", "_")
            host_dir = f"{_RUNTIME_STAGE_BASE}/{safe_tag}"

            rc, out = await self._run_cmd(f"docker pull -q {image}")
            if rc != 0:
                logger.warning("runtime image pull failed (%d): %s", rc, out)
                return None

            uniq = f"{safe_tag}.tmp.{os.getpid()}"
            if self._agent_runtime_stage_helper_image:
                rc, out = await self._stage_runtime_without_shell(
                    image=image,
                    subpath=subpath,
                    safe_tag=safe_tag,
                    uniq=uniq,
                )
                if rc != 0:
                    logger.warning(
                        "runtime stage via helper failed (%d): %s", rc, out
                    )
                    return None
                _runtime_staged[image] = host_dir
                logger.info(
                    "Agent runtime staged without source-image shell at remote %s",
                    host_dir,
                )
                return host_dir

            # Already staged on the remote host (possibly by another worker process)?
            # The marker + all path ops go through the remote daemon (host_dir lives on
            # the remote host, not here), using the runtime image itself as the toolbox.
            rc, _ = await self._run_cmd(
                f"docker run --rm -v {_RUNTIME_STAGE_BASE}:/base {image} "
                f"sh -c 'test -f /base/{safe_tag}/.staged_ok'"
            )
            if rc == 0:
                _runtime_staged[image] = host_dir
                logger.info("Agent runtime already staged on remote: %s", host_dir)
                return host_dir

            # Stage via tar, NOT `cp -a`: cp fails to recreate symlinks on some remote
            # filesystems (243's nydus snapshotter -> "cannot create symbolic link
            # libpython3.12.so"); tar preserves them. Each worker stages into a private,
            # pid-scoped tmp dir and atomically mv's it into place, so concurrent stagers
            # never collide (the old fixed-name `docker create` clashed under concurrency).
            rc, out = await self._run_cmd(
                f"docker run --rm -v {_RUNTIME_STAGE_BASE}:/base {image} sh -c '"
                f"rm -rf /base/{uniq}; mkdir -p /base/{uniq} && "
                f"cd /{subpath} && tar cf - . | tar xf - -C /base/{uniq} && "
                f"touch /base/{uniq}/.staged_ok && "
                f"rm -rf /base/{safe_tag} && mv /base/{uniq} /base/{safe_tag}'"
            )
            if rc != 0:
                logger.warning("runtime stage via tar failed (%d): %s", rc, out)
                await self._run_cmd(
                    f"docker run --rm -v {_RUNTIME_STAGE_BASE}:/base {image} "
                    f"sh -c 'rm -rf /base/{uniq}'"
                )
                return None

            _runtime_staged[image] = host_dir
            logger.info("Agent runtime staged at remote %s (via tar)", host_dir)
            return host_dir

    async def start(self, force_build: bool) -> None:
        host_dir = await self._ensure_runtime_staged()
        if host_dir:
            mount_spec = f"{host_dir}:{self._agent_runtime_mount_path}:ro"
            if self._mounts_json is None:
                self._mounts_json = []
            self._mounts_json.append(mount_spec)
            logger.info("Added runtime volume: %s", mount_spec)

        await super().start(force_build)
        await self.exec(
            f"mkdir -p {EnvironmentPaths.agent_dir} "
            f"{EnvironmentPaths.verifier_dir} "
            f"{EnvironmentPaths.artifacts_dir}"
        )

    def _write_mounts_compose_file(self) -> Path:
        """Override the base writer: alongside the bind-mounted runtime volume, inject
        CUSTOM_AGENT_RUNTIME_ROOT / CUSTOM_AGENT_PYTHON into the task container's env so
        the mounted-runtime agent (image_mounted_openhands_ai etc.) can locate the runtime.
        Docker has no ImageVolume, so — unlike KubernetesEnvironment, which injects these
        via V1EnvVar — the base class never sets them; without them the agent can't find
        its runtime and every trial fails (KeyError before any real trajectory)."""
        service: dict[str, object] = {"volumes": list(self._mounts_json or [])}
        if self._agent_runtime_image:
            mount = self._agent_runtime_mount_path.rstrip("/")
            service["environment"] = {
                "CUSTOM_AGENT_RUNTIME_ROOT": mount,
                "CUSTOM_AGENT_PYTHON": f"{mount}/bin/python",
            }
        compose: dict[str, object] = {"services": {"main": service}}
        path = self.trial_paths.trial_dir / "docker-compose-mounts.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(compose, indent=2))
        return path
