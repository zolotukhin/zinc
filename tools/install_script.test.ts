import { expect, test } from "bun:test";
import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, mkdirSync, readFileSync, readlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const repoRoot = join(import.meta.dir, "..");

function currentTarget(): "linux-x86_64" | "macos-aarch64" | null {
  if (process.platform === "linux" && process.arch === "x64") return "linux-x86_64";
  if (process.platform === "darwin" && process.arch === "arm64") return "macos-aarch64";
  return null;
}

function addShader(treeDir: string, target: string) {
  if (target === "linux-x86_64") {
    const shaderDir = join(treeDir, "share", "zinc", "shaders");
    mkdirSync(shaderDir, { recursive: true });
    writeFileSync(join(shaderDir, "test.spv"), "spirv\n");
    return;
  }
  const shaderDir = join(treeDir, "share", "zinc", "shaders", "metal");
  mkdirSync(shaderDir, { recursive: true });
  writeFileSync(join(shaderDir, "test.metal"), "// metal\n");
}

function makeRelease(opts: { version: string; includeChecksumEntry?: boolean; includeShaders?: boolean }) {
  const target = currentTarget();
  if (!target) return null;

  const root = mkdtempSync(join(tmpdir(), "zinc-install-test-"));
  const tag = `v${opts.version}`;
  const tagDir = join(root, "release", tag);
  const treeName = `zinc-${tag}-${target}`;
  const treeDir = join(tagDir, treeName);
  mkdirSync(join(treeDir, "bin"), { recursive: true });
  writeFileSync(join(treeDir, "bin", "zinc"), "#!/usr/bin/env bash\nprintf 'zinc test\\n'\n", { mode: 0o755 });
  writeFileSync(join(treeDir, "README.md"), "readme\n");
  writeFileSync(join(treeDir, "LICENSE"), "license\n");
  writeFileSync(join(treeDir, "VERSION.json"), "{}\n");
  if (opts.includeShaders !== false) addShader(treeDir, target);

  const asset = `${treeName}.tar.gz`;
  const archivePath = join(tagDir, asset);
  execFileSync("tar", ["-C", tagDir, "-czf", archivePath, treeName]);
  const digest = createHash("sha256").update(readFileSync(archivePath)).digest("hex");
  const checksumAsset = opts.includeChecksumEntry === false ? "other.tar.gz" : asset;
  writeFileSync(join(tagDir, "SHA256SUMS.txt"), `${digest}  ${checksumAsset}\n`);

  return {
    root,
    tag,
    baseUrl: `file://${join(root, "release")}`,
    installDir: join(root, "install"),
    binDir: join(root, "bin"),
  };
}

function runInstaller(release: NonNullable<ReturnType<typeof makeRelease>>) {
  return spawnSync("bash", ["scripts/install.sh"], {
    cwd: repoRoot,
    env: {
      ...process.env,
      ZINC_VERSION: release.tag,
      ZINC_BASE_URL: release.baseUrl,
      ZINC_INSTALL_DIR: release.installDir,
      ZINC_BIN_DIR: release.binDir,
    },
    encoding: "utf8",
  });
}

test("install.sh installs a local release archive with runtime shaders", () => {
  const release = makeRelease({ version: "9.9.9" });
  if (!release) return;

  const result = runInstaller(release);

  expect(result.status).toBe(0);
  expect(result.stdout).toContain("checksum verified");
  expect(readlinkSync(join(release.binDir, "zinc"))).toContain("current/bin/zinc");
});

test("install.sh reports a missing checksum entry instead of exiting silently", () => {
  const release = makeRelease({ version: "9.9.8", includeChecksumEntry: false });
  if (!release) return;

  const result = runInstaller(release);

  expect(result.status).toBe(1);
  expect(result.stderr).toContain("no checksum entry");
});

test("install.sh rejects archives missing target runtime shaders", () => {
  const release = makeRelease({ version: "9.9.7", includeShaders: false });
  if (!release) return;

  const result = runInstaller(release);

  expect(result.status).toBe(1);
  expect(result.stderr).toContain("unexpected archive layout");
  expect(result.stderr).toContain("shaders");
});
