{
  description = "ZINC — a fast LLM inference engine for Vulkan (RDNA3/RDNA4) and Metal (Apple Silicon)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # zig 0.15.x is currently in nixpkgs as zigpkgs or via the master
        # overlay; fall back to pkgs.zig if pkgs.zig_0_16 is not yet present.
        zigPkg =
          if pkgs ? zig_0_16 then pkgs.zig_0_16
          else if pkgs ? zig then pkgs.zig
          else throw "No suitable Zig package found in nixpkgs. Use an up-to-date nixpkgs-unstable channel.";

        zinc = pkgs.stdenv.mkDerivation {
          pname = "zinc";
          version = "0.1.0";

          src = ./.;

          nativeBuildInputs = [
            zigPkg
            pkgs.shaderc   # provides glslc for SPIR-V shader compilation
          ];

          buildInputs = pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.vulkan-loader
            pkgs.vulkan-headers
          ];

          # Zig uses $HOME for its local package cache during builds.
          # Point it at a writable temp directory inside the sandbox.
          preBuild = ''
            export HOME=$TMPDIR/zig-home
            mkdir -p $HOME
          '';

          buildPhase = ''
            runHook preBuild
            zig build -Doptimize=ReleaseFast --prefix $out
            runHook postBuild
          '';

          # zig build --prefix already places everything under $out,
          # so there is nothing more to install.
          dontInstall = true;

          meta = with pkgs.lib; {
            description = "ZINC LLM inference engine (Vulkan / Metal)";
            longDescription = ''
              ZINC is a high-performance LLM inference engine targeting AMD
              RDNA3/RDNA4 GPUs via Vulkan and Apple Silicon via Metal.  It
              speaks the OpenAI-compatible HTTP API and supports GGUF models
              (Qwen3, Gemma 4 MoE, and more).
            '';
            homepage = "https://github.com/zolotukhin/zinc";
            license = licenses.mit;
            maintainers = [];
            platforms = platforms.linux ++ platforms.darwin;
          };
        };
      in
      {
        packages = {
          default = zinc;
          inherit zinc;
        };

        # `nix develop` shell — matches the project's manual build instructions.
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            zigPkg
            pkgs.shaderc
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.vulkan-loader
            pkgs.vulkan-headers
            pkgs.vulkan-validation-layers
          ];

          shellHook = ''
            echo "zinc dev shell — zig $(zig version)"
          '';
        };
      }
    );
}
