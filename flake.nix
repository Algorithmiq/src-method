{
  description = "src_method: Successive Randomized Compression";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSystem = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forEachSystem (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.uv
            pkgs.git
            pkgs.gh
          ];

          env = {
            # Python itself and all project dependencies stay under uv's control,
            # so the flake only has to provide uv and the native libraries that
            # binary wheels (numpy, quimb, cupy, ...) dlopen at runtime.
            UV_PYTHON_DOWNLOADS = "automatic";
          }
          // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ];
          };

          # GPU extras are deliberately not synced: they are CUDA/ROCm specific.
          shellHook = ''
            uv sync --all-groups
            source .venv/bin/activate
          '';
        };
      });

      formatter = forEachSystem (pkgs: pkgs.nixfmt-rfc-style);
    };
}
