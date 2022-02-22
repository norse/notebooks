{
  description = "spiking neural network simulator";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }: {
    flake-utils.lib.eachDefaultSystem
      (system: 
        let 
          python = nixpkgs.python39Packages;
          pkgs = import nixpkgs { inherit system };
          requirements = ''
          norse
          matplotlib
          jupyter-book
          '';
        in rec {
          devShell = pkgs.mkShell {
            name = "norse";
            nativeBuildInputs = with pkgs; [
              python
            ];
          }
        }
          nixpkgs.mkShell rec {
            name = "impurePythonEnv";
            venvDir = "./.venv";
            buildInputs = [
              # A Python interpreter including the 'venv' module is required to bootstrap
              # the environment.
              pythonPackages.python

              # This execute some shell code to initialize a venv in $venvDir before
              # dropping into the shell
              pythonPackages.venvShellHook

              # Those are dependencies that we would like to use from nixpkgs, which will
              # add them to PYTHONPATH and thus make them accessible from within the venv.
              pythonPackages.matplotlib
              pythonPackages.jupyter
              pythonPackages.pytorch
              pythonPackages.pybind11
              pythonPackages.pytest

              cmake ninja
            ];

            # Run this command, only after creating the virtual environment
            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
              pip install --upgrade pip
              pip install -r requirements.txt
            '';

            # Now we can execute any commands within the virtual environment.
            # This is optional and can be left out to run pip manually.
            postShellHook = ''
              export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/
              # allow pip to install wheels
              unset SOURCE_DATE_EPOCH
            '';
          }
      );
  }
}