{
  description = "Cuda development environment";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-24.11";
  };

  nixConfig = {
    extra-substituters = ["https://cuda-maintainers.cachix.org"];

    extra-trusted-public-keys = ["cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="];
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      config.cudaSupport = true;
    };
  in {
    devShells."${system}".default = pkgs.mkShell {
      packages = with pkgs; [
        cudaPackages.cudatoolkit
        cudaPackages.cudnn
        cudaPackages.cuda_cudart
        linuxPackages.nvidia_x11
        libGLU
        libGL
        cmake
        gcc12
        gdb
        mpi
      ];

      shellHook = ''
        export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
        export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
        export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
        export EXTRA_CCFLAGS="-I/usr/include"
      '';
    };
  };
}