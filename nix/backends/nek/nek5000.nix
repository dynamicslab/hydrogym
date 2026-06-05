# nix/backends/nek/nek5000.nix
#
# Builds a case-specific nek5000 binary. NEK5000's build is case-aware:
# the SIZE file determines static array sizes, so each case produces its
# own binary. We isolate that under one derivation per case.
#
# The Python side (mpi4py) and the nek5000 binary MUST be built against
# the same MPI implementation; the backend's default.nix wires both
# against pkgs.mpich.
#
# Inputs:
#   case        - case name string, e.g. "TCFmini_3D_Re180"
#   sizeFile    - path to the case's SIZE file
#   usrFile     - path to the case's <case>.usr file
#   parFile     - path to the case's <case>.par file
#   nek5000Src  - path to the Nek5000 source tree (typically the submodule
#                 at third_party/nek5000)

{ stdenv, lib, gfortran, mpich }:

{ case, sizeFile, usrFile, parFile, nek5000Src }:

stdenv.mkDerivation {
  pname = "nek5000-${case}";
  version = "v19.0";

  src = nek5000Src;

  nativeBuildInputs = [ gfortran ];
  buildInputs = [ mpich ];

  # Stage the case-specific files into a build directory and invoke makenek.
  buildPhase = ''
    runHook preBuild

    mkdir -p build/${case}
    cp -r $src/. build/
    cp ${sizeFile} build/${case}/SIZE
    cp ${usrFile}  build/${case}/${case}.usr
    cp ${parFile}  build/${case}/${case}.par

    pushd build/${case}
    export SOURCE_ROOT="$PWD/../../core"
    # makenek picks up MPI compilers from PATH.
    ${nek5000Src}/bin/makenek ${case} || ./makenek ${case}
    popd

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin $out/share/nek/${case}
    cp build/${case}/nek5000 $out/bin/nek5000
    cp build/${case}/${case}.* $out/share/nek/${case}/
    cp build/${case}/SIZE      $out/share/nek/${case}/

    runHook postInstall
  '';

  meta = with lib; {
    description = "Nek5000 spectral element solver (case: ${case})";
    homepage = "https://nek5000.mcs.anl.gov/";
    license = licenses.bsd3;
    platforms = [ "x86_64-linux" ];
  };
}
