#!/usr/bin/env bash
#SBATCH --partition=csmpi_fpga_short
#SBATCH --job-name=cfal-accelerate
#SBATCH --time=10:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --mem=64G

set -euo pipefail

# ---------- constants ----------
defworkdir=/vol/itt/data/cfal/accelerate-shared-build-workdir

# ---------- option arguments ----------
opt_set_paths=
opt_workdir=
opt_sourcedir=

# ---------- global variables ----------
sourcedir=
workdir=
projectname=
extra_env_vars=()

# ---------- data ----------

packages=( accelerate accelerate-llvm c2hs cuda language-c llvm-hs )

declare -A package_repo
package_repo[accelerate]=https://github.com/AccelerateHS/accelerate
package_repo[accelerate-llvm]=https://github.com/ivogabe/accelerate-llvm
package_repo[c2hs]=https://github.com/tomsmeding/c2hs
package_repo[cuda]=https://github.com/tomsmeding/cuda
package_repo[language-c]=https://github.com/noahmartinwilliams/language-c
package_repo[llvm-hs]=https://github.com/llvm-hs/llvm-hs

declare -A package_commit
package_commit[accelerate]=master
package_commit[accelerate-llvm]=23ac1c82ff9189ce3f6bc9e475c3ef0afe6bd7de  # ivogabe
package_commit[c2hs]=cfal-patches
package_commit[cuda]=cfal-patches
package_commit[language-c]=98cca579839cd6b6e41baf9d28c1dd53a403f8be
package_commit[llvm-hs]=5bca2c1a2a3aa98ecfb19181e7a5ebbf3e212b76

cfal_machine_path_prefix="/vol/itt/data/cfal/haskell/.ghcup/bin:/vol/itt/data/cfal/llvm/prefix/bin:"


# The layout of the working directory is as follows:
# $workdir  (= $opt_workdir/$UID)
# |- deps
# |  |- accelerate
# |  |- accelerate-llvm
# |  ...
# |- cabal-dir  (cabal build product storage, etc. (traditional ~/.cabal))
# |  |- bin/
# |  |- config
# |  |- store
# |  ...
# '- project
#    |- cabal.project
#    '- dist-newstyle


# ---------- functions ----------

function usage() {
  echo "Usage: $0 [-B] [-w WORKDIR] -s SOURCEDIR"
  echo
  echo "As the SOURCEDIR, pass e.g. the 'MG/accelerate' directory."
  echo
  echo "  -B          Set paths so that the precompiled system dependencies on the"
  echo "              benchmark machine are found (currently, the Haskell toolchain"
  echo "              (GHC 9.4.8) and LLVM 15)."
  echo
  echo "  -w WORKDIR  This tool needs a working directory to put dependencies and"
  echo "              build products in. This defaults to"
  echo "                $defworkdir"
  echo "              if it exists; if it doesn't, '-w' is mandatory. The script"
  echo "              creates a folder in the WORKDIR per user to prevent clashes"
  echo "              on a multi-user system."
  echo
  echo "On the CFAL benchmark machine, you should probably use -B and skip -w, and run"
  echo "this on the cluster (the portal machine doesn't have the correct libraries"
  echo "installed)."
  echo "Example:"
  echo "  $ sbatch $0 -B -s MG/accelerate"
  echo "This works because this script has #SBATCH headers giving parameters to SLURM."
  echo
  echo "This script may take a long time building dependencies, and while doing so it"
  echo "may repeatedly crash with a \"(Directory not empty)\" error. If it does, just"
  echo "restart the job. This is some race condition with parallel jobs on an NFS."
}

function parseargs() {
  # Be helpful, we have mandatory arguments anyway
  if [[ $# -eq 0 ]]; then
    usage "$0" >&2
    exit 1
  fi

  while getopts "hBw:s:" opt; do
    case "$opt" in
      B) opt_set_paths=1 ;;
      w) opt_workdir=$OPTARG ;;
      s) opt_sourcedir=$OPTARG ;;
      h) usage "$0"; exit ;;
      *) usage "$0" >&2; exit 1 ;;
    esac
  done

  shift $((OPTIND - 1))
  if [[ $# -gt 0 ]]; then
    echo >&2 "Unexpected arguments"
    exit 1
  fi
  if [[ -z $opt_sourcedir ]]; then
    echo >&2 "'-s' is mandatory"
    exit 1
  fi

  if [[ -z $opt_workdir ]]; then
    if [[ -d $defworkdir ]]; then
      opt_workdir=$defworkdir
    else
      echo >&2 "Directory $defworkdir not available, '-w' is mandatory"
      exit 1
    fi
  fi
}

# Takes directory of Accelerate project as argument.
function detect_project_name() {
  local files
  files=$(ls "$1"/*.cabal)
  if [[ $(wc -l <<<"$files") -ne 1 ]]; then
    echo >&2 "Multiple *.cabal files found in directory '$1'?"
    exit 1
  fi
  basename "$files" | sed 's/\.[^.]*//'
}

function set_extra_env_vars() {
  extra_env_vars=( CABAL_DIR="$workdir/cabal-dir" )

  if [[ $opt_set_paths = 1 ]]; then
    extra_env_vars+=(
      PATH="$cfal_machine_path_prefix$PATH"
      # this is necessary for TemplateHaskell splices to find the libraries;
      # the fact that such splices don't use the path information available
      # elsewhere (as auto-detected by Accelerate config scripts) is an
      # annoying current Haskell quirk
      LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64
    )
  fi
}

function init_dep_repos() {
  mkdir -p "$workdir/deps"

  local repourl
  local gitref

  local curhead

  for name in "${packages[@]}"; do
    repourl=${package_repo[$name]}
    gitref=${package_commit[$name]}

    if [[ ! -d "$workdir/deps/$name/.git" ]]; then
      # Initial clone
      (
        set -x
        git -C "$workdir/deps" clone --no-checkout "$repourl" "$name"
        git -C "$workdir/deps/$name" checkout "$gitref"
      )
    else
      # First ensure the right ref is checked out
      curhead=$(sed 's,^ref: refs/heads/,,' <"$workdir/deps/$name/.git/HEAD")
      if [[ $curhead != "$gitref" ]]; then
        # It isn't; before we checkout, make sure we have the right remote
        (
          set -x
          git -C "$workdir/deps/$name" remote set-url origin "$repourl"
          git -C "$workdir/deps/$name" fetch
          git -C "$workdir/deps/$name" checkout "$gitref"
        )
      fi

      # If the given ref is a branch name (i.e. not a commit hash) -- this is a
      # very hacky proxy for a proper check
      if [[ ${#gitref} -ne 40 ]]; then
        # Make sure local branch corresponds to remote branch
        if [[ "$(git -C "$workdir/deps/$name" rev-parse "$gitref")" != "$(git -C "$workdir/deps/$name" rev-parse "origin/$gitref")" ]]; then
          (
            set -x
            git -C "$workdir/deps/$name" reset "origin/$gitref" --hard
          )
        fi
      fi
    fi
  done
}

# We just overwrite the same project file with a different package list each
# time. This is a bit weird, but cabal handles it fine and it allows us to
# reuse the built dependencies.
function write_project_file() {
  mkdir -p "$workdir/project"
  cat >"$workdir/project/cabal.project" <<EOF
packages:
  $sourcedir
  $workdir/deps/accelerate
  $workdir/deps/accelerate-llvm/accelerate-llvm
  $workdir/deps/accelerate-llvm/accelerate-llvm-native
  $workdir/deps/accelerate-llvm/accelerate-llvm-ptx
  $workdir/deps/llvm-hs/llvm-hs-pure
  $workdir/deps/llvm-hs/llvm-hs
  $workdir/deps/cuda
  $workdir/deps/c2hs
  $workdir/deps/language-c

with-compiler: ghc-9.4.8

allow-newer:
  lens-accelerate:lens

-- These are still necessary for the linker to find Cuda
package $projectname
  extra-lib-dirs: /usr/local/cuda/lib64 /usr/local/cuda/nvvm/lib64
EOF
}

# The arguments are passed on to cabal.
function run_cabal_command() {
  local env_args
  env_args=()
  if [[ $opt_set_paths = 1 ]]; then
    env_args+=(
      PATH="$cfal_machine_path_prefix$PATH"
      # this is necessary for TemplateHaskell splices to find the libraries;
      # the fact that such splices don't use the path information available
      # elsewhere (as auto-detected by Accelerate config scripts) is an
      # annoying current Haskell quirk
      LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64
    )
  fi

  (
    set -x
    cd "$workdir/project"
    env CABAL_DIR="$workdir/cabal-dir" "${env_args[@]}" cabal "$@"
  )
}

function main() {
  parseargs "$@"

  # Resolve the source directory
  sourcedir="$(realpath "$opt_sourcedir")"

  # Add username to working directory to facilitate multi-user systems
  workdir=$(realpath "$opt_workdir/$(id -un)")

  projectname=$(detect_project_name "$sourcedir")

  set_extra_env_vars

  echo "# Setting up benchmark: $projectname"
  init_dep_repos

  write_project_file

  # Check if 'cabal update' has already happened for this user
  if [[ ! -f "$workdir/cabal-dir/packages/hackage.haskell.org/01-index.cache" ]]; then
    echo
    echo "# Setting up Cabal (getting package list from Hackage)"
    (
      set -x
      env "${extra_env_vars[@]}" cabal update
    )
  fi

  echo
  echo "# Building benchmark: $projectname"
  (
    set -x
    cd "$workdir/project"
    env "${extra_env_vars[@]}" cabal build "$projectname"
  )

  echo
  echo "# Running benchmark: $projectname"
  local exepath
  exepath=$( (
    set -x
    cd "$workdir/project"
    env "${extra_env_vars[@]}" cabal list-bin "$projectname"
  ) )
  exepath=$(realpath "$exepath")

  echo "Executable = $exepath"

  (
    set -x
    cd "$sourcedir"
    env "${extra_env_vars[@]}" "$exepath"
  )
}

main "$@"
