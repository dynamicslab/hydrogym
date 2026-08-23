# Contributing to Nek5000 

Nek5000 on Github follows a forking workflow.
If you are unfamiliar with forking, take a look at [this great guide](https://guides.github.com/activities/forking/).
For a more detailed description of forking workflow, here's a [slightly longer read](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow).

Please branch off of and open pull requests to the `develop` branch.
The `master` branch is reserved for releases.

## Where did the sources go?

As part of the move to git, the contents of the svn repo have been reorganized to match the modularlity of the codebase.  Here's a brief description of each top-level directory:

### `core`
`core` contains the majority of the Nek5000 application sources.

### `jl`
`jl` contains gather/scatter communication, interpolation, and preconditioners written in highly general C code.
`jl` used to live in `nek5_svn/trunk/nek`, but is being promoted to the top level to emphasize its library-like relationship to the rest of the source.
In fact, `jl` has been extended externally in [gslib](https://github.com/gslib/gslib), which is used in other projects.

### `bin`
`bin` contains bash scripts for running nek5000 and manipulating its output on a variety of platforms, e.g. `nekmpi`.

### `tools`
`tools` contains the sources for the pre- and post-processing tools, e.g. `genmap`, which are stand-alone fortran programs.

### `tests` 
`tests` contains a light-weight subset of the validation tests found in nek5000-examples and, in the future, unit-like tests to validate specific features.
Validation tests that require binary inputs, e.g. `re2` or `map` files, belong in the separate nek5000-examples repo.

### `lib`
`lib` contains original nek5000 sources that are general enough to be used by 3rd party plugin/toolbox developers or are shared with the tools.
`lib` is starting nearly empty, with sources slowly migrating into them from `core`.

### `3rd_party`
`3rd_party` contains nothing, and should remain empty save git-related meta-data files.
Its purpose it to provide a consistent place for 3rd part plugin/toolbox developers to place their code.

## Porting to the new layout

If you had previously forked the Nek5000/Nek5000 repository and made changes, you need to match the new layout before pulling.
Fortunately, this is easy!

```
cd <PATH_TO_NEK5000>
git filter-branch --force --index-filter \
  'git ls-files -s | sed "s-\t-&core/-" |
  GIT_INDEX_FILE=$GIT_INDEX_FILE.new \
  git update-index --index-info &&
  mv $GIT_INDEX_FILE.new $GIT_INDEX_FILE' HEAD

git filter-branch --force --tree-filter \
  'test -d core/jl && mv core/jl . || echo "Nothing"' HEAD
```

Now you should be able to pull in the rest of the new layout via pull request or the command line.
