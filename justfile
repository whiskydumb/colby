set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

nightly := "nightly"
locked := ""

# hot-reload build settings. -Cprefer-dynamic is what makes the host and the
# game module share one std and one colby_core: without it each side gets its
# own allocator, its own panic runtime (catch_unwind across the boundary then
# aborts) and its own copy of every static, which breaks the unload canary.
#
# @note: passed as CARGO_ENCODED_RUSTFLAGS, not RUSTFLAGS. The two are not
# interchangeable as far as cargo's fingerprints are concerned: alternating
# between them re-runs build scripts that track the encoded variable, which
# marks the runner dirty and makes cargo try to relink the executable that is
# currently running. The runner rebuilds the game with the encoded form, so this
# has to match it. Several flags are separated by U+001F, not by spaces.
hot_profile := "hot"
hot_flags := "-Cprefer-dynamic"
hot_dir := "target" / hot_profile

# list available recipes
default:
    @just --list

# format all code
fmt:
    rustup run {{nightly}} cargo fmt --all

# verify formatting without writing
fmt-check:
    rustup run {{nightly}} cargo fmt --all --check

# type-check the workspace
check:
    cargo check --workspace --all-targets {{locked}}

# lint the workspace, warnings are errors
clippy:
    cargo clippy --workspace --all-targets {{locked}} -- -D warnings

# lint the shipping configuration: the game linked in, no module loader
clippy-static:
    cargo clippy --package colby --all-targets --no-default-features --features static_game {{locked}} -- -D warnings

# run the test suite
test:
    cargo test --workspace {{locked}}

# spell-check sources and docs
typos:
    typos --color always

# formatting, spelling and lints
lint: fmt-check typos clippy clippy-static

# the full gate
ci: lint check test

# debug build with full symbols
dbg:
    cargo build --profile dbg {{locked}}

# optimized build
release:
    cargo build --release {{locked}}

# shippable build, fat LTO
dist:
    cargo build --profile dist {{locked}}

# compile assets/ into target/assets/
#
# @note: the runner does this for itself, on a timer, so `just hot` and
# `just shot` never need it. It is here for building a tree without starting the
# engine, and for looking at what a source turned into: pass --force to
# recompile everything, --help for the rest.
assets *args:
    cargo run --quiet --package colby_assetc {{locked}} -- {{args}}

# build every crate for hot-reload, with a shared std and a shared colby_core
hot-build:
    $env:CARGO_ENCODED_RUSTFLAGS = "{{hot_flags}}"; cargo build --profile {{hot_profile}} {{locked}}

# run the engine with hot-reload enabled
hot: hot-build
    ./{{hot_dir}}/colby.exe

# render one frame of the game to a png, without opening a window
shot path="colby.png": hot-build
    ./{{hot_dir}}/colby.exe --shot "{{path}}"

# build and open the documentation
doc:
    cargo doc --workspace --no-deps --open

# install the toolchain and tools these recipes need
setup:
    rustup toolchain install {{nightly}} --profile minimal --component rustfmt
    cargo install typos-cli --locked

# remove build artifacts
clean:
    cargo clean
