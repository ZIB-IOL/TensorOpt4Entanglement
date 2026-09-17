#!/usr/bin/env python3
"""Clear the executable-stack flag on a copy of the MOSEK shared libraries.

MOSEK ships libmosek64.so marked PT_GNU_STACK = RWE. Recent glibc and kernels
refuse to make the stack executable when a library is dlopen'd, so loading it
fails with

    cannot enable executable stack as shared object requires: Invalid argument

even though the file is present and ldd resolves every dependency. The marking
is a build artifact -- an object assembled without a .note.GNU-stack section
makes the linker mark the whole library conservatively -- not a real need for
an executable stack, so clearing the bit is the standard remedy (it is what
`execstack -c` and `patchelf --clear-execstack` do).

A site install lives on a read-only filesystem, so this copies the libraries
somewhere writable and patches the copies; point MOSEKBINDIR at that copy and
rebuild Mosek.jl. Written against the ELF spec directly so it needs neither
patchelf nor execstack, which clusters rarely have.

    python3 scripts/fix_execstack.py --find           # what MOSEK is here, which fits
    python3 scripts/fix_execstack.py --check          # report, change nothing
    python3 scripts/fix_execstack.py                  # patch a local copy

Letting Mosek.jl download its own MOSEK does not avoid this. It is pinned to
its own major.minor (10.2 here, from Manifest.toml) and the newest 10.2 patch,
10.2.19, carries the same marking; MOSEK cleared it in 11.x. So the version to
patch is found rather than hardcoded: bump Mosek.jl and this follows.
"""
import argparse, glob, os, re, shutil, struct, subprocess, sys

PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def _hdr(f):
    """Return (endian, is64, e_phoff, e_phentsize, e_phnum) or None if not ELF."""
    f.seek(0)
    ident = f.read(16)
    if len(ident) < 16 or ident[:4] != b"\x7fELF":
        return None
    is64 = ident[4] == 2
    endian = "<" if ident[5] == 1 else ">"
    # e_phoff is at 0x20 (ELF64) / 0x1c (ELF32); e_phentsize/e_phnum follow
    if is64:
        f.seek(0x20); phoff = struct.unpack(endian + "Q", f.read(8))[0]
        f.seek(0x36); phentsize, phnum = struct.unpack(endian + "HH", f.read(4))
    else:
        f.seek(0x1C); phoff = struct.unpack(endian + "I", f.read(4))[0]
        f.seek(0x2A); phentsize, phnum = struct.unpack(endian + "HH", f.read(4))
    return endian, is64, phoff, phentsize, phnum


def execstack_offset(f):
    """Byte offset of the PT_GNU_STACK p_flags field, and its value."""
    h = _hdr(f)
    if h is None:
        return None
    endian, is64, phoff, phentsize, phnum = h
    # ELF64 lays out p_type, p_flags, ...; ELF32 puts p_flags second to last
    flags_at = 4 if is64 else 24
    for i in range(phnum):
        base = phoff + i * phentsize
        f.seek(base)
        p_type = struct.unpack(endian + "I", f.read(4))[0]
        if p_type == PT_GNU_STACK:
            f.seek(base + flags_at)
            return base + flags_at, struct.unpack(endian + "I", f.read(4))[0], endian
    return None


def is_execstack(path):
    try:
        with open(path, "rb") as f:
            got = execstack_offset(f)
    except OSError:
        return None
    return None if got is None else bool(got[1] & PF_X)


def clear_execstack(path):
    """Clear PF_X in PT_GNU_STACK. Returns True if the file was changed."""
    with open(path, "r+b") as f:
        got = execstack_offset(f)
        if got is None:
            return False
        off, flags, endian = got
        if not flags & PF_X:
            return False
        f.seek(off)
        f.write(struct.pack(endian + "I", flags & ~PF_X))
    return True


def required_version(manifest="Manifest.toml"):
    """MOSEK major.minor that Mosek.jl demands, read from the Manifest.

    Mosek.jl's build derives it from its own package version and will accept
    nothing else, so this is what a bin directory has to match -- hardcoding
    a version here would go stale the moment the Manifest is bumped.
    """
    try:
        with open(manifest) as f:
            block = False
            for line in f:
                line = line.strip()
                if line.startswith("[["):
                    block = line == "[[deps.Mosek]]"
                elif block and line.startswith("version"):
                    v = line.split("=", 1)[1].strip().strip('"').split(".")
                    return f"{v[0]}.{v[1]}"
    except OSError:
        pass
    return None


def bindir_version(d):
    """MOSEK version reported by <d>/mosek, the way Mosek.jl's build reads it."""
    exe = os.path.join(d, "mosek")
    if not os.access(exe, os.X_OK):
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"MOSEK Version (\d+\.\d+\.\d+)", out)
    return m.group(1) if m else None


def candidates():
    """Places a MOSEK bin directory plausibly lives, site installs first."""
    pats = ["/software/mosek/*/tools/platform/*/bin",
            "/opt/mosek/*/tools/platform/*/bin",
            os.path.expanduser("~/mosek/*/tools/platform/*/bin"),
            ".julia_depot/packages/Mosek/*/deps/src/mosek/*/tools/platform/*/bin",
            os.path.expanduser("~/.julia/packages/Mosek/*/deps/src/mosek/*/tools/platform/*/bin")]
    seen, out = set(), []
    for p in pats:
        for d in sorted(glob.glob(p)):
            r = os.path.realpath(d)
            if r not in seen:
                seen.add(r)
                out.append(d)
    return out


def find_bindir(want):
    """First candidate whose `mosek` reports major.minor == want."""
    for d in candidates():
        v = bindir_version(d)
        if v and ".".join(v.split(".")[:2]) == want:
            return d, v
    return None, None


def is_elf(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def entries(d):
    return sorted(os.listdir(d))


def human(n):
    for u in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or u == "GiB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None,
                    help="MOSEK bin directory (default: $MOSEKBINDIR, else searched for)")
    ap.add_argument("--find", action="store_true",
                    help="list every MOSEK found and which one matches the Manifest")
    ap.add_argument("--dst", default=".mosek_bin",
                    help="where to put the patched copy (default: ./.mosek_bin)")
    ap.add_argument("--check", action="store_true",
                    help="report what is marked, change nothing")
    a = ap.parse_args()

    want = required_version()

    if a.find:
        found = candidates()
        if not found:
            print("no MOSEK installation found in the usual places")
            return 1
        print(f"Mosek.jl (Manifest.toml) requires MOSEK {want or '?'}\n")
        for d in found:
            v = bindir_version(d)
            if v is None:
                mark, note_ = "  ?", "no runnable 'mosek' -- Mosek.jl's build would reject it"
            elif want and ".".join(v.split(".")[:2]) == want:
                x = is_execstack(os.path.join(d, f"libmosek64.so.{want}"))
                mark = "  *"
                note_ = f"{v}  MATCHES" + ("  (executable stack -- needs patching)" if x else "")
            else:
                mark, note_ = "   ", f"{v}  (Mosek.jl {want} will not accept this)"
            print(f"{mark} {d}\n      {note_}")
        print("\n  * = usable as MOSEKBINDIR")
        return 0

    a.src = a.src or os.environ.get("MOSEKBINDIR")
    if not a.src:
        a.src, v = find_bindir(want) if want else (None, None)
        if not a.src:
            sys.exit(f"no MOSEK {want or ''} found: pass --src, set MOSEKBINDIR, "
                     "or run with --find to see what is here")
        print(f"using {a.src} (MOSEK {v}, matching Mosek.jl {want})\n")
    if not os.path.isdir(a.src):
        sys.exit(f"not a directory: {a.src}")
    got = bindir_version(a.src)
    if want and got and ".".join(got.split(".")[:2]) != want:
        sys.exit(f"{a.src} is MOSEK {got}, but Mosek.jl needs {want}; "
                 "its build accepts no other version. Run with --find.")

    names = entries(a.src)
    if not names:
        sys.exit(f"nothing in {a.src}")

    if a.check:
        bad = elves = 0
        for n in names:
            p = os.path.join(a.src, n)
            if os.path.islink(p):
                print(f"  link  {n} -> {os.readlink(p)}")
                continue
            if not is_elf(p):
                continue
            elves += 1
            m = is_execstack(p)
            label = {True: "EXECSTACK", False: "ok", None: "no PT_GNU_STACK"}[m]
            print(f"  {label:>15}  {n}")
            bad += m is True
        print(f"\n{bad} of {elves} need patching"
              if bad else "\nnothing to patch: nothing asks for an executable stack")
        return 0 if bad == 0 else 1

    # Mirror the whole directory, not just the libraries: Mosek.jl's build
    # determines the version by RUNNING `<bindir>/mosek` and parsing its
    # banner, so a copy holding only lib*.so* is rejected with
    # "does not point to a MOSEK <x.y> bin directory".
    os.makedirs(a.dst, exist_ok=True)
    patched = copied = links = nbytes = 0
    for n in names:
        src, dst = os.path.join(a.src, n), os.path.join(a.dst, n)
        if os.path.lexists(dst):
            shutil.rmtree(dst) if os.path.isdir(dst) and not os.path.islink(dst) else os.remove(dst)
        if os.path.islink(src):
            # keep the symlink chain (libmosek64.so -> libmosek64.so.10.2)
            os.symlink(os.readlink(src), dst)
            links += 1
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
            continue
        shutil.copy2(src, dst)
        os.chmod(dst, os.stat(dst).st_mode | 0o200)   # a read-only source copies read-only
        copied += 1
        nbytes += os.path.getsize(dst)
        if clear_execstack(dst):
            patched += 1
            print(f"  cleared executable stack: {n}")

    print(f"\ncopied {copied} files + {links} symlinks ({human(nbytes)}) to {a.dst}, "
          f"patched {patched}")
    mosekbin = os.path.join(a.dst, "mosek")
    if not os.path.isfile(mosekbin):
        print("warning: no 'mosek' executable in the copy -- Mosek.jl's build probes")
        print("         the version by running it and will reject this directory")
    if patched == 0:
        print("nothing was marked -- the load failure has another cause")
        return 1
    full = os.path.abspath(a.dst)
    print("\nnow point Mosek.jl at the patched copy and rebuild:")
    print(f'  export MOSEKBINDIR="{full}"')
    print(f'  export LD_LIBRARY_PATH="{full}:$LD_LIBRARY_PATH"')
    print("  \"$JULIA_BIN\" --project=. -e 'using Pkg; Pkg.build(\"Mosek\"); Pkg.precompile()'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
