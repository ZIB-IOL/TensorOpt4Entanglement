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

    python3 scripts/fix_execstack.py --check          # report, change nothing
    python3 scripts/fix_execstack.py                  # patch a local copy
"""
import argparse, os, shutil, struct, sys

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


def libs(d):
    return sorted(n for n in os.listdir(d) if ".so" in n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.environ.get("MOSEKBINDIR"),
                    help="MOSEK bin directory (default: $MOSEKBINDIR)")
    ap.add_argument("--dst", default=".mosek_bin",
                    help="where to put the patched copy (default: ./.mosek_bin)")
    ap.add_argument("--check", action="store_true",
                    help="report which libraries are marked, change nothing")
    a = ap.parse_args()

    if not a.src:
        sys.exit("no source directory: pass --src or set MOSEKBINDIR")
    if not os.path.isdir(a.src):
        sys.exit(f"not a directory: {a.src}")

    names = libs(a.src)
    if not names:
        sys.exit(f"no shared libraries in {a.src}")

    if a.check:
        bad = 0
        for n in names:
            p = os.path.join(a.src, n)
            if os.path.islink(p):
                print(f"  link  {n} -> {os.readlink(p)}")
                continue
            m = is_execstack(p)
            label = {True: "EXECSTACK", False: "ok", None: "not an ELF file"}[m]
            print(f"  {label:>15}  {n}")
            bad += m is True
        print(f"\n{bad} of {len(names)} need patching"
              if bad else "\nnothing to patch: no library asks for an executable stack")
        return 0 if bad == 0 else 1

    os.makedirs(a.dst, exist_ok=True)
    patched = copied = 0
    for n in names:
        src, dst = os.path.join(a.src, n), os.path.join(a.dst, n)
        if os.path.lexists(dst):
            os.remove(dst)
        if os.path.islink(src):
            # keep the symlink chain (libmosek64.so -> libmosek64.so.10.2)
            os.symlink(os.readlink(src), dst)
            continue
        shutil.copy2(src, dst)
        os.chmod(dst, os.stat(dst).st_mode | 0o200)   # a read-only source copies read-only
        copied += 1
        if clear_execstack(dst):
            patched += 1
            print(f"  cleared executable stack: {n}")

    print(f"\ncopied {copied} libraries to {a.dst}, patched {patched}")
    if patched == 0:
        print("no library was marked -- the load failure has another cause")
        return 1
    full = os.path.abspath(a.dst)
    print("\nnow point Mosek.jl at the patched copy and rebuild:")
    print(f'  export MOSEKBINDIR="{full}"')
    print(f'  export LD_LIBRARY_PATH="{full}:$LD_LIBRARY_PATH"')
    print("  \"$JULIA_BIN\" --project=. -e 'using Pkg; Pkg.build(\"Mosek\"); Pkg.precompile()'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
